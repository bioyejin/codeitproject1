# src/train.py
import os
import json
import yaml
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

from src.dataset import download_data, PillDataset, FoldDataset, collate_fn_list, collate_fn_stack, coco_to_xyxy

def prepare_targets(targets, device, box_format):
    """targets를 device로 이동하고, box_format에 따라 bbox를 변환합니다."""
    if box_format == 'xyxy':
        return [
            {'boxes': coco_to_xyxy(t['boxes']).to(device), 'labels': t['labels'].to(device)}
            for t in targets
        ]
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


from src.model import get_model
from src.utils import get_groups, set_seed, save_checkpoint, plot_history, load_checkpoint, get_category_names
from src.visualize import collect_predictions, visualize_errors_from_data


def load_data(unique_only=True, batch_size=4, collate='stack', seed=42):
    """
    데이터를 다운로드하고 단일 fold의 train/val DataLoader를 생성합니다.
    GroupKFold는 셔플을 하지 않으므로, 그룹 순서를 미리 무작위로 섞어서
    비슷한 촬영조건의 데이터가 특정 fold에만 몰리는 것을 방지합니다.

    Args:
        unique_only (bool): 고유 이미지만 사용할지 여부
        batch_size (int): 배치 크기
        collate (str): 'list' 또는 'stack' - 모델 입력 형태에 맞게 선택
        seed (int): 셔플 시드

    Returns:
        train_loader, val_loader
        label_to_category_id (dict): 모델 출력 라벨(1~N) → 원본 category_id(dl_idx) 매핑
    """
    # 1. 데이터 다운로드
    path = download_data()
    print(f"데이터 경로: {path}")

    # 2. 전체 데이터셋 생성
    full_dataset = PillDataset(path, train=True, unique_only=unique_only)
    val_full_dataset = PillDataset(path, train=False, unique_only=unique_only)
    print(f"전체 이미지 수: {len(full_dataset)}")

    label_to_category_id = full_dataset.label_to_category_id

    # 3. group 정보 추출
    groups = get_groups(full_dataset.image_names)
    print(f"고유 group 수: {len(set(groups))}")

    # 4. 인덱스를 셔플해서 데이터 순서로 인한 편향을 제거
    indices = list(range(len(full_dataset)))
    random.seed(seed)
    random.shuffle(indices)
    shuffled_groups = [groups[i] for i in indices]

    # 5. GroupKFold로 train/val 분리 (첫 번째 fold만 사용)
    gkf = GroupKFold(n_splits=5)
    train_pos, val_pos = next(gkf.split(indices, groups=shuffled_groups))

    train_idx = [indices[i] for i in train_pos]
    val_idx = [indices[i] for i in val_pos]
    print(f"train: {len(train_idx)}장, val: {len(val_idx)}장")

    # 6. train/val Subset 생성
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(val_full_dataset, val_idx)

    # 7. DataLoader 생성
    collate_func = collate_fn_list if collate == 'list' else collate_fn_stack

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_func
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_func
    )

    return train_loader, val_loader, label_to_category_id


def load_data_kfold(unique_only=True, batch_size=4, n_splits=5, collate='stack', seed=42):
    """
    K-Fold로 분리된 모든 train/val DataLoader 쌍을 생성합니다.
    GroupKFold는 셔플을 하지 않으므로, 그룹 순서를 미리 무작위로 섞어서
    비슷한 촬영조건의 데이터가 특정 fold에만 몰리는 것을 방지합니다.

    Args:
        unique_only (bool): 고유 이미지만 사용할지 여부
        batch_size (int): 배치 크기
        n_splits (int): fold 수
        collate (str): 'list' 또는 'stack' - 모델 입력 형태에 맞게 선택
        seed (int): 셔플 시드

    Returns:
        path (str): 데이터 경로
        fold_loaders (list): [(train_loader, val_loader), ...] 길이 n_splits
        label_to_category_id (dict): 모델 출력 라벨(1~N) → 원본 category_id(dl_idx) 매핑
    """
    path = download_data()
    print(f"데이터 경로: {path}")

    full_dataset = PillDataset(path, train=True, unique_only=unique_only)
    val_full_dataset = PillDataset(path, train=False, unique_only=unique_only)
    print(f"전체 이미지 수: {len(full_dataset)}")

    label_to_category_id = full_dataset.label_to_category_id

    groups = get_groups(full_dataset.image_names)
    print(f"고유 group 수: {len(set(groups))}")

    # 인덱스를 셔플해서 데이터 순서로 인한 편향을 제거
    indices = list(range(len(full_dataset)))
    random.seed(seed)
    random.shuffle(indices)

    shuffled_groups = [groups[i] for i in indices]

    collate_func = collate_fn_list if collate == 'list' else collate_fn_stack

    gkf = GroupKFold(n_splits=n_splits)
    fold_loaders = []

    for fold, (train_pos, val_pos) in enumerate(gkf.split(indices, groups=shuffled_groups)):
        # train_pos, val_pos는 셔플된 indices 안에서의 위치 → 원래 인덱스로 변환
        train_idx = [indices[i] for i in train_pos]
        val_idx = [indices[i] for i in val_pos]

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(val_full_dataset, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_func
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=collate_func
        )

        fold_loaders.append((train_loader, val_loader))
        print(f"Fold {fold+1}: train {len(train_idx)}장, val {len(val_idx)}장")

    return path, fold_loaders, label_to_category_id


def get_optimizer_and_scheduler(model, lr=1e-4, weight_decay=1e-4, warmup=False, total_epochs=20, steps_per_epoch=1):
    """
    옵티마이저와 scheduler를 생성합니다.

    warmup=True:
        - 첫 epoch: LinearLR로 lr*0.01 → lr 선형 증가 (step 단위)
        - 이후 epoch: CosineAnnealingLR epoch 단위
        - CosineAnnealingLR은 warmup epoch 완료 후 train_model 내부에서 생성합니다.
          (PyTorch scheduler는 생성 시 step()을 즉시 호출하므로,
           동시 생성하면 CosineAnnealingLR이 LR을 lr로 덮어써 warmup이 망가집니다.)
    warmup=False:
        - 처음부터 CosineAnnealingLR epoch 단위

    Args:
        model: 학습할 모델
        lr (float): 기본 learning rate
        weight_decay (float): AdamW weight decay
        warmup (bool): 첫 epoch warmup 적용 여부
        total_epochs (int): 전체 epoch 수
        steps_per_epoch (int): 한 epoch당 step(batch) 수 (warmup 시 사용)

    Returns:
        optimizer, warmup_scheduler, main_scheduler
        warmup_scheduler: 첫 epoch step 단위용 (warmup=False면 None)
        main_scheduler:   epoch 단위 CosineAnnealingLR (warmup=True면 None, train_model 내부에서 생성)
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if warmup:
        # 생성 즉시 LR = lr * 0.01로 설정됨
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=steps_per_epoch)
        main_scheduler = None  # train_model 내부에서 warmup 완료 후 생성
    else:
        warmup_scheduler = None
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    return optimizer, warmup_scheduler, main_scheduler


def train_one_epoch(model, train_loader, optimizer, device, box_format='xyxy', step_scheduler=None):
    """
    한 epoch 동안 모델을 학습시킵니다.

    Args:
        model: 학습할 모델
        train_loader: 학습 DataLoader
        optimizer: 옵티마이저
        device: 'cuda' or 'cpu'
        box_format (str): 'xyxy'면 COCO bbox를 [x1,y1,x2,y2]로 변환, 'coco'면 그대로
        step_scheduler: 배치마다 step()할 scheduler (warmup epoch에만 전달, 나머지는 None)

    Returns:
        float: epoch 평균 손실
    """
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(train_loader, desc="Training"):
        images = [img.to(device) for img in images]
        targets = prepare_targets(targets, device, box_format)

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_scheduler is not None:
            step_scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """
    검증 데이터셋에 대해 mAP를 계산합니다.
    표준 mAP@0.5:0.95, mAP@0.5와 함께
    엄격한 기준인 mAP@0.75:0.95(IoU 0.75~0.95 구간 평균)도 함께 계산합니다.

    Args:
        model: 평가할 모델
        val_loader: 검증 DataLoader
        device: 'cuda' or 'cpu'

    Returns:
        dict: {'map', 'map_50', 'map_75_95', 'map_per_class', ...}
    """
    model.eval()

    metric_standard = MeanAveragePrecision(class_metrics=True)
    # IoU threshold를 0.75~0.95, 0.05 간격으로 5개 지정 (COCO 기본 step과 동일한 방식)
    metric_strict = MeanAveragePrecision(iou_thresholds=[0.75, 0.80, 0.85, 0.90, 0.95])

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images = [img.to(device) for img in images]

            # torchmetrics는 항상 xyxy 형식을 기대하므로 box_format과 무관하게 변환
            metric_targets = [
                {'boxes': coco_to_xyxy(t['boxes']).to(device), 'labels': t['labels'].to(device)}
                for t in targets
            ]

            preds = model(images)
            metric_standard.update(preds, metric_targets)
            metric_strict.update(preds, metric_targets)

    result_standard = metric_standard.compute()
    result_strict = metric_strict.compute()

    return {
        'map': result_standard['map'].item(),
        'map_50': result_standard['map_50'].item(),
        'map_per_class': result_standard.get('map_per_class'),
        'classes': result_standard.get('classes'),
        'map_75_95': result_strict['map'].item(),
    }


def train_model(model, train_loader, val_loader, optimizer, warmup_scheduler, main_scheduler, device, epochs, save_path, box_format='xyxy'):
    """
    여러 epoch을 학습시키고 history를 기록하며, val mAP 기준 best model을 저장합니다.

    LR 스케줄 흐름:
        warmup_scheduler가 있는 경우 (warmup=True):
            - epoch 1: LinearLR step 단위 warmup → lr*0.01 에서 lr까지 선형 증가
            - epoch 2~: CosineAnnealingLR(T_max=epochs-1) epoch 단위
            - CosineAnnealingLR은 warmup 완료 직후 내부에서 생성 (LR 충돌 방지)
        warmup_scheduler가 없는 경우 (warmup=False):
            - epoch 1~: CosineAnnealingLR(T_max=epochs) epoch 단위

    Args:
        model: 학습할 모델
        train_loader, val_loader: DataLoader
        optimizer: 옵티마이저
        warmup_scheduler: 첫 epoch step 단위 warmup용 (없으면 None)
        main_scheduler: epoch 단위 CosineAnnealingLR (warmup=True면 None으로 전달)
        device: 'cuda' or 'cpu'
        epochs (int): 학습 epoch 수
        save_path (str): best model 저장 경로
        box_format (str): 'xyxy' 또는 'coco'

    Returns:
        dict: {'train_loss': [...], 'val_map': [...], 'val_map_50': [...], 'val_map_75_95': [...]}
    """
    history = {'train_loss': [], 'val_map': [], 'val_map_50': [], 'val_map_75_95': []}
    best_map = -1.0

    for epoch in range(epochs):
        is_warmup_epoch = (epoch == 0 and warmup_scheduler is not None)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            box_format=box_format,
            step_scheduler=warmup_scheduler if is_warmup_epoch else None
        )

        if is_warmup_epoch:
            # warmup 완료 시점: LR = lr (LinearLR이 lr까지 올림)
            # 이 시점에 생성해야 CosineAnnealingLR init step()이 LR을 lr로 세팅 → 충돌 없음
            main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 1)
        else:
            main_scheduler.step()

        result = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_map'].append(result['map'])
        history['val_map_50'].append(result['map_50'])
        history['val_map_75_95'].append(result['map_75_95'])

        if result['map_75_95'] > best_map:
            best_map = result['map_75_95']
            save_checkpoint(model, save_path)
            marker = "   [Best] mAP@0.75:0.95"
        else:
            marker = ""

        print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | mAP: {result['map']:.4f} | mAP@50: {result['map_50']:.4f} | mAP@75:95: {result['map_75_95']:.4f}{marker}")

    return history


def evaluate_from_data(all_data, device):
    """
    collect_predictions()으로 수집한 데이터로 mAP를 계산합니다 (추론 없음).
    evaluate()와 동일한 결과를 반환하지만 모델을 다시 실행하지 않습니다.

    Args:
        all_data: collect_predictions()의 반환값
        device: 'cuda' or 'cpu'

    Returns:
        dict: {'map', 'map_50', 'map_75_95', 'map_per_class', 'classes'}
    """
    metric_standard = MeanAveragePrecision(class_metrics=True)
    metric_strict = MeanAveragePrecision(iou_thresholds=[0.75, 0.80, 0.85, 0.90, 0.95])

    for data in all_data:
        metric_targets = [{'boxes':  data['gt_boxes'].to(device),
                           'labels': data['gt_labels'].to(device)}]
        preds = [{'boxes':  data['pred_boxes'].to(device),
                  'labels': data['pred_labels'].to(device),
                  'scores': data['pred_scores'].to(device)}]
        metric_standard.update(preds, metric_targets)
        metric_strict.update(preds, metric_targets)

    result_standard = metric_standard.compute()
    result_strict   = metric_strict.compute()

    return {
        'map':           result_standard['map'].item(),
        'map_50':        result_standard['map_50'].item(),
        'map_per_class': result_standard.get('map_per_class'),
        'classes':       result_standard.get('classes'),
        'map_75_95':     result_strict['map'].item(),
    }


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_kfold(config_path, max_folds=None, override_epochs=None):
    """
    config.yaml을 불러와 K-Fold 전체를 학습하고, fold별 결과를 종합합니다.

    Args:
        config_path (str): config.yaml 경로
        max_folds (int): 실행할 최대 fold 수 (None이면 전체 fold 실행, sanity check용)

    Returns:
        list: 각 fold의 Best mAP@0.75:0.95 리스트
    """
    config = load_config(config_path)
    if override_epochs is not None:
        config['train']['epochs'] = override_epochs
    set_seed(config['train']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    path, fold_loaders, label_to_category_id = load_data_kfold(   # ← 변수 3개로 받기
        unique_only=config['data']['unique_only'],
        batch_size=config['data']['batch_size'],
        n_splits=config['data']['n_splits'],
        collate=config['data']['collate'],
        seed=config['train']['seed']
    )

    if max_folds is not None:
        fold_loaders = fold_loaders[:max_folds]  # ← sanity check 시 일부만 실행

    category_map = get_category_names(path)   # category_id(dl_idx) → 이름

    model_name = config['model']['name']
    all_fold_results = []

    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n{'='*50}\nFold {fold+1}/{len(fold_loaders)} 시작 ({model_name})\n{'='*50}")

        model = get_model(
            model_name,
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            freeze_backbone=config['model']['freeze_backbone']
        ).to(device)

        optimizer, warmup_scheduler, main_scheduler = get_optimizer_and_scheduler(
            model,
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'],
            warmup=config['train']['warmup'],
            total_epochs=config['train']['epochs'],
            steps_per_epoch=len(train_loader)
        )

        save_path = os.path.join(config['output']['save_dir'], f"{model_name}_fold{fold+1}.pt")

        history = train_model(
            model, train_loader, val_loader, optimizer, warmup_scheduler, main_scheduler,
            device, epochs=config['train']['epochs'], save_path=save_path,
            box_format=config['model']['box_format']
        )

        plot_history(history, title=f"{model_name} - Fold {fold+1}",
                     save_path=os.path.join(config['output']['save_dir'], f"{model_name}_fold{fold+1}_history.png"))

        # best model을 다시 불러와서 클래스별 mAP 계산
        best_model = get_model(
            model_name,
            num_classes=config['model']['num_classes'],
            pretrained=False,   # 가중치는 체크포인트에서 불러올 거라 사전학습 불필요
            freeze_backbone=config['model']['freeze_backbone']
        ).to(device)
        best_model = load_checkpoint(best_model, save_path, device=device)

        # best model 추론 1회 실행 → mAP 계산 + 시각화에 공유
        pred_data = collect_predictions(best_model, val_loader, device)

        per_class_result = evaluate_from_data(pred_data, device)

        print(f"\n[Fold {fold+1}] 클래스별 mAP (best epoch 기준)")
        for cls, ap in zip(per_class_result['classes'], per_class_result['map_per_class']):
            label = cls.item()
            cat_id = label_to_category_id.get(label)
            cls_name = category_map.get(cat_id, '?') if cat_id is not None else '?'
            cls_name = cls_name.replace('\xa0', ' ')
            print(f"  {cls_name}({cat_id}): {ap.item():.4f}")

        vis_dir = os.path.join(config['output']['save_dir'], f"{model_name}_fold{fold+1}_errors")
        visualize_errors_from_data(pred_data, label_to_category_id, save_dir=vis_dir)

        best_map_75_95 = max(history['val_map_75_95'])
        all_fold_results.append(best_map_75_95)
        print(f"Fold {fold+1} 완료 | Best mAP@0.75:0.95: {best_map_75_95:.4f}")

    avg_map = np.mean(all_fold_results)
    std_map = np.std(all_fold_results)
    print(f"\n{'='*50}\n{model_name} 최종 결과 (5-fold 평균)\nmAP@0.75:0.95: {avg_map:.4f} ± {std_map:.4f}\n{'='*50}")

    return all_fold_results


def run_kfold_from_dir(dataset_dir, config_path, max_folds=None, override_epochs=None):
    """
    Pre-split fold 디렉토리에서 K-Fold 학습 (Colab + Google Drive 워크플로우용).
    dataset_dir/fold{i}/train·valid/_annotations.coco.json 구조를 사용합니다.

    Args:
        dataset_dir (str): unzip된 dataset 루트 경로 (예: '/content/dataset')
        config_path (str): config yaml 경로
        max_folds (int): 실행할 최대 fold 수 (None이면 전체, sanity check용)
        override_epochs (int): config epochs 덮어쓰기 (sanity check용)

    Returns:
        list: 각 fold의 Best mAP@0.75:0.95 리스트
    """
    config = load_config(config_path)
    if override_epochs is not None:
        config['train']['epochs'] = override_epochs
    set_seed(config['train']['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    with open(os.path.join(dataset_dir, 'label_map.json'), 'r') as f:
        label_map = json.load(f)
    label_to_category_id = {int(k): v for k, v in label_map['label2cat'].items()}

    model_name = config['model']['name']
    collate_func = collate_fn_list if config['data']['collate'] == 'list' else collate_fn_stack
    batch_size = config['data']['batch_size']
    n_splits = config['data']['n_splits']
    n_folds = min(max_folds, n_splits) if max_folds is not None else n_splits

    all_fold_results = []
    all_per_class    = []   # fold별 {label: ap} 딕셔너리 리스트

    for fold in range(n_folds):
        print(f"\n{'='*50}\nFold {fold+1}/{n_folds} 시작 ({model_name})\n{'='*50}")

        train_dir = os.path.join(dataset_dir, f'fold{fold}', 'train')
        valid_dir = os.path.join(dataset_dir, f'fold{fold}', 'valid')

        train_dataset = FoldDataset(train_dir, train=True)
        val_dataset   = FoldDataset(valid_dir, train=False)
        print(f"train: {len(train_dataset)}장, val: {len(val_dataset)}장")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=2, collate_fn=collate_func)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                                  num_workers=2, collate_fn=collate_func)

        model = get_model(
            model_name,
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            freeze_backbone=config['model']['freeze_backbone']
        ).to(device)

        optimizer, warmup_scheduler, main_scheduler = get_optimizer_and_scheduler(
            model,
            lr=config['train']['lr'],
            weight_decay=config['train']['weight_decay'],
            warmup=config['train']['warmup'],
            total_epochs=config['train']['epochs'],
            steps_per_epoch=len(train_loader)
        )

        save_path = os.path.join(config['output']['save_dir'], f"{model_name}_fold{fold+1}.pt")

        history = train_model(
            model, train_loader, val_loader, optimizer, warmup_scheduler, main_scheduler,
            device, epochs=config['train']['epochs'], save_path=save_path,
            box_format=config['model']['box_format']
        )

        plot_history(history, title=f"{model_name} - Fold {fold+1}",
                     save_path=os.path.join(config['output']['save_dir'], f"{model_name}_fold{fold+1}_history.png"))

        best_model = get_model(
            model_name,
            num_classes=config['model']['num_classes'],
            pretrained=False,
            freeze_backbone=config['model']['freeze_backbone']
        ).to(device)
        best_model = load_checkpoint(best_model, save_path, device=device)

        pred_data = collect_predictions(best_model, val_loader, device)
        per_class_result = evaluate_from_data(pred_data, device)

        print(f"\n[Fold {fold+1}] 클래스별 mAP (best epoch 기준)")
        fold_class_ap = {}
        for cls, ap in zip(per_class_result['classes'], per_class_result['map_per_class']):
            label = cls.item()
            cat_id = label_to_category_id.get(label, '?')
            ap_val = ap.item()
            fold_class_ap[label] = ap_val
            print(f"  category {cat_id}: {ap_val:.4f}")
        all_per_class.append(fold_class_ap)

        vis_dir = os.path.join(config['output']['save_dir'], f"{model_name}_fold{fold+1}_errors")
        visualize_errors_from_data(pred_data, label_to_category_id, save_dir=vis_dir)

        best_map_75_95 = max(history['val_map_75_95'])
        all_fold_results.append(best_map_75_95)
        print(f"Fold {fold+1} 완료 | Best mAP@0.75:0.95: {best_map_75_95:.4f}")

    avg_map = np.mean(all_fold_results)
    std_map = np.std(all_fold_results)
    print(f"\n{'='*50}\n{model_name} 최종 결과 ({n_folds}-fold 평균)\nmAP@0.75:0.95: {avg_map:.4f} ± {std_map:.4f}\n{'='*50}")

    return {'fold_results': all_fold_results, 'per_class': all_per_class}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--max_folds', type=int, default=None, help='sanity check용: 실행할 fold 수 제한')
    parser.add_argument('--epochs', type=int, default=None, help='config epochs 덮어쓰기 (sanity check용)')
    args = parser.parse_args()

    run_kfold(args.config, max_folds=args.max_folds, override_epochs=args.epochs)