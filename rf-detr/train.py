# rf-detr/train.py
"""RF-DETR 5-fold 학습 루프. 원본 rfdetr_train_5fold_colab.py의 [3] 블록을 함수로 분리한 것."""
import os
import shutil

import yaml
import torch

from model import get_rfdetr_model
from dataset import load_label_map
from utils import read_metrics_csv, plot_history, report_fold_result, summarize_kfold_results


def load_config(path):
    """yaml config 파일을 읽어 dict로 반환합니다."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_fold(fold_idx, dataset_dir, model_variant, model_tag, train_cfg,
               local_output_dir, backup_dir):
    """
    fold 하나를 학습하고 best 체크포인트를 backup_dir에 복사합니다.
    backup 파일이 이미 있으면 학습을 건너뜁니다 (이어하기).

    Args:
        fold_idx (int): fold 번호 (0-indexed)
        dataset_dir (str): fold별 데이터 루트 ('{dataset_dir}/fold{fold_idx}' 사용)
        model_variant (str): 'small' | 'medium' 등 RF-DETR 변형
        model_tag (str): 실험명 태그 (체크포인트 파일명에 사용)
        train_cfg (dict): config.yaml의 train 섹션
        local_output_dir (str): fold 학습 중 임시 산출물을 저장할 로컬 경로
        backup_dir (str): best 체크포인트를 백업할 경로

    Returns:
        str or None: 백업된 체크포인트 경로 (백업 실패 시 None)
    """
    exp = f'{model_tag}_fold{fold_idx}'
    dst = os.path.join(backup_dir, f'{exp}_best.pth')

    if os.path.exists(dst):
        print(f'[fold {fold_idx}] 백업 존재 → 건너뜀')
        return dst

    out = os.path.join(local_output_dir, exp)
    os.makedirs(out, exist_ok=True)
    print(f"\n{'='*50}\n[fold {fold_idx}] 학습 시작\n{'='*50}")

    model = get_rfdetr_model(model_variant)
    model.train(
        dataset_dir=os.path.join(dataset_dir, f'fold{fold_idx}'),
        output_dir=out,
        epochs=train_cfg['epochs'],
        batch_size=train_cfg['batch_size'],
        grad_accum_steps=train_cfg['grad_accum_steps'],
        lr=train_cfg['lr'],
        lr_encoder=train_cfg['lr_encoder'],
        weight_decay=train_cfg['weight_decay'],
        lr_scheduler=train_cfg['lr_scheduler'],
        warmup_epochs=train_cfg['warmup_epochs'],
        lr_min_factor=train_cfg['lr_min_factor'],
        early_stopping=train_cfg['early_stopping'],
        early_stopping_patience=train_cfg['early_stopping_patience'],
        early_stopping_min_delta=train_cfg['early_stopping_min_delta'],
        tensorboard=train_cfg['tensorboard'],
        seed=train_cfg.get('seed'),
    )

    os.makedirs(backup_dir, exist_ok=True)
    src = os.path.join(out, 'checkpoint_best_total.pth')
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f'[fold {fold_idx}] best 백업 → {dst}')
    else:
        dst = None
        print(f'[fold {fold_idx}] checkpoint_best_total.pth 없음 — 백업 실패')

    metrics_csv = os.path.join(out, 'metrics.csv')
    if os.path.exists(metrics_csv):
        history = read_metrics_csv(out)
        plot_history(history, title=f'{model_tag} - Fold {fold_idx}',
                     save_path=os.path.join(backup_dir, f'{exp}_history.png'))
    else:
        print(f'[fold {fold_idx}] metrics.csv 없음 — 학습 곡선 생략')

    del model
    torch.cuda.empty_cache()
    print(f'[fold {fold_idx}] 완료')
    return dst


def run_kfold(config, max_folds=None):
    """
    config에 정의된 n_splits만큼 fold를 순회하며 train_fold를 실행하고,
    fold마다 report_fold_result()(클래스별 mAP 출력 + 오답 시각화)를 자동으로 돌린 뒤
    마지막에 summarize_kfold_results()로 5-fold 평균±표준편차를 출력합니다.

    Args:
        config (dict): load_config()의 반환값
        max_folds (int): 실행할 최대 fold 수 (None이면 전체, sanity check용)

    Returns:
        list: fold별 백업 체크포인트 경로 (길이 = 실행한 fold 수)
    """
    print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

    data_cfg = config['data']
    n_folds = max_folds if max_folds is not None else data_cfg['n_splits']
    label_to_category_id = load_label_map(data_cfg['dataset_dir'])['label2cat']

    results = []
    fold_metrics = []
    for fi in range(n_folds):
        dst = train_fold(
            fold_idx=fi,
            dataset_dir=data_cfg['dataset_dir'],
            model_variant=config['model']['variant'],
            model_tag=config['model']['tag'],
            train_cfg=config['train'],
            local_output_dir=config['output']['local_output_dir'],
            backup_dir=config['output']['backup_dir'],
        )
        results.append(dst)

        if dst is None:
            print(f'[fold {fi}] 체크포인트 없음 — 리포팅 생략')
            continue

        vis_dir = os.path.join(config['output']['backup_dir'], f"{config['model']['tag']}_fold{fi}_errors")
        metrics = report_fold_result(
            fold_idx=fi,
            checkpoint_path=dst,
            model_variant=config['model']['variant'],
            dataset_dir=data_cfg['dataset_dir'],
            label_to_category_id=label_to_category_id,
            vis_dir=vis_dir,
        )
        fold_metrics.append(metrics)

    if fold_metrics:
        summarize_kfold_results(fold_metrics)

    print(f'\n▶ {n_folds}폴드 학습 완료')
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                         default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    parser.add_argument('--max_folds', type=int, default=None,
                         help='sanity check용: 실행할 fold 수 제한')
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_kfold(cfg, max_folds=args.max_folds)
