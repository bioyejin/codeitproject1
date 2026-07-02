# rf-detr/utils.py
"""
학습 곡선(history) 구성, fold별/5-fold 리포팅 자동화.
src/utils.py(plot_history)와 src/train.py(run_kfold 말미의 리포팅 로직)를 RF-DETR용으로 옮긴 것.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import get_rfdetr_model
from visualize import collect_predictions_from_coco, evaluate_from_data, visualize_errors


def read_metrics_csv(output_dir, use_ema=False):
    """
    RF-DETR가 학습 중 자동으로 남기는 {output_dir}/metrics.csv를 읽어
    plot_history()가 기대하는 형태의 history dict로 변환합니다.

    확인된 실제 컬럼(rf-detr 소스 tests/training/test_metrics_csv.py 기준):
      train/loss, train/lr, val/loss, val/mAP_50, val/mAP_50_95, val/mAP_75, val/mAR
      (use_ema=True면 val/ema_mAP_50 등 EMA 버전도 존재)

    주의: src/train.py가 쓰던 'mAP@0.75:0.95'(IoU 0.75~0.95 5개 지점 평균)는 RF-DETR이
    epoch마다 기본 제공하지 않음. 여기서는 RF-DETR이 기본 제공하는 'val/mAP_75'
    (IoU=0.75 단일 지점)를 학습 곡선용 3번째 지표로 대신 씀 - 다른 모델(swin_fpn/rt_detr)의
    mAP@0.75:0.95와는 다른 정의이니 절대값을 직접 비교하면 안 됨. 정확한 mAP@0.75:0.95는
    fold 최종 결과(report_fold_result/summarize_kfold_results, evaluate_from_data 기반)
    에서만 원본과 동일하게 계산됨.

    Args:
        output_dir (str): model.train(output_dir=...)에 넘긴 경로
        use_ema (bool): EMA 가중치 기준 지표를 쓸지 여부

    Returns:
        dict: {'train_loss', 'val_map', 'val_map_50', 'val_map_75'}
    """
    df = pd.read_csv(os.path.join(output_dir, 'metrics.csv'))
    prefix = 'ema_' if use_ema else ''

    def per_epoch(col):
        # 같은 epoch에 여러 행(step)이 있고 컬럼별 로깅 시점이 달라 NaN이 섞여 있으므로
        # epoch 단위로 묶어서 각 epoch의 마지막 유효값만 취함
        return df.groupby('epoch')[col].last().dropna().tolist()

    return {
        'train_loss': per_epoch('train/loss'),
        'val_map': per_epoch(f'val/{prefix}mAP_50_95'),
        'val_map_50': per_epoch(f'val/{prefix}mAP_50'),
        'val_map_75': per_epoch(f'val/{prefix}mAP_75'),
    }


def plot_history(history, title='Training History', save_path=None):
    """
    src/utils.py의 plot_history() 로직을 그대로 가져오되, 3번째 지표를
    'val_map_75_95' -> 'val_map_75'(RF-DETR 기본 제공, IoU=0.75 단일 지점)로 바꿈.
    read_metrics_csv()가 만든 history dict를 그대로 넣으면 됨.

    Args:
        history (dict): {'train_loss': [...], 'val_map': [...], 'val_map_50': [...], 'val_map_75': [...]}
        title (str): 그래프 제목
        save_path (str): 저장 경로 (None이면 화면에 표시만)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], marker='o', markersize=3, color='steelblue')
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    axes[1].plot(history['val_map'], marker='o', markersize=3, color='coral', label='mAP@0.5:0.95')
    axes[1].plot(history['val_map_50'], marker='o', markersize=3, color='seagreen', label='mAP@0.5')
    axes[1].plot(history['val_map_75'], marker='o', markersize=3, color='purple',
                 label='mAP@0.75 (RF-DETR native, 참고용 - 다른 모델의 0.75:0.95 평균과 다른 지표)')

    best_epoch = int(np.argmax(history['val_map_75'])) + 1
    axes[1].axvline(x=best_epoch - 1, color='gray', linestyle='--', label=f'Best epoch {best_epoch}')
    axes[1].set_title('Validation mAP')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].legend(fontsize=7)
    axes[1].grid(True)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def print_per_class_map(metrics, label_to_category_id):
    """클래스별 mAP를 출력합니다 (src/train.py의 run_kfold() 콘솔 출력 로직과 동일)."""
    for cls, ap in zip(metrics['classes'], metrics['map_per_class']):
        label = cls.item() if hasattr(cls, 'item') else int(cls)
        cat_id = label_to_category_id.get(label, '?')
        ap_val = ap.item() if hasattr(ap, 'item') else float(ap)
        print(f'  category {cat_id}: {ap_val:.4f}')


def report_fold_result(fold_idx, checkpoint_path, model_variant, dataset_dir,
                        label_to_category_id, vis_dir, score_threshold=0.5, iou_threshold=0.5):
    """
    fold 하나에 대해 mAP 계산 + 클래스별 mAP 출력 + 오답 이미지 시각화를 자동으로 수행합니다.
    train.run_kfold()의 fold 루프 안에서 각 fold 학습 직후 호출됩니다.

    mAP 계산(collect_predictions_from_coco 1회)과 오답 시각화(visualize_errors 원스텝
    래퍼 내부에서 collect_predictions_from_coco 재호출)가 분리되어 있어 추론이 2번
    수행됩니다. fold당 1회씩만 실행되니 비용 부담은 작고, 코드를 단순하게 유지하기 위한
    선택입니다.

    Args:
        fold_idx (int): fold 번호 (0-indexed)
        checkpoint_path (str): train_fold()가 반환한 best 체크포인트 경로
        model_variant (str): 'small' | 'medium' 등
        dataset_dir (str): fold 디렉토리들의 루트
        label_to_category_id (dict): 모델 라벨 -> 원본 category_id 매핑
        vis_dir (str): 오답 이미지 저장 폴더
        score_threshold (float): 오답 판정 시 예측 최소 confidence
        iou_threshold (float): GT-예측 매칭 IoU 기준

    Returns:
        dict: evaluate_from_data()의 반환값 (map, map_50, map_75_95 등 - 5-fold 집계용)
    """
    valid_dir = os.path.join(dataset_dir, f'fold{fold_idx}', 'valid')
    coco_json_path = os.path.join(valid_dir, '_annotations.coco.json')

    model = get_rfdetr_model(model_variant, checkpoint_path=checkpoint_path)

    pred_data = collect_predictions_from_coco(model, coco_json_path, valid_dir, score_threshold=0.0)
    metrics = evaluate_from_data(pred_data)

    print(f"\n[Fold {fold_idx}] mAP@0.5:0.95={metrics['map']:.4f}  "
          f"mAP@0.5={metrics['map_50']:.4f}  mAP@0.75:0.95={metrics['map_75_95']:.4f}")
    print(f'[Fold {fold_idx}] 클래스별 mAP')
    print_per_class_map(metrics, label_to_category_id)

    saved = visualize_errors(model, coco_json_path, valid_dir, label_to_category_id, vis_dir,
                              score_threshold=score_threshold, iou_threshold=iou_threshold,
                              file_prefix=f'fold{fold_idx}_error')
    print(f'[Fold {fold_idx}] 오답 이미지 {saved}장 저장 -> {vis_dir}')

    return metrics


def summarize_kfold_results(fold_metrics):
    """
    fold별 evaluate_from_data() 결과 리스트를 받아 평균±표준편차를 출력합니다.
    src/train.py의 run_kfold() 말미와 동일하게 mAP@0.75:0.95를 선정 기준으로 강조합니다.

    Args:
        fold_metrics (list): report_fold_result()가 fold마다 반환한 dict 리스트

    Returns:
        dict: {'map': (mean, std), 'map_50': (mean, std), 'map_75_95': (mean, std)}
    """
    def agg(key):
        vals = [m[key] for m in fold_metrics]
        return float(np.mean(vals)), float(np.std(vals))

    map_mean, map_std = agg('map')
    map50_mean, map50_std = agg('map_50')
    strict_mean, strict_std = agg('map_75_95')

    print(f"\n{'='*50}\n{len(fold_metrics)}-fold 결과 요약\n"
          f"mAP@0.5:0.95:  {map_mean:.4f} ± {map_std:.4f}\n"
          f"mAP@0.5:       {map50_mean:.4f} ± {map50_std:.4f}\n"
          f"mAP@0.75:0.95: {strict_mean:.4f} ± {strict_std:.4f}  (선정 기준 지표)\n{'='*50}")

    return {'map': (map_mean, map_std), 'map_50': (map50_mean, map50_std), 'map_75_95': (strict_mean, strict_std)}
