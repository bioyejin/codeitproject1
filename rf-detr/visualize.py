# rf-detr/visualize.py
"""
RF-DETR 예측 수집, mAP 계산, 오답 시각화.
src/visualize.py의 로직(+ src/train.py의 evaluate_from_data)을 RF-DETR용으로 옮긴 것.
model.predict()가 supervision.Detections(.xyxy/.confidence/.class_id)를 반환한다는 점은
rf-detr 소스(rfdetr/detr.py의 'from supervision import Detections' 타입힌트)로 확인함.
"""
import os
import json
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision


def _xywh_to_xyxy(bbox):
    """COCO [x,y,w,h] -> [x1,y1,x2,y2] (dataset.py의 build_coco가 COCO 포맷으로 저장하므로 필요)."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _draw_box(ax, box, label_idx, label_to_category_id, color, prefix, score=None):
    """src/visualize.py의 _draw_box()와 동일 - 모델 무관 순수 시각화 로직."""
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)

    cat_id = label_to_category_id.get(label_idx, '?')
    text = f'{prefix}: {cat_id}'
    if score is not None:
        text += f' ({score:.2f})'

    ax.text(x1, y1 - 4, text, color=color, fontsize=7,
            bbox=dict(facecolor='black', alpha=0.5, pad=1, edgecolor='none'))


def _is_error(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold):
    """src/visualize.py의 _is_error()와 동일 - GT 누락 또는 예측 오탐 여부 판정."""
    if len(gt_boxes) == 0:
        return len(pred_boxes) > 0

    if len(pred_boxes) == 0:
        return True

    iou = box_iou(gt_boxes, pred_boxes)   # (num_gt, num_pred)

    matched_pred = set()
    for gt_idx in range(len(gt_boxes)):
        best_iou, best_pred_idx = iou[gt_idx].max(0)
        best_pred_idx = best_pred_idx.item()

        if (best_iou >= iou_threshold
                and pred_labels[best_pred_idx] == gt_labels[gt_idx]
                and best_pred_idx not in matched_pred):
            matched_pred.add(best_pred_idx)
        else:
            return True

    return len(matched_pred) < len(pred_boxes)


def collect_predictions_from_coco(model, coco_json_path, image_dir, score_threshold=0.0):
    """
    src/visualize.py의 collect_predictions() 로직(추론을 1회만 돌리고 결과를 캐싱해서
    이후 여러 threshold로 재시각화 가능하게 함)을 RF-DETR용으로 옮긴 것.
    DataLoader 대신 dataset.py가 만든 fold의 _annotations.coco.json + 이미지 폴더를
    직접 읽어서 추론합니다.

    Args:
        model: get_rfdetr_model()으로 만든 RF-DETR 모델 (학습된 가중치 로드된 상태)
        coco_json_path (str): fold의 '_annotations.coco.json' 경로
        image_dir (str): 위 json과 같은 폴더의 이미지 경로
        score_threshold (float): model.predict()에 전달할 최소 confidence

    Returns:
        list of dicts: [{'image': np.ndarray(RGB), 'gt_boxes': Tensor(xyxy), 'gt_labels': Tensor,
                          'pred_boxes': Tensor(xyxy), 'pred_labels': Tensor, 'pred_scores': Tensor}, ...]
    """
    with open(coco_json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    all_data = []
    for img_info in coco['images']:
        img_path = os.path.join(image_dir, img_info['file_name'])
        image = np.array(Image.open(img_path).convert('RGB'))

        anns = anns_by_image[img_info['id']]
        if anns:
            gt_boxes = torch.tensor([_xywh_to_xyxy(a['bbox']) for a in anns], dtype=torch.float32)
            gt_labels = torch.tensor([a['category_id'] for a in anns], dtype=torch.int64)
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.int64)

        detections = model.predict(image, threshold=score_threshold)
        pred_boxes = torch.tensor(np.array(detections.xyxy), dtype=torch.float32)
        pred_scores = torch.tensor(np.array(detections.confidence), dtype=torch.float32)
        pred_labels = torch.tensor(np.array(detections.class_id), dtype=torch.int64)

        all_data.append({
            'image': image,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'pred_boxes': pred_boxes,
            'pred_labels': pred_labels,
            'pred_scores': pred_scores,
        })

    return all_data


def evaluate_from_data(all_data, device='cpu'):
    """
    src/train.py의 evaluate_from_data() 로직을 그대로 가져옴.
    collect_predictions_from_coco()로 모아둔 예측/정답 데이터로 mAP를 계산합니다
    (모델 재추론 없음). RF-DETR의 metrics.csv와 무관하게 직접 torchmetrics로 계산하므로,
    mAP@0.75:0.95(5개 IoU 지점 평균)까지 원본과 완전히 동일하게 정확히 나옵니다.

    Args:
        all_data: collect_predictions_from_coco()의 반환값
        device (str): 'cuda' or 'cpu'

    Returns:
        dict: {'map', 'map_50', 'map_per_class', 'classes', 'map_75_95'}
    """
    metric_standard = MeanAveragePrecision(class_metrics=True)
    # IoU threshold 0.75~0.95, 0.05 간격 (COCO 기본 step과 동일한 방식) - 엄격한 기준
    metric_strict = MeanAveragePrecision(iou_thresholds=[0.75, 0.80, 0.85, 0.90, 0.95])

    for data in all_data:
        metric_targets = [{
            'boxes': data['gt_boxes'].to(device),
            'labels': data['gt_labels'].to(device),
        }]
        preds = [{
            'boxes': data['pred_boxes'].to(device),
            'labels': data['pred_labels'].to(device),
            'scores': data['pred_scores'].to(device),
        }]
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


def visualize_errors_from_data(all_data, label_to_category_id, save_dir,
                                score_threshold=0.5, iou_threshold=0.5, file_prefix='error'):
    """
    src/visualize.py의 visualize_errors_from_data() 로직을 그대로 가져옴.
    RF-DETR은 이미지를 정규화된 텐서가 아니라 원본 RGB 배열(0~255)로 다루므로
    mean/std 역정규화 단계만 제거함 (나머지 오차 판정/시각화 로직은 동일).

    Args:
        all_data: collect_predictions_from_coco()의 반환값
        label_to_category_id (dict): 모델 라벨 -> 원본 category_id 매핑
        save_dir (str): 오답 이미지 저장 폴더
        score_threshold (float): 예측 최소 confidence
        iou_threshold (float): GT-예측 매칭 IoU 기준
        file_prefix (str): 저장 파일명 접두어 (fold/sanity check 등 산출물 구분용)

    Returns:
        int: 저장된 오답 이미지 수
    """
    os.makedirs(save_dir, exist_ok=True)
    error_count = 0

    for idx, data in enumerate(all_data):
        gt_boxes = data['gt_boxes']
        gt_labels = data['gt_labels']
        pred_boxes = data['pred_boxes']
        pred_labels = data['pred_labels']
        pred_scores = data['pred_scores']

        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        if not _is_error(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold):
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(data['image'])

        for box, label in zip(gt_boxes, gt_labels):
            _draw_box(ax, box, label.item(), label_to_category_id, color='lime', prefix='GT')

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            _draw_box(ax, box, label.item(), label_to_category_id,
                      color='red', prefix='Pred', score=score.item())

        ax.set_title(f'Error image {idx}  (score_thr={score_threshold}, iou_thr={iou_threshold})',
                     fontsize=10)
        ax.axis('off')

        save_path = os.path.join(save_dir, f'{file_prefix}_{error_count:04d}_{idx:04d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        error_count += 1

    print(f'Saved {error_count} error images -> {save_dir}')
    return error_count


def visualize_errors(model, coco_json_path, image_dir, label_to_category_id, save_dir,
                      score_threshold=0.5, iou_threshold=0.5, file_prefix='error'):
    """
    src/visualize.py의 visualize_errors() 원스텝 래퍼와 동일한 역할.
    collect_predictions_from_coco() + visualize_errors_from_data()를 한 번에 실행합니다.

    mAP 계산까지 같은 예측 데이터로 처리해서 추론 중복을 피하고 싶다면, 이 래퍼 대신
    collect_predictions_from_coco()를 직접 호출해 evaluate_from_data()/
    visualize_errors_from_data()에 나눠 넘기는 쪽이 더 효율적입니다
    (utils.report_fold_result()가 그 방식을 씀).

    Returns:
        int: 저장된 오답 이미지 수
    """
    all_data = collect_predictions_from_coco(model, coco_json_path, image_dir, score_threshold=0.0)
    return visualize_errors_from_data(all_data, label_to_category_id, save_dir,
                                       score_threshold=score_threshold, iou_threshold=iou_threshold,
                                       file_prefix=file_prefix)
