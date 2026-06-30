# src/visualize.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou

from tqdm import tqdm
from src.dataset import coco_to_xyxy

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])


def collect_predictions(model, val_loader, device):
    """
    Run inference once and return all prediction data.
    Pass the result to visualize_errors_from_data to re-visualize
    with different thresholds without re-running inference.

    Returns:
        list of dicts: [{'image', 'gt_boxes', 'gt_labels',
                          'pred_boxes', 'pred_labels', 'pred_scores'}, ...]
    """
    model.eval()
    all_data = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Collecting predictions"):
            images_device = [img.to(device) for img in images]
            preds = model(images_device)

            for img, target, pred in zip(images, targets, preds):
                all_data.append({
                    'image':       img.cpu(),
                    'gt_boxes':    coco_to_xyxy(target['boxes']),
                    'gt_labels':   target['labels'],
                    'pred_boxes':  pred['boxes'].cpu(),
                    'pred_labels': pred['labels'].cpu(),
                    'pred_scores': pred['scores'].cpu(),
                })

    return all_data


def visualize_errors_from_data(all_data, label_to_category_id, save_dir,
                                score_threshold=0.5, iou_threshold=0.5):
    """
    Visualize error images from pre-collected prediction data (no inference).
    Use this to compare different score thresholds without re-running the model.

    'Error image' criteria (predictions filtered by score_threshold):
    - Any GT box has no matching prediction (IoU < iou_threshold or wrong class) -> miss
    - Any prediction is unmatched to a GT box -> false positive

    Green boxes: GT  /  Red boxes: Prediction (category_id + confidence)

    Args:
        all_data: output of collect_predictions()
        label_to_category_id (dict): model label (1~N) -> category_id
        save_dir (str): directory to save visualized images
        score_threshold (float): minimum prediction confidence (default 0.5)
        iou_threshold (float): GT-prediction matching IoU threshold (default 0.5)

    Returns:
        int: number of error images saved
    """
    os.makedirs(save_dir, exist_ok=True)
    error_count = 0

    for idx, data in enumerate(all_data):
        gt_boxes  = data['gt_boxes']
        gt_labels = data['gt_labels']
        pred_boxes  = data['pred_boxes']
        pred_labels = data['pred_labels']
        pred_scores = data['pred_scores']

        keep = pred_scores >= score_threshold
        pred_boxes  = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        if not _is_error(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold):
            continue

        img_np = data['image'].permute(1, 2, 0).numpy()
        img_np = (img_np * _STD + _MEAN).clip(0, 1)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img_np)

        for box, label in zip(gt_boxes, gt_labels):
            _draw_box(ax, box, label.item(), label_to_category_id, color='lime', prefix='GT')

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            _draw_box(ax, box, label.item(), label_to_category_id,
                      color='red', prefix='Pred', score=score.item())

        ax.set_title(f'Error image {idx}  (score_thr={score_threshold}, iou_thr={iou_threshold})',
                     fontsize=10)
        ax.axis('off')

        save_path = os.path.join(save_dir, f'error_{error_count:04d}_{idx:04d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        error_count += 1

    print(f"Saved {error_count} error images -> {save_dir}")
    return error_count


def visualize_errors(model, val_loader, device, label_to_category_id, save_dir,
                     score_threshold=0.5, iou_threshold=0.5):
    """
    Collect predictions and visualize error images in one step.

    For re-visualization with different thresholds without re-running inference:
        data = collect_predictions(model, val_loader, device)
        visualize_errors_from_data(data, label_to_category_id, 'errors_03', score_threshold=0.3)
        visualize_errors_from_data(data, label_to_category_id, 'errors_05', score_threshold=0.5)
    """
    all_data = collect_predictions(model, val_loader, device)
    return visualize_errors_from_data(all_data, label_to_category_id, save_dir,
                                       score_threshold, iou_threshold)


def _is_error(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_threshold):
    """Return True if any GT is missed or any prediction is a false positive."""
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


def _draw_box(ax, box, label_idx, label_to_category_id, color, prefix, score=None):
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
