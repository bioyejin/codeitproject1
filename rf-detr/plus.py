# plus.py
# 아직 어느 파일로 옮길지 정리 중인 스테이징 파일.
# 함수마다 상단 주석으로 "추후 어떤 파일로 들어가야 하는지"를 표시해둠.
# (plot_history/collect_predictions_from_coco/visualize_errors_from_data/evaluate_from_data는
#  utils.py/visualize.py로 이미 옮겨졌음 - 여기서는 sanity_check()만 남음)
import os
import json
import copy

from train import load_config, run_kfold
from model import get_rfdetr_model
from visualize import collect_predictions_from_coco, evaluate_from_data, visualize_errors_from_data


# ============================================================
# 추후 어떤 파일로 들어가야 하는지: -> train.py
# ============================================================

def sanity_check(config_path='config.yaml', max_folds=1, epochs=1,
                  score_threshold=0.5, iou_threshold=0.5):
    """
    학습 -> 체크포인트 저장 -> 추론 -> mAP 계산 -> 오답 이미지 시각화까지
    파이프라인 전체가 에러 없이 이어지는지 fold 1개 x epoch 1회 정도로 빠르게 확인합니다.

    실제 학습 산출물과 절대 섞이지 않도록:
    - model.tag에 'sanitycheck_' 접두어를 붙여서 체크포인트 파일명에 그대로 드러나게 함
      (예: sanitycheck_small_res512_fold0_best.pth)
    - 백업/임시 출력 경로도 output.backup_dir, output.local_output_dir 아래
      'sanity_check' 하위 폴더로 분리
    - 오답 시각화 이미지도 'sanitycheck_error_...png'로 저장
    또한 재실행할 때마다 실제로 다시 검증되도록, 이전 sanity check 체크포인트가 남아있으면
    지우고 시작합니다 (train_fold의 '이미 있으면 건너뛰기' 로직에 걸려 검증이 스킵되는 것 방지).

    run_kfold()를 그대로 호출하므로, fold별 리포팅(클래스별 mAP + 오답 시각화)과
    5-fold 요약(폴드 1개뿐이라 요약은 사실상 그 1개 값)도 함께 검증됩니다.

    Returns:
        dict: {'checkpoint_paths', 'metrics', 'vis_dir'}
    """
    config = copy.deepcopy(load_config(config_path))
    config['train']['epochs'] = epochs

    original_tag = config['model']['tag']
    config['model']['tag'] = f'sanitycheck_{original_tag}'
    config['output']['backup_dir'] = os.path.join(config['output']['backup_dir'], 'sanity_check')
    config['output']['local_output_dir'] = os.path.join(config['output']['local_output_dir'], 'sanity_check')

    # 이전 sanity check 잔여 체크포인트 제거 (매번 실제로 다시 돌아가게)
    os.makedirs(config['output']['backup_dir'], exist_ok=True)
    for fi in range(max_folds):
        stale = os.path.join(config['output']['backup_dir'], f"{config['model']['tag']}_fold{fi}_best.pth")
        if os.path.exists(stale):
            os.remove(stale)
            print(f'[sanity check] 이전 잔여 체크포인트 삭제: {stale}')

    checkpoint_paths = run_kfold(config, max_folds=max_folds)
    print(f'[sanity check] 학습 + 체크포인트 저장 + fold 리포팅 확인: {checkpoint_paths}')

    checkpoint_path = checkpoint_paths[0] if checkpoint_paths else None
    if not checkpoint_path:
        print('[sanity check] 체크포인트가 저장되지 않음 -> 추가 추론/시각화 단계는 건너뜀')
        return {'checkpoint_paths': checkpoint_paths, 'metrics': None, 'vis_dir': None}

    # run_kfold()가 이미 report_fold_result()로 fold0에 대해 mAP+시각화를 수행했지만,
    # sanity_check 전용 접두어('sanitycheck_error_')가 붙은 별도 이미지로 한 번 더
    # 저장해 파이프라인 각 단계를 명시적으로 재확인함.
    fold0_valid_dir = os.path.join(config['data']['dataset_dir'], 'fold0', 'valid')
    coco_json_path = os.path.join(fold0_valid_dir, '_annotations.coco.json')

    label_map_path = os.path.join(config['data']['dataset_dir'], 'label_map.json')
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    label_to_category_id = {int(k): v for k, v in label_map['label2cat'].items()}

    model = get_rfdetr_model(config['model']['variant'], checkpoint_path=checkpoint_path)
    pred_data = collect_predictions_from_coco(model, coco_json_path, fold0_valid_dir, score_threshold=0.0)
    print(f'[sanity check] 추론 확인: {len(pred_data)}장 처리')

    metrics = evaluate_from_data(pred_data)
    print(f"[sanity check] mAP 계산 확인: mAP={metrics['map']:.4f} / mAP@50={metrics['map_50']:.4f} "
          f"/ mAP@0.75:0.95={metrics['map_75_95']:.4f}")

    vis_dir = os.path.join(config['output']['backup_dir'], 'sanity_check_vis')
    saved = visualize_errors_from_data(
        pred_data, label_to_category_id, save_dir=vis_dir,
        score_threshold=score_threshold, iou_threshold=iou_threshold,
        file_prefix='sanitycheck_error',
    )
    print(f'[sanity check] 시각화 이미지 저장 확인: {saved}장 -> {vis_dir}')
    print('[sanity check] 전체 파이프라인(학습->체크포인트->fold 리포팅->추론->mAP->시각화) 통과')

    return {'checkpoint_paths': checkpoint_paths, 'metrics': metrics, 'vis_dir': vis_dir}


if __name__ == '__main__':
    sanity_check()
