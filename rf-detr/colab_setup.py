# rf-detr/colab_setup.py
"""
Colab 환경 전용 진입 스크립트.
dataset.py/model.py/train.py는 순수 로직만 담고, Google Drive 마운트처럼
Colab 런타임에 종속된 부분과 그 오케스트레이션만 이 파일에 모아둠.

사전 준비 (노트북 셀에서 직접 실행, 코드로 감싸지 않음):
    !pip install -q "rfdetr[train,loggers]"

사용 예 (노트북 셀):
    from colab_setup import mount_drive, prepare_data, restore_data
    from train import load_config, run_kfold

    mount_drive()
    config = load_config('config.yaml')

    # 최초 1회: 5-fold 데이터 생성 + zip 백업
    prepare_data(config)
    # 이후 세션(zip 있음): 압축만 복원
    # restore_data(config)

    run_kfold(config)
"""
import os

from dataset import find_data_root, check_data_paths, build_fold_dataset, restore_dataset


def mount_drive():
    """Google Drive를 /content/drive에 마운트합니다."""
    from google.colab import drive
    drive.mount('/content/drive')


def prepare_data(config):
    """
    data_root를 찾아 5-fold 데이터셋을 생성하고 zip으로 백업합니다.
    (원본 [1] 경로 탐색 + [2-A] 5-fold 데이터 생성 단계에 해당)

    Args:
        config (dict): train.load_config()의 반환값

    Returns:
        dict: dataset.build_fold_dataset()의 반환값
    """
    data_cfg = config['data']
    data_root = find_data_root(
        candidates=data_cfg.get('data_root_candidates'),
        search_root=data_cfg.get('search_root'),
    )
    print('DATA_ROOT:', data_root)
    check_data_paths(data_root)

    proj_root = os.path.dirname(data_root)
    archive_base_path = os.path.join(proj_root, data_cfg.get('archive_name', 'dataset_5fold'))

    result = build_fold_dataset(
        data_root=data_root,
        output_dir=data_cfg['dataset_dir'],
        corrections_path=data_cfg['corrections_path'],
        cache_dir=data_cfg['cache_dir'],
        n_splits=data_cfg['n_splits'],
        seed=data_cfg['seed'],
        archive_base_path=archive_base_path,
    )

    suggested_backup = os.path.join(proj_root, 'outputs')
    print('PROJ_ROOT 기준 백업 예시 경로:', suggested_backup)
    print('→ config.yaml의 output.backup_dir을 이 경로(혹은 원하는 경로)로 채워주세요.')
    return result


def restore_data(config):
    """
    이미 생성해둔 dataset_5fold.zip을 복원합니다. (원본 [2-B] 단계에 해당)

    Args:
        config (dict): train.load_config()의 반환값
    """
    data_cfg = config['data']
    data_root = find_data_root(
        candidates=data_cfg.get('data_root_candidates'),
        search_root=data_cfg.get('search_root'),
    )
    proj_root = os.path.dirname(data_root)
    archive_name = data_cfg.get('archive_name', 'dataset_5fold')
    zip_path = os.path.join(proj_root, f'{archive_name}.zip')
    print('zip 존재:', os.path.exists(zip_path))

    local_zip = (
        os.path.join('/content', os.path.basename(zip_path))
        if zip_path.startswith('/content/drive') else None
    )
    restore_dataset(zip_path, data_cfg['dataset_dir'], local_stage_path=local_zip)
