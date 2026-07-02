# rf-detr/dataset.py
"""
RF-DETR용 5-fold COCO 데이터셋 생성/복원 로직.
원본 rfdetr_train_5fold_colab.py의 [1], [2-A], [2-B] 블록을 함수로 분리한 것.
"""
import os
import glob
import json
import shutil
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def find_data_root(candidates=None, search_root=None, target_name='sprint_ai_project1_data'):
    """
    sprint_ai_project1_data 폴더를 후보 경로에서 우선 탐색하고,
    없으면 search_root 아래를 재귀 검색합니다.

    Args:
        candidates (list): 우선 확인할 후보 경로 리스트
        search_root (str): 후보에 없을 때 재귀 검색할 루트 (예: '/content/drive/MyDrive')
        target_name (str): 찾을 폴더 이름

    Returns:
        str: 데이터 루트 경로
    """
    candidates = candidates or []
    data_root = next((c for c in candidates if os.path.exists(c)), None)
    if data_root is None and search_root:
        hits = glob.glob(os.path.join(search_root, '**', target_name), recursive=True)
        data_root = hits[0] if hits else None
    assert data_root, f"{target_name}를 못 찾음 — 공유 폴더 바로가기 확인"
    return data_root


def check_data_paths(data_root):
    """train/test 이미지·annotation 하위 경로 존재 여부를 점검합니다."""
    paths = {
        'train_images': os.path.join(data_root, 'train_images'),
        'train_annotations': os.path.join(data_root, 'train_annotations'),
        'test_images': os.path.join(data_root, 'test_images'),
    }
    for name, p in paths.items():
        print(name, p, '->', os.path.exists(p))
    return paths


def load_raw_annotations(train_ann_dir):
    """
    박스당 1개 JSON으로 흩어진 원본 annotation을 파일명 기준으로 병합합니다.

    Args:
        train_ann_dir (str): train_annotations 루트 경로

    Returns:
        boxes_by_image (dict): file_name -> [bbox, ...]
        cats_by_image (dict): file_name -> [category_id, ...]
        img_meta (dict): file_name -> (width, height)
    """
    boxes_by_image, cats_by_image, img_meta = defaultdict(list), defaultdict(list), {}
    for p in glob.glob(os.path.join(train_ann_dir, '**', '*.json'), recursive=True):
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
        im = d['images'][0]
        fn = im['file_name']
        img_meta[fn] = (im['width'], im['height'])
        for a in d['annotations']:
            boxes_by_image[fn].append(a['bbox'])
            cats_by_image[fn].append(a['category_id'])
    return boxes_by_image, cats_by_image, img_meta


def apply_corrections(boxes_by_image, cats_by_image, corrections_path):
    """
    corrections.json을 coord_fix -> remove_boxes -> modify_boxes -> add_boxes 순서로 적용합니다.
    boxes_by_image/cats_by_image를 제자리(in-place)에서 수정하고 그대로 반환합니다.

    Args:
        boxes_by_image (dict): load_raw_annotations()의 반환값
        cats_by_image (dict): load_raw_annotations()의 반환값
        corrections_path (str): corrections.json 경로

    Returns:
        boxes_by_image, cats_by_image (수정된 딕셔너리, in-place와 동일 객체)
    """
    with open(corrections_path, 'r', encoding='utf-8') as f:
        corr = json.load(f)

    # 1) 좌표 오염 수정 (동일 bbox를 가진 항목을 모두 교체)
    for fn, fixes in corr.get('coord_fix', {}).items():
        for fix in fixes:
            original, corrected = fix['original'], fix['corrected']
            for i, b in enumerate(boxes_by_image[fn]):
                if b == original:
                    boxes_by_image[fn][i] = corrected

    # 2) 중복/오류 박스 제거 (항목당 첫 매치 1개만)
    for fn, removals in corr.get('remove_boxes', {}).items():
        for rm in removals:
            kept_boxes, kept_cats, done = [], [], False
            for c, b in zip(cats_by_image[fn], boxes_by_image[fn]):
                if (not done) and c == rm['category_id'] and b == rm['bbox']:
                    done = True
                    continue
                kept_boxes.append(b)
                kept_cats.append(c)
            boxes_by_image[fn], cats_by_image[fn] = kept_boxes, kept_cats

    # 3) 좌표 수정 (category_id [+ match_bbox] 매치, 첫 매치만 수정)
    for fn, mods in corr.get('modify_boxes', {}).items():
        for mod in mods:
            mc = mod['category_id']
            w = mod.get('match_bbox')
            new = mod.get('directive', mod.get('new_bbox'))
            for i, (c, b) in enumerate(zip(cats_by_image[fn], boxes_by_image[fn])):
                if c == mc and (w is None or b == w):
                    if new == 'EXTEND_DOWN_95':
                        boxes_by_image[fn][i] = [b[0], b[1], b[2], b[3] + 95]
                    else:
                        boxes_by_image[fn][i] = new
                    break

    # 4) 누락 박스 추가
    for fn, adds in corr.get('add_boxes', {}).items():
        for add in adds:
            cats_by_image[fn].append(add['category_id'])
            boxes_by_image[fn].append(add['bbox'])

    return boxes_by_image, cats_by_image


def build_category_mapping(cats_by_image):
    """
    category_id를 1-indexed 라벨로 매핑합니다.
    0은 RF-DETR가 기대하는 더미 배경 카테고리("pill")용으로 비워둡니다.

    Returns:
        all_cats (list): 정렬된 원본 category_id 리스트
        cat2label (dict): category_id -> label(1~N)
        label2cat (dict): label(1~N) -> category_id (역매핑, 결과 해석용)
    """
    all_cats = sorted({c for cs in cats_by_image.values() for c in cs})
    cat2label = {c: i + 1 for i, c in enumerate(all_cats)}
    label2cat = {v: k for k, v in cat2label.items()}
    return all_cats, cat2label, label2cat


def build_coco(files, boxes_by_image, cats_by_image, img_meta, cat2label, all_cats):
    """
    주어진 파일 목록으로 COCO 포맷 dict를 만듭니다.
    categories에 id=0 더미("pill")를 포함시켜 RF-DETR의 category id 규약을 맞춥니다.

    Args:
        files (list): 이 split에 포함될 file_name 리스트

    Returns:
        dict: COCO 포맷 {'images', 'annotations', 'categories'}
    """
    imgs, anns, aid = [], [], 1
    for iid, fn in enumerate(files, 1):
        W, H = img_meta[fn]
        imgs.append({'id': iid, 'file_name': fn, 'width': W, 'height': H})
        for c, b in zip(cats_by_image[fn], boxes_by_image[fn]):
            anns.append({
                'id': aid, 'image_id': iid, 'category_id': cat2label[c],
                'bbox': [float(v) for v in b], 'area': float(b[2] * b[3]), 'iscrowd': 0,
            })
            aid += 1
    cats = [{'id': 0, 'name': 'pill', 'supercategory': 'none'}] + \
           [{'id': cat2label[c], 'name': str(c), 'supercategory': 'pill'} for c in all_cats]
    return {'images': imgs, 'annotations': anns, 'categories': cats}


def make_folds(file_names, boxes_by_image, cats_by_image, cat2label, n_splits, seed):
    """
    StratifiedGroupKFold로 fold를 나눕니다.
    - group: 구성 코드 (파일명에서 '_0_2' 앞부분) -> 같은 구성이 train/val에 섞이지 않게 함
    - 층화 라벨: 이미지에 등장하는 카테고리 중 전체 등장 빈도가 가장 낮은 것 (클래스 불균형 완화)

    Returns:
        list: [(train_idx, val_idx), ...] (file_names 기준 인덱스, 길이 n_splits)
    """
    cls_freq = defaultdict(int)
    for cs in cats_by_image.values():
        for c in cs:
            cls_freq[c] += 1

    groups = np.array([fn.split('_0_2')[0] for fn in file_names])
    strat = np.array([
        cat2label[min(cats_by_image[fn], key=lambda c: cls_freq[c])]
        for fn in file_names
    ])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(file_names, strat, groups))


def cache_images(train_img_dir, cache_dir):
    """
    원본 이미지를 로컬 cache_dir로 1회 복사합니다.
    (드라이브 read 1회로 줄이고, 이후 fold별 복사는 로컬끼리라 빨라짐)
    """
    os.makedirs(cache_dir, exist_ok=True)
    src_paths = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(train_img_dir, '**', '*.png'), recursive=True)
    }
    for fn, src in src_paths.items():
        shutil.copy(src, os.path.join(cache_dir, fn))
    print('이미지 캐시:', len(src_paths))
    return cache_dir


def write_fold_dirs(folds, file_names, boxes_by_image, cats_by_image, img_meta,
                     cat2label, all_cats, cache_dir, output_dir):
    """fold별 {output_dir}/fold{i}/{train,valid} 디렉토리에 COCO json + 이미지를 배치합니다."""
    for fi, (tr, va) in enumerate(folds):
        for idxs, split in [(tr, 'train'), (va, 'valid')]:
            files = [file_names[i] for i in idxs]
            d = os.path.join(output_dir, f'fold{fi}', split)
            os.makedirs(d, exist_ok=True)
            coco = build_coco(files, boxes_by_image, cats_by_image, img_meta, cat2label, all_cats)
            with open(os.path.join(d, '_annotations.coco.json'), 'w') as f:
                json.dump(coco, f)
            for fn in files:
                shutil.copy(os.path.join(cache_dir, fn), os.path.join(d, fn))
        print(f'fold{fi}: train {len(tr)} / valid {len(va)}')


def save_label_map(cat2label, label2cat, output_dir):
    """label_map.json을 output_dir에 저장합니다 (문자열 키로 직렬화)."""
    path = os.path.join(output_dir, 'label_map.json')
    with open(path, 'w') as f:
        json.dump({
            'cat2label': {str(k): v for k, v in cat2label.items()},
            'label2cat': {str(k): v for k, v in label2cat.items()},
        }, f)
    return path


def load_label_map(output_dir):
    """save_label_map()이 저장한 label_map.json을 읽어옵니다 (정수 키로 역직렬화)."""
    path = os.path.join(output_dir, 'label_map.json')
    with open(path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    return {
        'cat2label': {int(k): v for k, v in label_map['cat2label'].items()},
        'label2cat': {int(k): v for k, v in label_map['label2cat'].items()},
    }


def archive_dataset(output_dir, archive_base_path):
    """
    output_dir 전체를 zip으로 묶습니다.

    Args:
        archive_base_path (str): 확장자 없는 경로 (예: '/path/dataset_5fold')

    Returns:
        str: 생성된 zip 파일의 전체 경로
    """
    archive_path = shutil.make_archive(archive_base_path, 'zip', output_dir)
    print('zip 저장 완료:', archive_path)
    return archive_path


def restore_dataset(zip_path, extract_dir, local_stage_path=None):
    """
    dataset_5fold.zip을 extract_dir에 복원합니다.

    Args:
        zip_path (str): 원본 zip 경로 (드라이브 등)
        extract_dir (str): 압축을 풀 디렉토리
        local_stage_path (str): 주어지면 zip을 이 경로로 먼저 복사한 뒤 그 사본을 풂
                                 (Colab에서 드라이브 반복 read를 줄이기 위한 옵션)
    """
    os.makedirs(extract_dir, exist_ok=True)
    src = zip_path
    if local_stage_path:
        shutil.copy(zip_path, local_stage_path)
        src = local_stage_path
    shutil.unpack_archive(src, extract_dir, 'zip')
    print('복원 fold:', sorted(d for d in os.listdir(extract_dir) if d.startswith('fold')))


def build_fold_dataset(data_root, output_dir, corrections_path, cache_dir,
                        n_splits=5, seed=42, archive_base_path=None):
    """
    원본 annotation 로드 -> corrections 적용 -> StratifiedGroupKFold 분할
    -> fold 디렉토리 생성까지 한 번에 수행합니다 (원본 [2-A] 단계 전체에 해당).

    Args:
        data_root (str): sprint_ai_project1_data 경로
        output_dir (str): fold 디렉토리를 만들 루트 (예: '/content/dataset')
        corrections_path (str): corrections.json 경로
        cache_dir (str): 이미지 로컬 캐시 경로
        n_splits (int): fold 수
        seed (int): StratifiedGroupKFold 시드
        archive_base_path (str): 주어지면 완료 후 zip으로 백업 (확장자 제외 경로)

    Returns:
        dict: {'cat2label', 'label2cat', 'num_classes', 'archive_path'(옵션)}
    """
    train_img_dir = os.path.join(data_root, 'train_images')
    train_ann_dir = os.path.join(data_root, 'train_annotations')

    boxes_by_image, cats_by_image, img_meta = load_raw_annotations(train_ann_dir)
    boxes_by_image, cats_by_image = apply_corrections(boxes_by_image, cats_by_image, corrections_path)

    file_names = sorted(boxes_by_image.keys())
    print('이미지', len(file_names), '/ 박스', sum(len(v) for v in boxes_by_image.values()))

    all_cats, cat2label, label2cat = build_category_mapping(cats_by_image)
    folds = make_folds(file_names, boxes_by_image, cats_by_image, cat2label, n_splits, seed)

    cache_images(train_img_dir, cache_dir)
    write_fold_dirs(folds, file_names, boxes_by_image, cats_by_image, img_meta,
                     cat2label, all_cats, cache_dir, output_dir)
    save_label_map(cat2label, label2cat, output_dir)

    result = {'cat2label': cat2label, 'label2cat': label2cat, 'num_classes': len(cat2label)}
    if archive_base_path:
        result['archive_path'] = archive_dataset(output_dir, archive_base_path)
    return result
