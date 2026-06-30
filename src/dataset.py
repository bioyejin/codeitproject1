# src/dataset.py
import kagglehub
import os
import json
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


def download_data():
    """
    kagglehub에서 대회 데이터를 Download 합니다.
    이미 download된 경우 cache에서 불러옵니다.

    Returns:
        path (str): 다운로드된 대회 데이터의 경로
    """
    path = kagglehub.competition_download("ai12-level1-project")
    return path


def apply_corrections(annotations, corrections_path='corrections.json', data_path=None):
    """
    corrections.json을 적용하여 annotation을 수정/추가합니다.

    Args:
        annotations (dict): 파일명 → bbox 리스트 딕셔너리
        corrections_path (str): corrections.json 경로
        data_path (str): kagglehub 데이터 루트 경로 (download_data()의 반환값)

    Returns:
        dict: 수정된 annotations
    """
    if not os.path.exists(corrections_path):
        print("corrections.json 없음. 원본 데이터 사용")
        return annotations

    if data_path is None:
        print("data_path 없음. corrections 적용 불가")
        return annotations

    with open(corrections_path, 'r', encoding='utf-8') as f:
        corrections = json.load(f)

    # train_annotations 폴더 탐색
    json_files = glob.glob(
        os.path.join(data_path, 'sprint_ai_project1_data', 'train_annotations', '**', '*.json'),
        recursive=True
    )

    # 원본 json에서 ann_id → file_name 매핑
    all_ann_id_map = {}
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        file_name = data['images'][0]['file_name']
        ann = data['annotations'][0]
        all_ann_id_map[ann['id']] = {
            'file_name': file_name,
            'bbox': ann['bbox'],
            'category_id': ann['category_id']
        }

    # 1. 좌표 수정
    for corr in corrections['bbox_corrections']:
        ann_id = corr['ann_id']
        if ann_id in all_ann_id_map:
            file_name = all_ann_id_map[ann_id]['file_name']
            category_id = all_ann_id_map[ann_id]['category_id']
            original = corr['original']
            corrected = corr['corrected']

            if file_name in annotations:
                for ann in annotations[file_name]:
                    if ann['bbox'] == original and ann['category_id'] == category_id:
                        ann['bbox'] = corrected
                        print(f"✅ 수정 - ann_id {ann_id}: {original} → {corrected}")
                        break

    # 2. bbox 추가
    for add in corrections['bbox_additions']:
        file_name = add['file_name']
        new_ann = {
            'bbox': add['bbox'],
            'category_id': add['category_id']
        }
        if file_name in annotations:
            annotations[file_name].append(new_ann)
        else:
            annotations[file_name] = [new_ann]
        print(f"✅ 추가 - {file_name}: {add['bbox']}")

    return annotations


class PillDataset(Dataset):
    def __init__(self, path, transform=None, corrections_path='corrections.json', unique_only=True):
        """
        Args:
            path (str): 데이터 루트 경로 (download_data()의 반환값)
            transform: 이미지 증강/변환 (없으면 None)
            corrections_path (str): corrections.json 경로
        """
        self.train_img_dir = os.path.join(path, 'sprint_ai_project1_data', 'train_images')
        self.transform = transform

        # 이미지별 annotation 수집
        json_files = glob.glob(
            os.path.join(path, 'sprint_ai_project1_data', 'train_annotations', '**', '*.json'),
            recursive=True
        )

        self.annotations = {}  # 파일명 → bbox 리스트
        for jf in json_files:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_name = data['images'][0]['file_name']
            if file_name not in self.annotations:
                self.annotations[file_name] = []

            self.annotations[file_name].append({
                'bbox': data['annotations'][0]['bbox'],
                'category_id': data['annotations'][0]['category_id']
            })

        # corrections.json 적용
        self.annotations = apply_corrections(self.annotations, corrections_path, data_path=path)

        # unique_only 적용
        if unique_only:
            # 파일명에서 구성 코드 추출 후 구성당 가장 낮은 위도 1장만 유지
            from collections import defaultdict
            group_files = defaultdict(list)
            for file_name in self.annotations.keys():
                group = '_'.join(file_name.split('_')[:5])
                # 위도값 추출 (파일명의 6번째 요소)
                camera_la = int(file_name.split('_')[5])
                group_files[group].append((camera_la, file_name))
    
            # 각 구성에서 가장 낮은 위도 이미지 1장만 선택
            unique_image_names = []
            for group, files in group_files.items():
                files.sort(key=lambda x: x[0])  # 위도 기준 오름차순 정렬
                unique_image_names.append(files[0][1])  # 가장 낮은 위도 선택
    
            self.image_names = unique_image_names
        else:
            self.image_names = list(self.annotations.keys())

        # category_id → 연속된 라벨(1~N) 매핑 생성
        all_category_ids = sorted(set(
            ann['category_id']
            for anns in self.annotations.values()
            for ann in anns
        ))
        self.category_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(all_category_ids)}  # 0은 배경용으로 비움
        self.label_to_category_id = {v: k for k, v in self.category_id_to_label.items()}  # 역매핑 (결과 해석용)

    def __len__(self):
        """데이터셋 크기를 반환합니다."""
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        idx번째 이미지와 annotation 반환합니다.
        
        Returns:
            image (tensor): 이미지 텐서
            target (dict): {'boxes': tensor, 'labels': tensor}
        """
        file_name = self.image_names[idx]
        img_path = os.path.join(self.train_img_dir, file_name)
        
        image = Image.open(img_path).convert('RGB')
        
        # bbox와 label을 텐서로 변환
        boxes = torch.tensor(
            [ann['bbox'] for ann in self.annotations[file_name]],
            dtype=torch.float32
        )
        # category_id를 연속된 라벨로 변환
        labels = torch.tensor(
            [self.category_id_to_label[ann['category_id']] for ann in self.annotations[file_name]],
            dtype=torch.int64
        )
        
        target = {'boxes': boxes, 'labels': labels}
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def coco_to_xyxy(boxes):
    """
    COCO 포맷 [x, y, w, h] → [x1, y1, x2, y2]로 변환합니다.
    (torchvision FasterRCNN 등이 기대하는 형식)

    Args:
        boxes (Tensor): [N, 4] 형태, COCO 포맷

    Returns:
        Tensor: [N, 4] 형태, [x1, y1, x2, y2] 포맷
    """
    x, y, w, h = boxes.unbind(1)
    return torch.stack([x, y, x + w, y + h], dim=1)


def get_transform(train=True):
    """
    이미지 전처리 및 증강을 정의합니다.
    
    Args:
        train (bool): 학습용이면 True, 검증/테스트용이면 False
    
    Returns:
        transforms.Compose: 변환 파이프라인
    """
    if train:
        return transforms.Compose([
            transforms.Resize((640, 640)),       # 크기 통일
            transforms.RandomHorizontalFlip(),   # 좌우 반전
            transforms.ColorJitter(              # 밝기/대비 조정
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),               # PIL → 텐서
            transforms.Normalize(                # 정규화
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def get_dataloader(path, batch_size=4, train=True, shuffle=None, unique_only=True, collate='stack'):
    """
    DataLoader를 생성합니다.
    
    Args:
        path (str): 데이터 루트 경로
        batch_size (int): 배치 크기
        train (bool): 학습용이면 True, 검증용이면 False
        shuffle (bool): 순서 섞기 (None이면 train 여부에 따라 자동 설정)
        unique_only (bool): 고유 이미지만 사용할지 여부
        collate (str): 'list' 또는 'stack' - 배치 묶는 방식

    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = train  # 학습용이면 shuffle, 검증용이면 안 함
    
    transform = get_transform(train=train)
    dataset = PillDataset(path, transform=transform, unique_only=unique_only)
    
    collate_func = collate_fn_stack if collate == 'stack' else collate_fn_list

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,       # Windows에서는 0으로 설정
        collate_fn=collate_func
    )


def collate_fn_stack(batch):
    """
    이미지마다 bbox 개수가 달라서 기본 collate가 안 되므로
    zip으로 묶은 뒤 targets를 리스트로 변환해서 반환합니다.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def collate_fn_list(batch):
    """
    이미지를 리스트 형태로 유지합니다. (torchvision FasterRCNN 등 List[Tensor] 입력을 기대하는 모델용)
    """
    images, targets = zip(*batch)
    return list(images), list(targets)