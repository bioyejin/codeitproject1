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


class PillDataset(Dataset):
    def __init__(self, path, transform=None):
        """
        Args:
            path (str): 데이터 루트 경로 (download_data()의 반환값)
            transform: 이미지 증강/변환 (없으면 None)
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
                'bbox': data['annotations'][0]['bbox'],        # [x, y, w, h]
                'category_id': data['annotations'][0]['category_id']
            })
        
        self.image_names = list(self.annotations.keys())

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
        labels = torch.tensor(
            [ann['category_id'] for ann in self.annotations[file_name]],
            dtype=torch.int64
        )
        
        target = {'boxes': boxes, 'labels': labels}
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


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


def get_dataloader(path, batch_size=4, train=True, shuffle=None):
    """
    DataLoader를 생성합니다.
    
    Args:
        path (str): 데이터 루트 경로
        batch_size (int): 배치 크기
        train (bool): 학습용이면 True, 검증용이면 False
        shuffle (bool): 순서 섞기 (None이면 train 여부에 따라 자동 설정)
    
    Returns:
        DataLoader
    """
    if shuffle is None:
        shuffle = train  # 학습용이면 shuffle, 검증용이면 안 함
    
    transform = get_transform(train=train)
    dataset = PillDataset(path, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,       # Windows에서는 0으로 설정
        collate_fn=collate_fn
    )


def collate_fn(batch):
    """
    이미지마다 bbox 개수가 달라서 기본 collate가 안 되므로
    zip으로 묶은 뒤 targets를 리스트로 변환해서 반환합니다.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)