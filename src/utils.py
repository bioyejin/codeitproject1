# src/utils.py
import os
import json
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def get_groups(image_names):
    """
    이미지 파일명에서 구성(group) 코드를 추출합니다.
    같은 구성(같은 알약을 다른 각도로 촬영한 이미지)은 같은 group 값을 가집니다.

    Args:
        image_names (list): 이미지 파일명 리스트

    Returns:
        list: 각 이미지에 대응하는 group 코드 리스트
    """
    groups = []
    for file_name in image_names:
        group = '_'.join(file_name.split('_')[:5])
        groups.append(group)
    return groups


def plot_history(history, title='Training History', save_path=None):
    """
    학습 history(train_loss, val_map 등)를 시각화합니다.

    Args:
        history (dict): {'train_loss': [...], 'val_map': [...], 'val_map_50': [...], 'val_map_75_95': [...]}
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
    axes[1].plot(history['val_map_75_95'], marker='o', markersize=3, color='purple', label='mAP@0.75:0.95 (selection metric)')

    best_epoch = int(np.argmax(history['val_map_75_95'])) + 1   # ← 기준 지표 변경
    best_val = max(history['val_map_75_95'])
    axes[1].axvline(x=best_epoch - 1, color='gray', linestyle='--', label=f'Best epoch {best_epoch}')
    axes[1].set_title('Validation mAP')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mAP')
    axes[1].legend()
    axes[1].grid(True)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def set_seed(seed=42):
    """
    재현성을 위해 시드(seed)를 설정합니다.

    Args:
        seed (int): 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, path):
    """
    모델 가중치를 저장합니다. 상위 폴더가 없으면 생성합니다.

    Args:
        model: 저장할 모델
        path (str): 저장 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device='cpu'):
    """
    저장된 가중치를 모델에 불러옵니다.

    Args:
        model: 가중치를 불러올 모델
        path (str): 가중치 파일 경로
        device (str): 'cuda' or 'cpu'

    Returns:
        model: 가중치가 로드된 모델
    """
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def get_category_names(data_path):
    """
    train_annotations json 파일에서 category_id → 클래스명 매핑을 만듭니다.

    Args:
        data_path (str): kagglehub 데이터 루트 경로 (download_data()의 반환값)

    Returns:
        dict: {category_id: name}
    """
    json_files = glob.glob(
        os.path.join(data_path, 'sprint_ai_project1_data', 'train_annotations', '**', '*.json'),
        recursive=True
    )

    category_map = {}
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for cat in data['categories']:
            category_map[cat['id']] = cat['name']

    return category_map