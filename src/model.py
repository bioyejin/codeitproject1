# src/model.py
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import swin_t, Swin_T_Weights


def build_swin_fpn(num_classes, pretrained=True, freeze_backbone=True):
    """
    Swin-T 백본 + FPN + Faster R-CNN 헤드로 구성된 모델을 반환합니다.

    Args:
        num_classes (int): 클래스 수 (배경 포함)
        pretrained (bool): Swin-T 사전학습 가중치 사용 여부
        freeze_backbone (bool): 백본 freeze 여부

    Returns:
        model: Swin-FPN 모델
    """
    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = swin_t(weights=weights)
    backbone = nn.Sequential(*list(backbone.children())[:-3])

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False

    backbone.out_channels = 768

    # feature map 1개에 맞게 anchor 설정
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],    # feature map 1개
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


def get_model(model_name, num_classes=57, pretrained=True, freeze_backbone=True):
    """
    모델 이름으로 모델을 반환합니다.

    Args:
        model_name (str): 모델 이름 ('swin_fpn', 'swin_detr', 'rt_detr', 'dino')
        num_classes (int): 클래스 수 (배경 포함, 기본값 57 = 56클래스 + 배경)
        pretrained (bool): 사전학습 가중치 사용 여부
        freeze_backbone (bool): 백본 freeze 여부

    Returns:
        model
    """
    if model_name == 'swin_fpn':
        return build_swin_fpn(num_classes, pretrained, freeze_backbone)
    else:
        raise NotImplementedError(f"{model_name}은 아직 구현되지 않았습니다.")