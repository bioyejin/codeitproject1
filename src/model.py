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


class SwinDetrWrapper(nn.Module):
    """
    Swin-T 백본 + DETR 헤드 (HuggingFace transformers + timm).
    FasterRCNN과 동일한 인터페이스를 제공하는 래퍼입니다.

    - 학습: model(images, targets) → {'loss_detr': tensor}
    - 추론: model(images) → List[{'boxes': xyxy, 'labels': int, 'scores': float}]

    targets 형식: COCO [x, y, w, h] (box_format='coco' 설정 필요)
    """

    def __init__(self, num_classes, pretrained=True, freeze_backbone=True,
                 num_queries=100, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        import timm as _timm
        from transformers import DetrConfig, DetrForObjectDetection

        # DETR의 num_labels는 배경을 제외한 실제 객체 클래스 수 (no-object는 내부 처리)
        self._num_object_classes = num_classes - 1

        # transformers 5.x의 TimmBackbone은 backbone_kwargs의 strict_img_size를
        # timm.create_model에 전달하지 않아 640×640 입력 시 고정 크기 assertion 오류 발생.
        # 해결: DETR 모델을 먼저 생성한 뒤 backbone.model을 직접 교체.
        config = DetrConfig(
            use_timm_backbone=True,
            backbone='swin_tiny_patch4_window7_224',
            use_pretrained_backbone=False,   # 아래에서 timm 직접 로드
            num_labels=self._num_object_classes,
            num_queries=num_queries,
            d_model=256,
            encoder_layers=num_encoder_layers,
            decoder_layers=num_decoder_layers,
            backbone_kwargs={'out_indices': (3,)},  # intermediate_channel_sizes=[768] 설정용
        )
        self.model = DetrForObjectDetection(config)

        # timm Swin-T를 올바른 설정으로 직접 생성
        # strict_img_size=False: 640×640 입력 허용 (기본값 224 제한 해제)
        # Swin-T 마지막 스테이지(3): 20×20×768 feature map (640 입력 기준)
        timm_backbone = _timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,),
            strict_img_size=False,
        )

        # Swin-T timm 출력은 NHWC (B,H,W,C); DETR은 NCHW (B,C,H,W) 기대 → 래퍼로 변환
        class _SwinExtractor(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return [f.permute(0, 3, 1, 2).contiguous() for f in self.m(x)]

        extractor = _SwinExtractor(timm_backbone)

        # transformers 버전마다 내부 구조가 달라 경로 하드코딩이 불안정.
        # named_modules()로 탐색: .model 속성을 가지고, 그 안에 strict_img_size가 있는 모듈
        # = TimmBackbone (클래스명과 무관하게 동작)
        _replaced = False
        for _, mod in self.model.named_modules():
            inner = getattr(mod, 'model', None)
            if inner is not None and hasattr(inner, 'strict_img_size'):
                mod.model = extractor
                _replaced = True
                break
        if not _replaced:
            raise RuntimeError("TimmBackbone을 찾지 못했습니다. transformers 버전을 확인하세요.")

        if freeze_backbone:
            for param in extractor.m.parameters():
                param.requires_grad = False

    def _to_detr_targets(self, targets, H, W):
        """
        COCO [x,y,w,h] → DETR 정규화 [cx,cy,w,h] 변환.
        레이블도 1-indexed → 0-indexed로 변환합니다.
        """
        detr_targets = []
        for t in targets:
            boxes = t['boxes']                              # [N, 4] COCO [x,y,w,h]
            cx = (boxes[:, 0] + boxes[:, 2] / 2) / W
            cy = (boxes[:, 1] + boxes[:, 3] / 2) / H
            bw = boxes[:, 2] / W
            bh = boxes[:, 3] / H
            detr_targets.append({
                'class_labels': t['labels'] - 1,           # 1-indexed → 0-indexed
                'boxes': torch.stack([cx, cy, bw, bh], dim=1).clamp(0, 1),
            })
        return detr_targets

    def _to_frcnn_preds(self, outputs, H, W):
        """
        DETR 출력 → FasterRCNN 스타일 예측 변환.
        - logits의 마지막 차원이 no-object 클래스 → 제외 후 max
        - pred_boxes(정규화 cx,cy,w,h) → 절댓값 xyxy
        - 레이블 0-indexed → 1-indexed
        """
        probs = outputs.logits.softmax(-1)              # (B, Q, num_classes+1)
        scores, labels = probs[..., :-1].max(-1)        # no-object 클래스 제외

        cx, cy, bw, bh = outputs.pred_boxes.unbind(-1)  # 정규화 cx,cy,w,h
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0)

        results = []
        for i in range(scores.shape[0]):
            results.append({
                'boxes':  boxes[i],
                'labels': labels[i] + 1,                # 0-indexed → 1-indexed
                'scores': scores[i],
            })
        return results

    def forward(self, images, targets=None):
        pixel_values = torch.stack(images) if isinstance(images, list) else images
        _, _, H, W = pixel_values.shape

        if self.training and targets is not None:
            detr_targets = self._to_detr_targets(targets, H, W)
            outputs = self.model(pixel_values=pixel_values, labels=detr_targets)
            return {'loss_detr': outputs.loss}

        outputs = self.model(pixel_values=pixel_values)
        return self._to_frcnn_preds(outputs, H, W)


class RtDetrV2Wrapper(nn.Module):
    """
    RT-DETR V2 (HuggingFace transformers) 래퍼.
    ResNet-50 백본 + RT-DETR V2 하이브리드 인코더/디코더.
    FasterRCNN과 동일한 인터페이스 제공.

    - 학습: model(images, targets) → {'loss_rtdetr': tensor}
    - 추론: model(images) → List[{'boxes': xyxy, 'labels': int, 'scores': float}]

    targets 형식: COCO [x, y, w, h] (box_format='coco' 설정 필요)

    RT-DETR V2는 sigmoid 분류 (배경 클래스 없음) → DETR softmax와 다름.
    """

    def __init__(self, num_classes, pretrained=True, freeze_backbone=True):
        super().__init__()
        from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection

        self._num_object_classes = num_classes - 1  # 배경 제외

        if pretrained:
            try:
                # PekingU/rtdetr_v2_r50vd (COCO 80클래스) → 56클래스로 헤드 재초기화
                self.model = RTDetrV2ForObjectDetection.from_pretrained(
                    "PekingU/rtdetr_v2_r50vd",
                    num_labels=self._num_object_classes,
                    ignore_mismatched_sizes=True,
                )
            except Exception as e:
                print(f"[rt_detr] pretrained 로드 실패 ({e}), 랜덤 초기화로 대체")
                config = RTDetrV2Config(num_labels=self._num_object_classes)
                self.model = RTDetrV2ForObjectDetection(config)
        else:
            config = RTDetrV2Config(num_labels=self._num_object_classes)
            self.model = RTDetrV2ForObjectDetection(config)

        if freeze_backbone:
            for param in self.model.model.backbone.parameters():
                param.requires_grad = False

    def _to_rtdetr_targets(self, targets, H, W):
        """COCO [x,y,w,h] → 정규화 cx,cy,w,h, 레이블 1→0-indexed"""
        rtdetr_targets = []
        for t in targets:
            boxes = t['boxes']
            cx = (boxes[:, 0] + boxes[:, 2] / 2) / W
            cy = (boxes[:, 1] + boxes[:, 3] / 2) / H
            bw = boxes[:, 2] / W
            bh = boxes[:, 3] / H
            rtdetr_targets.append({
                'class_labels': t['labels'] - 1,
                'boxes': torch.stack([cx, cy, bw, bh], dim=1).clamp(0, 1),
            })
        return rtdetr_targets

    def _to_frcnn_preds(self, outputs, H, W):
        """
        RT-DETR V2 출력 → FasterRCNN 스타일 예측 변환.
        - logits: (B, Q, num_classes) sigmoid (배경 클래스 없음)
        - pred_boxes: 정규화 cx,cy,w,h → 절댓값 xyxy
        """
        scores, labels = outputs.logits.sigmoid().max(-1)  # sigmoid, 배경 클래스 없음

        cx, cy, bw, bh = outputs.pred_boxes.unbind(-1)
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        boxes = torch.stack([x1, y1, x2, y2], dim=-1).clamp(min=0)

        return [
            {'boxes': boxes[i], 'labels': labels[i] + 1, 'scores': scores[i]}
            for i in range(scores.shape[0])
        ]

    def forward(self, images, targets=None):
        pixel_values = torch.stack(images) if isinstance(images, list) else images
        _, _, H, W = pixel_values.shape

        if self.training and targets is not None:
            rtdetr_targets = self._to_rtdetr_targets(targets, H, W)
            outputs = self.model(pixel_values=pixel_values, labels=rtdetr_targets)
            return {'loss_rtdetr': outputs.loss}

        outputs = self.model(pixel_values=pixel_values)
        return self._to_frcnn_preds(outputs, H, W)


class DINOWrapper(SwinDetrWrapper):
    """
    DINO (DETR with Improved DeNoising Anchor Boxes) 근사 구현.
    Swin-T 백본 + DETR 헤드, DINO 논문의 핵심 설정(쿼리 300개, 6레이어)을 따릅니다.

    주의: CDN(Contrastive DeNoising), mixed query selection 등 DINO의 핵심 학습 기법은
    표준 라이브러리에 없어 포함되지 않습니다. 완전한 구현은 IDEA-Research/DINO 참고.
    """

    def __init__(self, num_classes, pretrained=True, freeze_backbone=True):
        super().__init__(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            num_queries=300,            # DINO 논문 기본값 (vanilla DETR는 100)
            num_encoder_layers=6,
            num_decoder_layers=6,
        )


def get_model(model_name, num_classes=57, pretrained=True, freeze_backbone=True):
    """
    모델 이름으로 모델을 반환합니다.

    Args:
        model_name (str): 'swin_fpn' | 'swin_detr' | 'rt_detr' | 'dino'
        num_classes (int): 클래스 수 (배경 포함, 기본값 57 = 56클래스 + 배경)
        pretrained (bool): 백본 사전학습 가중치 사용 여부
        freeze_backbone (bool): 백본 freeze 여부

    Returns:
        model
    """
    if model_name == 'swin_fpn':
        return build_swin_fpn(num_classes, pretrained, freeze_backbone)
    elif model_name == 'swin_detr':
        return SwinDetrWrapper(num_classes, pretrained, freeze_backbone)
    elif model_name == 'rt_detr':
        return RtDetrV2Wrapper(num_classes, pretrained, freeze_backbone)
    elif model_name == 'dino':
        return DINOWrapper(num_classes, pretrained, freeze_backbone)
    else:
        raise NotImplementedError(f"{model_name}은 아직 구현되지 않았습니다.")