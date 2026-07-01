# 경구약제 이미지 객체 검출 (Pill Object Detection)

알약 이미지에서 객체를 검출하는 모델을 학습하고 평가합니다.  
데이터는 COCO 포맷 JSON annotation과 PNG 이미지로 구성되어 있습니다.

---

## 프로젝트 구조

```
project/
├── data/                    # 데이터셋 (Git 제외 — 아래 데이터 준비 참고)
│   ├── train_images/
│   ├── train_annotations/
│   └── test_images/
├── src/
│   ├── dataset.py           # COCO 데이터 로드 및 전처리
│   ├── model.py             # Detection 모델 (swin_fpn, swin_detr, rt_detr, dino)
│   ├── train.py             # K-Fold 학습 루프 및 mAP 평가
│   ├── visualize.py         # 예측 결과 시각화 및 오류 분석
│   └── utils.py
├── configs/
│   ├── swin_fpn.yaml        # Swin-T + FPN + Faster RCNN
│   ├── rt_detr.yaml         # RT-DETR V2 (ResNet-50)
│   ├── swin_detr.yaml       # Swin-T + DETR (미사용 — 아래 참고)
│   └── dino.yaml            # DINO 근사 구현 (미사용 — 아래 참고)
├── outputs/                 # 학습 결과 (Git 제외)
│   └── weights/
├── notebooks/               # EDA, 실험용
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 데이터 준비

데이터는 용량(1.93GB)이 크기 때문에 레포지토리에 포함되어 있지 않습니다.  
아래 방법으로 직접 다운로드 후 배치해주세요.

1. [Kaggle 대회 데이터 페이지](https://www.kaggle.com/competitions/ai12-level1-project/data) 접속
2. **Download All** 클릭하여 압축 파일 다운로드
3. 압축 해제 후 아래 구조로 `data/` 폴더에 배치

```
data/
├── train_images/       # 훈련 이미지 (PNG, 232개)
├── train_annotations/  # 훈련 annotation (COCO JSON, 763개)
└── test_images/        # 테스트 이미지 (PNG, 842개)
```

또는 코드에서 자동 다운로드:

```python
from src.dataset import download_data
path = download_data()  # Kaggle 캐시에 자동 저장
```

---

## 데이터 분석 (EDA) 요약

### 데이터 구성

| 항목 | 수량 |
|---|---|
| 전체 이미지 | 232장 |
| 고유 이미지 (학습 사용) | 114장 |
| 테스트 이미지 | 842장 |
| 클래스 수 | 56종 |
| 전체 bbox | 763개 |
| 이미지당 평균 알약 수 | 3.3개 |

### 주요 발견

- **촬영 조건**: 같은 구성의 알약을 카메라 위도 70°/75°/90°로 각각 촬영하여 중복 이미지 존재
- **고유 이미지 사용**: 같은 구성의 알약을 여러 각도로 촬영한 중복 이미지 제거 → `unique_only=True` 옵션으로 구성당 가장 낮은 위도 이미지 1장만 사용 (114장)
- **클래스 불균형**: 최다 클래스 157개 ~ 최소 클래스 3개로 편차 큼
- **데이터 누수 방지**: 고유 이미지 114장을 구성 단위 Group K-Fold로 train/val 분리

---

## Annotation 수정 내역

원본 데이터의 annotation 오류를 수정한 내역이 `corrections.json`에 기록되어 있습니다.  
`PillDataset` 로드 시 자동으로 적용됩니다.

| 유형 | 건수 |
|---|---|
| bbox 좌표 오류 수정 | 6건 |
| 누락 bbox 추가 | 8건 |

---

## 모델

총 4가지 모델을 구현하였으며, 실험 결과에 따라 `swin_fpn`과 `rt_detr`을 주력 모델로 사용합니다.

### swin_fpn (주력)

Swin-T 백본 + FPN + Faster RCNN 헤드 구성입니다.

- 백본: `torchvision.models.swin_t` (ImageNet pretrained, freeze)
- 헤드: Faster RCNN (RPN + RoI Align + 분류/회귀)
- 입력: xyxy 절대 좌표
- 특징: 표준 2-stage 구조로 안정적인 학습

### rt_detr (주력)

RT-DETR V2 (Real-Time Detection Transformer) 구성입니다.

- 백본: ResNet-50-vd (COCO pretrained)
- 체크포인트: `PekingU/rtdetr_v2_r50vd` → 56클래스로 헤드 재초기화
- 입력: COCO [x, y, w, h] → 모델 내부에서 정규화 cx, cy, w, h 변환
- 특징: COCO pretrained transformer 가중치를 fine-tune하므로 수십 epoch 내 수렴

### swin_detr (미사용)

Swin-T 백본 + 원본 DETR 헤드 구성입니다.

- 백본: timm `swin_tiny_patch4_window7_224` (ImageNet pretrained, freeze)
- DETR transformer: **랜덤 초기화** (COCO pretrained 없음)
- 실험 결과: 200 epoch 학습 시 Loss는 감소하나 **mAP = 0.000** 유지

**mAP가 오르지 않는 이유:**

원본 DETR 논문은 COCO 데이터셋 기준 **500 epoch** 학습을 기준으로 합니다. DETR transformer가 랜덤 초기화 상태에서 시작하므로 200 epoch은 수렴에 턱없이 부족합니다. Loss가 감소하는 것은 Hungarian matching으로 인해 분류 loss는 줄어들지만, 박스 regression은 100개 쿼리 전부 이미지 중앙(cx ≈ 0.5)에서 벗어나지 못해 GT와 IoU가 낮은 상태가 지속됩니다. 또한 backbone이 freeze되어 detection에 최적화되지 않은 특징맵이 고정되므로 decoder 학습이 더욱 어렵습니다.

개선 방향(`facebook/detr-resnet-50` pretrained 사용)을 적용하더라도, 이미 동작 중인 rt_detr이 같은 ResNet-50 기반에 더 발전된 아키텍처(2024)로 성능이 높아 추가적인 실익이 없습니다.

### dino (미사용)

DINO (DETR with Improved DeNoising Anchor Boxes) 근사 구현입니다.

- 내부적으로 `SwinDetrWrapper`를 상속하여 쿼리 수만 100 → 300으로 늘린 구성
- DINO의 핵심 기법인 Contrastive DeNoising(CDN), Mixed Query Selection, Anchor point queries가 구현되어 있지 않아 실질적으로는 **쿼리가 3배 많은 swin_detr**과 동일
- swin_detr의 수렴 문제를 그대로 가지며, 쿼리가 많을수록 collapse가 더 심화될 수 있어 실험에서 제외

---

## 환경 설정

**1. 가상환경 생성 및 활성화**

```bash
conda create -n pill-detection python=3.13.3
conda activate pill-detection
```

**2. 패키지 설치**

```bash
pip install -r requirements.txt
```

---

## 학습 실행

```bash
# Swin-T + FPN + Faster RCNN
python src/train.py --config configs/swin_fpn.yaml

# RT-DETR V2
python src/train.py --config configs/rt_detr.yaml
```

---

## 데이터셋 출처

- Kaggle 대회: [AI12 경구약제 이미지 객체 검출](https://www.kaggle.com/competitions/ai12-level1-project/data)
- 원본 데이터: [AI Hub 경구약제 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=576)
