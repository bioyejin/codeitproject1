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
│   ├── dataset.py           # COCO 데이터 로드
│   ├── model.py             # Detection 모델
│   ├── train.py             # 학습 루프
│   └── utils.py
├── configs/
│   └── config.yaml          # 하이퍼파라미터 설정
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
python src/train.py --config configs/config.yaml
```

---

## 데이터셋 출처

- Kaggle 대회: [AI12 경구약제 이미지 객체 검출](https://www.kaggle.com/competitions/ai12-level1-project/data)
- 원본 데이터: [AI Hub 경구약제 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=576)