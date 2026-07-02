# RF-DETR 학습 파이프라인 (Colab)

알약 검출을 RF-DETR로 5-fold 학습하기 위한 코드입니다.
원래 Colab 노트북 셀 하나에 통합되어 있던 `rfdetr_train_5fold_colab.py`를
`model.py` / `train.py` / `config.yaml` 등 역할별 파일로 분리한 것입니다.

이 폴더는 별도 레포지토리로 분리될 예정이라 프로젝트 루트의 `src/`와는 완전히
무관하게(어떤 모듈도 import하지 않고) 동작합니다. RF-DETR(`rfdetr` 패키지, PyTorch
Lightning 기반)은 `model.train(...)` 한 번 호출로 데이터 로딩·optimizer·scheduler·
early stopping·체크포인트를 전부 내부에서 처리하는 자체 완결형 트레이너라서, `src/`의
공용 학습 루프(`train_one_epoch`/`evaluate`/`train_model`)가 기대하는
`forward(images, targets)` 계약과 맞지 않기 때문입니다.

---

## 폴더 구조

| 파일 | 역할 | 원본 스크립트 위치 |
|---|---|---|
| `corrections.json` | 라벨 보정 데이터 (좌표 오염 수정 / 중복 제거 / 좌표 수정 / 누락 박스 추가) | `[2-A]` 하드코딩 dict |
| `dataset.py` | annotation 로드, corrections 적용, StratifiedGroupKFold 분할, fold 디렉토리 생성/zip 백업/복원, label_map 저장·로드 | `[1]` 경로 탐색, `[2-A]` 5-fold 생성, `[2-B]` 복원 |
| `model.py` | `get_rfdetr_model(variant, checkpoint_path=None)` — RFDETR 변형 생성 + 체크포인트 로드 | `[3]`의 `from rfdetr import RFDETRSmall` |
| `visualize.py` | 예측 수집(`collect_predictions_from_coco`), mAP 계산(`evaluate_from_data`), 오답 시각화(`visualize_errors_from_data`, 원스텝 래퍼 `visualize_errors`) | 신규 (src/visualize.py + src/train.py 로직 이식) |
| `utils.py` | 학습 곡선 구성(`read_metrics_csv`, `plot_history`), fold별 리포팅(`report_fold_result`), 5-fold 요약(`summarize_kfold_results`) | 신규 (src/utils.py + src/train.py 로직 이식) |
| `config.yaml` | 데이터 경로, 모델 변형, 하이퍼파라미터 (seed 포함) | `[1]~[3]`에 흩어져 있던 하드코딩 값 |
| `train.py` | `train_fold()` / `run_kfold()` — fold 루프, 이어하기(skip), best 백업, 학습 곡선 저장, fold별 리포팅 + 5-fold 요약 자동 실행 | `[3]` |
| `colab_setup.py` | 드라이브 마운트, `prepare_data()` / `restore_data()` 오케스트레이션 | `[0]`, `[1]`, `[2-A]`/`[2-B]` 진입점 |
| `plus.py` | `sanity_check()` — fold 1개 × epoch 1회로 학습→체크포인트→fold 리포팅→추론→mAP→시각화 전체 파이프라인 확인 | 신규 |
| `requirements.txt` | 이 폴더가 실제로 import하는 모듈 기준 의존성 목록 | 신규 |
| `rfdetr_train_5fold_colab.py` | 원본 통합 스크립트 (참고용으로 보존, 더 이상 직접 실행 대상 아님) | — |

### 모듈 의존 방향 (순환 import 없음)

```
model.py, dataset.py, visualize.py   <- 서로 독립 (외부 패키지만 사용)
utils.py       -> model.py, visualize.py
train.py       -> model.py, dataset.py, utils.py
plus.py        -> train.py, model.py, visualize.py
```

---

## 사용 흐름 (Colab 기준)

```python
# 0) 노트북 셀에서 직접 실행 (코드로 감싸지 않음)
!pip install -q "rfdetr[train,loggers]"

# 1) 작업 디렉토리를 rf-detr로 이동한 뒤 (플랫 import를 위해 필요)
%cd /content/.../rf-detr

from colab_setup import mount_drive, prepare_data, restore_data
from train import load_config, run_kfold
from plus import sanity_check

mount_drive()
config = load_config('config.yaml')

# 2) 최초 1회: 5-fold 데이터 생성 + zip 백업
prepare_data(config)
# 이후 세션(zip 이미 있음)에는 대신:
# restore_data(config)

# 3) config.yaml의 output.backup_dir을 prepare_data() 출력 경로로 채운 뒤 --
#    본 학습 전에 파이프라인이 끝까지 도는지 빠르게 확인
sanity_check(max_folds=1, epochs=1)

# 4) 본 학습 (5-fold). fold마다 체크포인트 저장 + 학습 곡선 저장 +
#    클래스별 mAP 출력 + 오답 이미지 시각화가 자동 실행되고, 끝나면 5-fold 평균±표준편차 출력
run_kfold(config)
```

---

## corrections.json — 프로젝트 루트의 것과 다른 별도 파일

이 폴더의 `corrections.json`은 프로젝트 루트 `corrections.json`(6건 수정 + 8건 추가)과
**다른 독립적인 보정 세트**입니다. RF-DETR 스크립트가 자체적으로 정의해둔 것으로, 4종류로
구성됩니다.

| 유형 | 건수 | 내용 |
|---|---|---|
| `coord_fix` | 1건 | 좌표 오염 수정 (오타로 잘못 들어간 좌표 교체) |
| `remove_boxes` | 1건 | 중복 박스 제거 |
| `modify_boxes` | 2건 | 좌표 수정 (직접 교체 1건, 높이 +95 확장 1건) |
| `add_boxes` | 11건 | 누락 박스 추가 |

`dataset.apply_corrections()`가 `coord_fix → remove_boxes → modify_boxes → add_boxes`
순서로 적용합니다.

---

## mAP 지표 — 학습 곡선(epoch별)과 fold 최종 요약이 다른 기준을 씀

- **fold 최종 요약 / 클래스별 mAP** (`utils.report_fold_result`, `utils.summarize_kfold_results`):
  `visualize.evaluate_from_data()`가 직접 torchmetrics로 계산하므로, `src/train.py`와
  **완전히 동일한 정의**의 mAP@0.5:0.95 / mAP@0.5 / mAP@0.75:0.95(IoU 0.75~0.95 5개 지점
  평균)를 정확히 얻습니다. swin_fpn/rt_detr의 mAP@0.75:0.95와 직접 비교 가능합니다.
- **학습 곡선(epoch별 history)** (`utils.read_metrics_csv`, `utils.plot_history`):
  RF-DETR가 학습 중 자동으로 남기는 `{output_dir}/metrics.csv`(PTL CSVLogger, 항상
  기록됨)에서 읽어옵니다. 이 파일에는 `mAP_50_95`, `mAP_50`, `mAP_75`(IoU=0.75 **단일**
  지점)만 있고 `mAP_75:0.95`(5개 지점 평균)는 없습니다. 그래서 학습 곡선의 3번째 선은
  `mAP@0.75`(단일 지점)를 대신 씁니다 — **fold 최종 요약의 mAP@0.75:0.95와는 다른 지표**이니
  두 값을 서로 비교하지 않도록 주의하세요.

RF-DETR의 예전 `model.callbacks["on_fit_epoch_end"]` 콜백 방식은 v1.7.0부터
deprecated(v1.9.0에서 제거 예정)라 쓰지 않았습니다 — `metrics.csv`가 현재 권장되는 방식입니다.

---

## 호환성 / 주의사항

1. **셸 매직 제거**: 원본의 `!cp`, `!unzip`은 `shutil.copy` / `shutil.unpack_archive`로 대체했습니다.
   덕분에 노트북 셀뿐 아니라 일반 `.py` 스크립트로도 실행 가능합니다. (단 `!pip install`은
   패키지 설치라 코드로 대체하지 않고 `colab_setup.py` 상단에 안내만 남겨뒀습니다.)
2. **backup_dir은 수동 설정 필요**: 데이터 준비(`prepare_data`)와 학습(`run_kfold`)을 완전히
   분리했기 때문에, `PROJ_ROOT`(드라이브 경로)가 학습 단계까지 자동으로 전달되지 않습니다.
   `prepare_data()` 실행 후 콘솔에 출력되는 경로를 확인해 `config.yaml`의 `output.backup_dir`을
   직접 채워야 합니다 (최초 1회만).
3. **폴더명(`rf-detr`)에 하이픈이 있어 파이썬 패키지로 import 불가**: `from rf_detr import ...`
   같은 dotted import는 안 됩니다. `%cd rf-detr` 후 `from dataset import ...`처럼 플랫하게
   import하거나, `python train.py`로 직접 실행해야 합니다.
4. **`corrections_path` 기본값은 상대경로**: cwd가 `rf-detr` 폴더일 때를 기준으로 합니다.
   다른 위치에서 실행한다면 `config.yaml`에서 절대경로로 바꿔야 합니다.
5. **재현성**: `config.yaml`의 `train.seed`가 `model.train(seed=...)`로 전달됩니다
   (`TrainConfig.seed` 필드로 실제 존재함을 roboflow/rf-detr 소스에서 확인함). fold 분할
   시드(`data.seed`)와는 별개 값입니다.
6. **`get_rfdetr_model()`의 variant/체크포인트 로딩 방식**: `nano`/`small`/`medium`/`base`/`large`
   전부 rf-detr 소스의 `_VARIANT_EXPORTS`에 존재함을 확인했고, `pretrain_weights` 생성자
   인자로 체크포인트를 로드하는 것도 소스로 확인했습니다. `model.predict()`가
   `supervision.Detections`(`.xyxy`/`.confidence`/`.class_id`)를 반환하는 것도 확인됨.
7. **`unique_only`(중복 각도 제거) 미적용**: `src/dataset.py`의 `PillDataset`은 같은 구성을
   여러 각도로 찍은 중복 이미지를 제거해 114장만 쓰지만, 이 폴더의 `dataset.py`는 원본
   콜랩 스크립트 그대로 232장을 전부 사용합니다. `StratifiedGroupKFold`로 같은 구성이
   train/val에 섞이지 않게는 막혀 있어 데이터 누수는 없지만, 의도적으로 다르게 가는
   설계인지는 별도로 확인이 필요합니다.
