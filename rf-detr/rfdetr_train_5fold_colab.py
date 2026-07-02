# ============================================================
# RF-DETR 경구약제 검출 — 설치 ~ 학습(A~J) 통합
# 실행: [0 설치] → [1 마운트+경로] → (최초 [2-A] / 이후 [2-B]) → [3 학습]
# 매 세션 필수: [0][1] + zip 복원([2-B]). 런타임 끊기면 /content 전부 소실.
# 학습 전 런타임 GPU(L4) 고정. 중간 GPU 전환 시 재시작 → 처음부터.
# ============================================================

# === [0] 설치 (매 세션) ==========================================
!pip install -q "rfdetr[train,loggers]"    # RF-DETR 학습(train)+로깅(loggers) 의존성


# === [1] 마운트 + 경로 자동 탐색 (매 세션) ========================
from google.colab import drive                          # 코랩↔드라이브 연결 도구
drive.mount('/content/drive')                            # 드라이브 마운트

import os, glob                                          # 경로·탐색 도구
# 사람마다 '1팀 공유 문서' 유무가 달라 → sprint_ai_project1_data를 드라이브에서 자동 탐색
CANDS = [                                                # 흔한 후보 먼저
    '/content/drive/MyDrive/1팀 공유 문서/ai12-level1-project/sprint_ai_project1_data',
    '/content/drive/MyDrive/ai12-level1-project/sprint_ai_project1_data',
]
DATA_ROOT = next((c for c in CANDS if os.path.exists(c)), None)   # 존재하는 첫 경로 채택
if DATA_ROOT is None:                                    # 후보에 없으면 전체 재귀 검색
    hits = glob.glob('/content/drive/MyDrive/**/sprint_ai_project1_data', recursive=True)
    DATA_ROOT = hits[0] if hits else None
assert DATA_ROOT, "sprint_ai_project1_data를 못 찾음 — 공유 폴더 바로가기 확인"  # 못 찾으면 중단
PROJ_ROOT = os.path.dirname(DATA_ROOT)                   # .../ai12-level1-project (zip·백업 공통 상위)

TRAIN_IMG = os.path.join(DATA_ROOT, 'train_images')      # 학습 이미지 232장
TRAIN_ANN = os.path.join(DATA_ROOT, 'train_annotations') # 박스당 JSON 763개
TEST_IMG  = os.path.join(DATA_ROOT, 'test_images')       # 제출용 842장
print("DATA_ROOT:", DATA_ROOT)                           # 채택 경로 확인
for p in [TRAIN_IMG, TRAIN_ANN, TEST_IMG]:               # 세 경로 점검
    print(p, '->', os.path.exists(p))                    # 전부 True여야 함


# === [2-A] A단계: 5폴드 데이터 생성 — 최초 1회만 ===================
# dataset_5fold.zip 이미 있으면 이 블록 건너뛰고 [2-B]로.
import json, shutil                                      # JSON·복사 도구
import numpy as np                                       # 수치 도구
from collections import defaultdict                      # 박스 묶음 도구
from sklearn.model_selection import StratifiedGroupKFold # 그룹누수차단+층화 분할

OUT_DIR = "/content/dataset"                             # 출력 루트(로컬)
SEED, N_SPLITS = 42, 5                                   # 재현 seed / 5폴드

# --- 라벨 패치 4종 (육안검수 확정 13장 + 좌표오염 1건) ---
COORD_FIX = {"K-003351-016262-018357_0_2_0_2_75_000_200.png": [([6567,625,311,315],[567,625,311,315])]}  # 좌표오염
ADD_BOXES = {                                            # 누락 박스 추가(11장)
    "K-001900-016548-019607-033009_0_2_0_2_70_000_200.png": [(16548,[90,870,245,240])],
    "K-003351-013900-021325_0_2_0_2_70_000_200.png":[(3351,[400,830,180,180])],
    "K-003351-013900-036637_0_2_0_2_70_000_200.png":[(3351,[440,880,175,175])],
    "K-003351-020014-022074_0_2_0_2_90_000_200.png":[(20014,[65,720,325,315])],
    "K-003351-020238-031863_0_2_0_2_70_000_200.png":[(3351,[580,290,215,215])],
    "K-003351-021325-032310_0_2_0_2_90_000_200.png":[(32310,[595,830,345,245])],
    "K-003351-029667-031863_0_2_0_2_70_000_200.png":[(3351,[375,870,165,165])],
    "K-003351-032310-038162_0_2_0_2_70_000_200.png":[(3351,[390,855,185,185])],
    "K-003351-033880-038162_0_2_0_2_75_000_200.png":[(33880,[70,600,310,425])],
    "K-003351-035206-041768_0_2_0_2_70_000_200.png":[(3351,[460,875,180,180])],
    "K-003544-004543-012247-016548_0_2_0_2_90_000_200.png":[(4543,[640,195,205,190])],
}
REMOVE_BOXES = {"K-001900-016548-019607-033009_0_2_0_2_70_000_200.png": [(16548,[88,255,366,209])]}  # 중복 1개 제거
MODIFY_BOXES = {                                         # 좌표 수정(None=bbox무시, EXTEND_DOWN_95=h+95)
    "K-003351-020014-020238_0_2_0_2_90_000_200.png":[(3351,None,[390,260,170,165])],
    "K-003351-019232-029667_0_2_1_2_70_000_200.png":[(19232,None,"EXTEND_DOWN_95")],
}

# 1) 로드+병합 (박스당 JSON → file_name 기준)
boxes_by_image, cats_by_image, img_meta = defaultdict(list), defaultdict(list), {}
for p in glob.glob(os.path.join(TRAIN_ANN,"**","*.json"), recursive=True):
    d = json.load(open(p, encoding="utf-8")); im = d["images"][0]; fn = im["file_name"]
    img_meta[fn] = (im["width"], im["height"])
    for a in d["annotations"]:
        boxes_by_image[fn].append(a["bbox"]); cats_by_image[fn].append(a["category_id"])

# 1.5) 패치 적용 (보정→제거→수정→추가)
for fn,fx in COORD_FIX.items():
    for w,r in fx:
        for i,b in enumerate(boxes_by_image[fn]):
            if b==w: boxes_by_image[fn][i]=r
for fn,rm in REMOVE_BOXES.items():
    for rc,rb in rm:
        kb,kc,done=[],[],False
        for c,b in zip(cats_by_image[fn],boxes_by_image[fn]):
            if (not done) and c==rc and b==rb: done=True; continue
            kb.append(b); kc.append(c)
        boxes_by_image[fn],cats_by_image[fn]=kb,kc
for fn,md in MODIFY_BOXES.items():
    for mc,w,new in md:
        for i,(c,b) in enumerate(zip(cats_by_image[fn],boxes_by_image[fn])):
            if c==mc and (w is None or b==w):
                boxes_by_image[fn][i]=[b[0],b[1],b[2],b[3]+95] if new=="EXTEND_DOWN_95" else new; break
for fn,ad in ADD_BOXES.items():
    for ac,ab in ad: cats_by_image[fn].append(ac); boxes_by_image[fn].append(ab)

file_names = sorted(boxes_by_image.keys())
print("이미지", len(file_names), "/ 박스", sum(len(v) for v in boxes_by_image.values()), "(기대 232/773)")

# 2) category_id → 1-indexed 매핑 (RF-DETR: num_classes=max(id), 0=background 더미)
all_cats = sorted({c for cs in cats_by_image.values() for c in cs})
cat2label = {c:i+1 for i,c in enumerate(all_cats)}; label2cat = {i:c for c,i in cat2label.items()}
NUM_CLASSES = len(cat2label)

# 3) StratifiedGroupKFold 5폴드 (구성 114 그룹, 최희소 클래스 층화)
cls_freq = defaultdict(int)
for cs in cats_by_image.values():
    for c in cs: cls_freq[c]+=1
groups = np.array([fn.split("_0_2")[0] for fn in file_names])                # 그룹=구성코드
strat  = np.array([cat2label[min(cats_by_image[fn], key=lambda c: cls_freq[c])] for fn in file_names])
sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(file_names, strat, groups))                          # 5개 (train_idx, val_idx)

# 4) COCO 빌더 (id=0 더미 + 실제 1..N)
def build_coco(files):
    imgs, anns, aid = [], [], 1
    for iid, fn in enumerate(files, 1):
        W,H = img_meta[fn]; imgs.append({"id":iid,"file_name":fn,"width":W,"height":H})
        for c,b in zip(cats_by_image[fn], boxes_by_image[fn]):
            anns.append({"id":aid,"image_id":iid,"category_id":cat2label[c],
                         "bbox":[float(v) for v in b],"area":float(b[2]*b[3]),"iscrowd":0}); aid+=1
    cats = [{"id":0,"name":"pill","supercategory":"none"}] + \
           [{"id":cat2label[c],"name":str(c),"supercategory":"pill"} for c in all_cats]
    return {"images":imgs,"annotations":anns,"categories":cats}

# 5) 이미지 로컬 캐시 (드라이브 read 1회 → fold 복사는 로컬이라 빠름)
CACHE = "/content/img_cache"; os.makedirs(CACHE, exist_ok=True)
src_paths = {os.path.basename(p):p for p in glob.glob(os.path.join(TRAIN_IMG,"**","*.png"), recursive=True)}
for fn, src in src_paths.items(): shutil.copy(src, os.path.join(CACHE, fn))
print("이미지 캐시:", len(src_paths))

# 6) fold0~4 디렉토리 배치
for fi,(tr,va) in enumerate(folds):
    for idxs, split in [(tr,"train"),(va,"valid")]:
        files = [file_names[i] for i in idxs]
        d = os.path.join(OUT_DIR, f"fold{fi}", split); os.makedirs(d, exist_ok=True)
        json.dump(build_coco(files), open(os.path.join(d,"_annotations.coco.json"),"w"))
        for fn in files: shutil.copy(os.path.join(CACHE,fn), os.path.join(d,fn))
    print(f"fold{fi}: train {len(tr)} / valid {len(va)}")

# 7) 매핑 저장 + zip (PROJ_ROOT 밑에 저장 → 경로 자동 통일)
json.dump({"cat2label":{str(k):v for k,v in cat2label.items()},
           "label2cat":{str(k):v for k,v in label2cat.items()}}, open(os.path.join(OUT_DIR,"label_map.json"),"w"))
shutil.make_archive(os.path.join(PROJ_ROOT, "dataset_5fold"), "zip", OUT_DIR)  # PROJ_ROOT 사용
print("zip 저장 완료:", os.path.join(PROJ_ROOT, "dataset_5fold.zip"))


# === [2-B] A단계 복원 — zip 있을 때(2회차부터). [2-A] 대신 이것만 =====
ZIP = os.path.join(PROJ_ROOT, "dataset_5fold.zip")       # PROJ_ROOT 기준(경로 자동 통일)
print("zip 존재:", os.path.exists(ZIP))                  # True 확인
!cp "$ZIP" /content/                                     # 드라이브→로컬 복사
!unzip -qo /content/dataset_5fold.zip -d /content/dataset  # 압축 해제(-o=덮어쓰기)
print("복원 fold:", sorted(d for d in os.listdir("/content/dataset") if d.startswith("fold")))


# === [3] B~J: 5폴드 학습 + best 백업 + 이어하기 ====================
import torch                                             # GPU 도구
from rfdetr import RFDETRSmall                             # 모델 변형: Small (Medium=RFDETRMedium)

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")  # GPU 확인

# --- 하이퍼파라미터 (C / C-1 / E / J) ---
LR, LR_ENCODER, WEIGHT_DECAY        = 1e-4, 1.5e-4, 1e-4  # C: 학습률·encoder차등·L2
LR_SCHEDULER, WARMUP_EPOCHS, LR_MIN = "cosine", 0.0, 0.0  # C-1: 스케줄러·warmup·최저lr
EPOCHS, BATCH_SIZE, GRAD_ACCUM      = 100, 8, 2           # E: epoch·배치·누적(유효16, L4)
ES_PATIENCE, ES_MIN_DELTA           = 10, 0.001          # J: early stopping
MODEL_TAG = "small_res512"                                # 실험명 태그(변형 바꾸면 교체)

BACKUP = os.path.join(PROJ_ROOT, "outputs")              # best 백업(PROJ_ROOT 기준)
os.makedirs(BACKUP, exist_ok=True)

for fi in range(5):                                       # fold 0~4
    exp = f"{MODEL_TAG}_lr1e-4_fold{fi}"                   # D: 실험명(변수 하나만 바꿔 구분)
    dst = os.path.join(BACKUP, f"{exp}_best.pth")
    if os.path.exists(dst):                                # 이미 끝난 fold면
        print(f"[fold {fi}] 백업 존재 → 건너뜀"); continue  # 이어하기

    out = f"/content/outputs/{exp}"; os.makedirs(out, exist_ok=True)
    print(f"\n{'='*50}\n[fold {fi}] 학습 시작\n{'='*50}")

    model = RFDETRSmall()                                  # B: fold마다 새 모델(초기화)
    model.train(                                           # E~J: 학습·검증·best·earlystop·로깅 내장
        dataset_dir      = f"/content/dataset/fold{fi}",   # A: fold별 데이터
        output_dir       = out,
        epochs           = EPOCHS, batch_size = BATCH_SIZE, grad_accum_steps = GRAD_ACCUM,  # E
        lr = LR, lr_encoder = LR_ENCODER, weight_decay = WEIGHT_DECAY,                       # C
        lr_scheduler = LR_SCHEDULER, warmup_epochs = WARMUP_EPOCHS, lr_min_factor = LR_MIN,  # C-1
        early_stopping = True, early_stopping_patience = ES_PATIENCE,                        # J
        early_stopping_min_delta = ES_MIN_DELTA,                                             # J
        tensorboard = True,                                                                  # D
    )

    src = os.path.join(out, "checkpoint_best_total.pth")   # 학습이 저장한 best
    if os.path.exists(src):
        shutil.copy(src, dst); print(f"[fold {fi}] best 백업 → {dst}")
    del model; torch.cuda.empty_cache()                    # GPU 정리(OOM 방지)
    print(f"[fold {fi}] 완료")

print("\n▶ 5폴드 학습 완료 — best 5개 준비됨(앙상블용)")