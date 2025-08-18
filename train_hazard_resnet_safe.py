import os, json, itertools, shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

# ============== 설정 ==============
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

# ✅ 원본 데이터 루트(분할 구조여도/아니어도 OK)
DATA_DIR = "/content/drive/MyDrive/project_2nd/dataset_build_v1/splits_3cls_525"

IMG_SIZE = (224, 224)
BATCH = 32                 # ResNet50은 메모리 사용량이 커서 32부터 권장
EPOCHS_STAGE1 = 12         # 동결 단계
EPOCHS_STAGE2 = 5          # 미세조정 단계
VAL_SPLIT_FALLBACK = 0.2   # 분할 폴더가 없으면 자동 분할 비율

# ★ 라벨 (폴더명과 정확히 일치)
CLASSES = ["awl","knife","scissor"]
NUM_CLASSES = len(CLASSES)

# ===== 출력 경로: 로컬에 저장(드라이브 끊김 방지) =====
OUTPUT_DIR = "/content/hazard_resnet_runs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_PATH   = os.path.join(OUTPUT_DIR, "hazard_resnet50.keras")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "history.json")
CM_PNG       = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
ART_DIR      = os.path.join(OUTPUT_DIR, "artifacts_hazard")
os.makedirs(ART_DIR, exist_ok=True)

# (옵션) 학습 종료 후 드라이브로 백업
BACKUP_TO_DRIVE = True
DRIVE_SAVE_DIR = "/content/drive/MyDrive/project_2nd/hazard_exports_resnet"

# ============== 유틸 ==============
ALLOWED_EXT = {".jpg",".jpeg",".png",".bmp",".gif"}  # TF 디코더가 읽을 수 있는 확장자

def has_phase_dirs(data_dir: str) -> bool:
    return all(tf.io.gfile.isdir(os.path.join(data_dir, p)) for p in ["train","val","test"])

def is_valid_image(path: str) -> bool:
    """깨진 이미지/빈 파일/숨김/리소스포크(._) 필터링"""
    name = os.path.basename(path)
    if name.startswith(".") or name.startswith("._"):
        return False
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXT:
        return False
    try:
        # 0바이트 파일 컷
        if tf.io.gfile.stat(path).length == 0:
            return False
    except Exception:
        return False
    # PIL로 실제 디코딩 가능한지 검증
    try:
        with tf.io.gfile.GFile(path, "rb") as f:
            with Image.open(f) as img:
                img.verify()  # 헤더 검증
        return True
    except Exception:
        return False

def sanitize_copy(src_root: str, dst_root: str, expect_phases: bool):
    """원본에서 유효한 이미지만 dst_root로 계층 복사(clone)"""
    if tf.io.gfile.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root, exist_ok=True)

    total_kept = 0; total_bad = 0

    if expect_phases:
        phases = ["train","val","test"]
        for ph in phases:
            for cls in CLASSES:
                src_dir = os.path.join(src_root, ph, cls)
                dst_dir = os.path.join(dst_root, ph, cls)
                os.makedirs(dst_dir, exist_ok=True)
                if not tf.io.gfile.isdir(src_dir):
                    print(f"[!] Missing: {src_dir} (skip)")
                    continue
                for name in tf.io.gfile.listdir(src_dir):
                    src = os.path.join(src_dir, name)
                    if tf.io.gfile.isdir(src):
                        continue
                    if is_valid_image(src):
                        shutil.copy2(src, dst_dir)
                        total_kept += 1
                    else:
                        total_bad += 1
    else:
        # 클래스 바로 아래에 이미지가 있는 구조 (자동 분할 모드로 사용할 예정)
        for cls in CLASSES:
            src_dir = os.path.join(src_root, cls)
            dst_dir = os.path.join(dst_root, cls)
            os.makedirs(dst_dir, exist_ok=True)
            if not tf.io.gfile.isdir(src_dir):
                print(f"[!] Missing: {src_dir} (skip)")
                continue
            for name in tf.io.gfile.listdir(src_dir):
                src = os.path.join(src_dir, name)
                if tf.io.gfile.isdir(src):
                    continue
                if is_valid_image(src):
                    shutil.copy2(src, dst_dir)
                    total_kept += 1
                else:
                    total_bad += 1

    print(f"🧹 Sanitize done → kept {total_kept} files, filtered {total_bad} bad files.")
    return dst_root

def make_ds_from_phases(data_dir, subdir, shuffle=True):
    return tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, subdir),
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=shuffle,
        seed=SEED,
    )

def make_auto_split_ds(data_dir, val_split=0.2):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=True,
        seed=SEED,
        validation_split=val_split,
        subset="training",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=False,
        seed=SEED,
        validation_split=val_split,
        subset="validation",
    )
    # 별도 test 없으면 임시로 val 사용 (권장: 별도 test 세트 유지)
    test_ds = val_ds
    print("[!] No train/val/test phases detected → using AUTO-SPLIT (test = val TEMP).")
    return train_ds, val_ds, test_ds

AUTOTUNE = tf.data.AUTOTUNE
def prep(ds, augment=False):
    ds = ds.map(lambda x,y: (preprocess_input(tf.image.resize(x, IMG_SIZE)), y),
                num_parallel_calls=AUTOTUNE)
    if augment:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.15),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x,y: (aug(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

# ============== 데이터 준비(클린 복사 후 로드) ==============
if not tf.io.gfile.exists(DATA_DIR):
    raise FileNotFoundError(f"[X] DATA_DIR not found: {DATA_DIR}")

CLEAN_ROOT = "/content/clean_dataset_resnet"
if has_phase_dirs(DATA_DIR):
    CLEAN_DIR = sanitize_copy(DATA_DIR, CLEAN_ROOT, expect_phases=True)
    # 구조 검증
    for ph in ["train","val","test"]:
        for cls in CLASSES:
            assert tf.io.gfile.isdir(os.path.join(CLEAN_DIR, ph, cls)), f"Missing: {ph}/{cls}"
    train_ds = make_ds_from_phases(CLEAN_DIR, "train", shuffle=True)
    val_ds   = make_ds_from_phases(CLEAN_DIR, "val",   shuffle=False)
    test_ds  = make_ds_from_phases(CLEAN_DIR, "test",  shuffle=False)
else:
    CLEAN_DIR = sanitize_copy(DATA_DIR, CLEAN_ROOT, expect_phases=False)
    train_ds, val_ds, test_ds = make_auto_split_ds(CLEAN_DIR, VAL_SPLIT_FALLBACK)

train_ds = prep(train_ds, augment=True)
val_ds   = prep(val_ds,   augment=False)
test_ds  = prep(test_ds,  augment=False)

# ============== 모델 ==============
base = ResNet50(include_top=False, weights="imagenet", input_shape=IMG_SIZE+(3,))
base.trainable = False  # 1단계: 특징 추출기 동결

inputs = layers.Input(shape=IMG_SIZE+(3,))
x = base(inputs, training=False)           # BN 고정
x = layers.GlobalAveragePooling2D()(x)     # feature map → 1D
x = layers.Dropout(0.3)(x)                 # 과적합 방지
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

cbs = [
    callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                              save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                            restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=2, verbose=1),
]

# ===== 1) 동결 단계 =====
hist = model.fit(train_ds, validation_data=val_ds,
                 epochs=EPOCHS_STAGE1, callbacks=cbs)

# 로그 저장
with open(HISTORY_PATH, "w") as f:
    json.dump(hist.history, f, indent=2)

# ===== 2) 미세조정 (conv5_x만 부분 해제 권장) =====
# 케라스 ResNet50에서 대략 143 이후가 conv5_block1 부근
unfreeze_from = 143
for l in base.layers[unfreeze_from:]:
    l.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(5e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=EPOCHS_STAGE2, callbacks=cbs)

# ============== 평가 ==============
best = tf.keras.models.load_model(MODEL_PATH)
test_loss, test_acc = best.evaluate(test_ds)
print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

# 리포트/혼동행렬
y_true = np.concatenate([y.numpy() for _,y in test_ds], axis=0)
y_prob = best.predict(test_ds)
y_pred = y_prob.argmax(1)

print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
cm = confusion_matrix(y_true, y_pred)

# 혼동행렬 시각화 저장
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
ticks = np.arange(NUM_CLASSES)
plt.xticks(ticks, CLASSES, rotation=45, ha="right")
plt.yticks(ticks, CLASSES)
th = cm.max()/2
for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i,j], ha="center",
             color="white" if cm[i,j]>th else "black")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(CM_PNG, dpi=150)
print(f"Saved confusion matrix to {CM_PNG}")

# 라벨맵
with open(os.path.join(ART_DIR, "class_to_idx.json"), "w", encoding="utf-8") as f:
    json.dump({c:i for i,c in enumerate(CLASSES)}, f, ensure_ascii=False, indent=2)
print(f"Saved label map to {os.path.join(ART_DIR, 'class_to_idx.json')}")

# (옵션) 드라이브로 백업
if BACKUP_TO_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

        for fn in ["hazard_resnet50.keras", "history.json", "confusion_matrix.png"]:
            src = os.path.join(OUTPUT_DIR, fn)
            if os.path.exists(src):
                shutil.copy(src, DRIVE_SAVE_DIR)

        shutil.copytree(ART_DIR, os.path.join(DRIVE_SAVE_DIR, "artifacts_hazard"), dirs_exist_ok=True)
        print(f"✅ Copied outputs to: {DRIVE_SAVE_DIR}")
    except Exception as e:
        print("[!] Drive backup skipped due to error:", e)
