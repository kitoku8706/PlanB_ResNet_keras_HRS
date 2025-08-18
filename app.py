import os, io, json, time, itertools
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --------------- UI 기본 설정 ---------------
st.set_page_config(page_title="Hazard Classifier UI", layout="wide")
st.title("🔪 Hazard Classifier (awl / knife / scissor)")
st.caption("ResNet50 / MobileNetV2 모델 체크포인트로 예측 · 시각화 · 리포트")

# --------------- 사이드바 설정 ---------------
with st.sidebar:
    st.header("⚙️ 설정")

    backbone = st.selectbox(
        "백본(전처리 선택)",
        options=["ResNet50", "MobileNetV2"],
        index=0,
        help="모델이 어떤 전처리를 썼는지에 따라 선택"
    )

    # Colab에서 학습한 기본 경로를 미리 채워둡니다(필요시 수정)
    default_model = "/content/hazard_resnet_runs/hazard_resnet50.keras"
    default_labelmap = "/content/hazard_resnet_runs/artifacts_hazard/class_to_idx.json"
    if backbone == "MobileNetV2":
        default_model = "/content/hazard_mobilenet_runs/hazard_mobilenetv2.keras"
        default_labelmap = "/content/hazard_mobilenet_runs/artifacts_hazard/class_to_idx.json"

    model_path = st.text_input("모델 경로(.keras)", value=default_model)
    labelmap_path = st.text_input("라벨맵 경로(class_to_idx.json)", value=default_labelmap)

    thresh = st.slider("불확실 임계치(↓면 과감, ↑면 보수)", min_value=0.0, max_value=0.99, value=0.75, step=0.01)
    topk = st.slider("Top-K 확률 표시", min_value=1, max_value=5, value=3, step=1)

    st.markdown("---")
    st.subheader("📂 폴더 일괄 예측 (선택)")
    batch_dir = st.text_input("폴더 경로(이미지들)", value="")
    show_grid = st.checkbox("그리드로 이미지/결과 미리보기", value=True)

    st.markdown("---")
    st.subheader("🧪 Test 폴더 리포트 (선택)")
    test_dir = st.text_input("Test 폴더 루트 (class별 하위폴더 구조)", value="/content/clean_dataset_resnet/test")

    st.markdown("---")
    st.caption("Tip: ResNet50은 resnet50 전처리, MobileNetV2는 mobilenet_v2 전처리를 사용해야 결과가 정확합니다.")

# --------------- 전처리 함수 ---------------
@st.cache_resource(show_spinner=False)
def get_preprocess(backbone_name: str):
    if backbone_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input

@st.cache_resource(show_spinner=False)
def load_model_safe(path: str):
    if not path or not os.path.exists(path):
        st.error(f"모델 파일을 찾을 수 없습니다: {path}")
        return None
    try:
        m = tf.keras.models.load_model(path)
        return m
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_labelmap_safe(path: str):
    if not path or not os.path.exists(path):
        st.error(f"라벨맵 파일을 찾을 수 없습니다: {path}")
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        # idx 순서대로 클래스 리스트
        classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
        return class_to_idx, idx_to_class
    except Exception as e:
        st.error(f"라벨맵 로드 실패: {e}")
        return None, None

def is_image_file(name: str):
    name = name.lower()
    return any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"])

def safe_open_image(fp: str):
    # 손상/숨김 파일 안전 핸들
    base = os.path.basename(fp)
    if base.startswith(".") or base.startswith("._"):
        raise UnidentifiedImageError("hidden/meta file")
    if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
        raise UnidentifiedImageError("empty or not file")
    with Image.open(fp) as im:
        im.verify()  # 헤더 검증
    im = Image.open(fp).convert("RGB")     # 실제 로딩
    return im

def preprocess_image(im: Image.Image, img_size=(224,224), preprocess=None):
    im_resized = im.resize(img_size)
    x = tf.keras.preprocessing.image.img_to_array(im_resized)
    x = preprocess(x)
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(model, preprocess, im: Image.Image, idx_to_class, threshold=0.75, topk=3):
    x = preprocess_image(im, (224,224), preprocess)
    prob = model.predict(x, verbose=0)[0]  # (num_classes,)
    order = np.argsort(prob)[::-1]
    top = [(idx_to_class[i], float(prob[i])) for i in order[:topk]]
    best_idx = int(order[0]); best_cls = idx_to_class[best_idx]; best_conf = float(prob[best_idx])
    label = best_cls if best_conf >= threshold else "uncertain"
    return label, best_conf, top, prob

# --------------- 모델/라벨맵 로드 ---------------
preprocess = get_preprocess(backbone)
model = load_model_safe(model_path)
class_to_idx, idx_to_class = load_labelmap_safe(labelmap_path)

# --------------- 단일 이미지 업로드 예측 ---------------
st.header("🖼️ 단일 이미지 예측")
uploaded_files = st.file_uploader("이미지 업로드 (여러 장 가능)", type=["jpg","jpeg","png","bmp","gif","webp"], accept_multiple_files=True)

if model and idx_to_class and uploaded_files:
    cols = st.columns(3)
    for i, uf in enumerate(uploaded_files):
        try:
            img = Image.open(io.BytesIO(uf.read())).convert("RGB")
            label, conf, top, _ = predict_image(model, preprocess, img, idx_to_class, threshold=thresh, topk=topk)
            with cols[i % 3]:
                st.image(img, caption=f"{uf.name}", use_column_width=True)
                st.markdown(f"**Pred:** `{label}`  |  **conf:** `{conf:.3f}`")
                st.markdown("Top-{}:".format(topk))
                for cls, p in top:
                    st.caption(f"- {cls}: {p:.3f}")
        except Exception as e:
            st.warning(f"{uf.name} 처리 실패: {e}")

# --------------- 폴더 일괄 예측 ---------------
st.header("📂 폴더 일괄 예측")
if model and idx_to_class and batch_dir and os.path.isdir(batch_dir):
    paths = [os.path.join(batch_dir, n) for n in os.listdir(batch_dir) if is_image_file(n)]
    paths.sort()
    st.write(f"이미지 {len(paths)}장 발견")

    preds = []
    grid_imgs, grid_caps = [], []
    start = time.time()
    for p in paths:
        try:
            im = safe_open_image(p)
            label, conf, top, _ = predict_image(model, preprocess, im, idx_to_class, threshold=thresh, topk=topk)
            preds.append({"path": p, "pred": label, "conf": conf, **{f"top{i+1}_cls": t[0] for i,t in enumerate(top)}, **{f"top{i+1}_prob": t[1] for i,t in enumerate(top)}})
            if show_grid and len(grid_imgs) < 24:
                grid_imgs.append(im.copy())
                grid_caps.append(f"{os.path.basename(p)}\n→ {label} ({conf:.2f})")
        except Exception as e:
            preds.append({"path": p, "pred": "error", "conf": 0.0})
    dur = time.time() - start
    st.success(f"완료: {len(paths)}장 / {dur:.1f}s")

    if show_grid and grid_imgs:
        cols = st.columns(6)
        for i, (im, cap) in enumerate(zip(grid_imgs, grid_caps)):
            with cols[i % 6]:
                st.image(im, caption=cap, use_column_width=True)

    # 결과 테이블 & 다운로드
    import pandas as pd
    df = pd.DataFrame(preds)
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 결과 CSV 다운로드", data=csv, file_name="batch_predictions.csv", mime="text/csv")

# --------------- Test 폴더 리포트 ---------------
st.header("🧪 Test 폴더 정확도/리포트")
if model and class_to_idx and idx_to_class and test_dir and os.path.isdir(test_dir):
    classes = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

    img_paths, y_true = [], []
    for cls in classes:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir): 
            continue
        for name in os.listdir(cls_dir):
            p = os.path.join(cls_dir, name)
            if is_image_file(p):
                try:
                    safe_open_image(p)  # 검증
                    img_paths.append(p)
                    y_true.append(class_to_idx[cls])
                except Exception:
                    pass  # 손상/숨김은 건너뜀

    if len(img_paths) == 0:
        st.warning("유효한 이미지가 없습니다.")
    else:
        y_pred = []
        for p in img_paths:
            im = Image.open(p).convert("RGB")
            label, conf, top, _ = predict_image(model, preprocess, im, idx_to_class, threshold=0.0, topk=topk)  # 임계치 없이 순수 예측
            # 'uncertain'이 나올 수 있으므로, 가장 높은 확률의 실제 클래스 매핑 필요
            # predict_image에서 이미 best class를 label로 반환하므로 label이 클래스면 OK, 'uncertain'이면 top1 사용
            if label == "uncertain":
                label = top[0][0]
            # 클래스명을 index로 변환
            pred_idx = [k for k,v in class_to_idx.items() if k == label]
            if pred_idx:
                y_pred.append(class_to_idx[label])
            else:
                # unknown fall-back (거의 없음): top1로 강제
                y_pred.append(class_to_idx[top[0][0]])

        # 리포트
        report = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=False)
        st.text("Classification Report\n" + report)

        # 정확도
        acc = (np.array(y_true) == np.array(y_pred)).mean()
        st.metric("Test Accuracy", f"{acc:.4f}")

        # 혼동행렬
        cm = confusion_matrix(y_true, y_pred, labels=[class_to_idx[c] for c in classes])

        fig = plt.figure(figsize=(6,5))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()
        ticks = np.arange(len(classes))
        plt.xticks(ticks, classes, rotation=45, ha="right")
        plt.yticks(ticks, classes)
        th = cm.max()/2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > th else "black")
        plt.ylabel("True")
        plt.xlabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig)

# --------------- 푸터 ---------------
st.markdown("---")
st.caption("✅ 모델은 과적합 시작 직전(Val 최고점) 체크포인트 사용을 권장합니다.")
