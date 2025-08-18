# app.py (Streamlit UI for Hazard Classifier: awl/knife/scissor)
# %%writefile app.py
import os, io, json, time, itertools, subprocess
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# =================== [NEW] 자동 다운로드 + 튼튼한 로더 ===================
MODEL_LOCAL_FALLBACK = "hazard_resnet50.keras"  # 모델 경로를 못 찾으면 여기에 다운로드

def ensure_model_via_gdown(local_path: str, env_key: str = "MODEL_FILE_ID"):
    """
    - local_path가 존재하면 그대로 반환
    - 없으면 환경변수(또는 Streamlit Secrets)에 있는 Google Drive file id로 gdown 다운로드
    - 성공 시 local_path 반환, 실패 시 None
    """
    try:
        if os.path.exists(local_path):
            return local_path
        file_id = os.environ.get(env_key, "").strip()
        if not file_id:
            return None
        url = f"https://drive.google.com/uc?id={file_id}"
        subprocess.run(["gdown", url, "-O", local_path], check=True)
        return local_path if os.path.exists(local_path) else None
    except Exception as e:
        st.warning(f"gdown 다운로드 실패: {e}")
        return None

def load_model_robust(path: str):
    """
    - 우선 keras.saving.load_model(.keras) 사용
    - 실패 시 tf.keras.models.load_model(..., compile=False) 재시도
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"모델 경로가 존재하지 않습니다: {path}")
    # .keras (Keras3) 우선
    try:
        import keras
        return keras.saving.load_model(path)
    except Exception:
        # tf.keras 로더로 재시도
        return tf.keras.models.load_model(path, compile=False)
# =======================================================================

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
