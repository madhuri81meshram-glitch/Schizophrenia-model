import streamlit as st

# MUST be first!
st.set_page_config(
    page_title="Multislice MRI Schizophrenia Classifier",
    layout="centered"
)

import os
import tempfile
import numpy as np
import nibabel as nib
import cv2
import joblib
import lightgbm as lgb
import pywt
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image

# -----------------------------------------------------
# CONFIG (USE YOUR WINDOWS PATHS)
# -----------------------------------------------------
# Use relative paths inside the repo (works on Streamlit Cloud)
MODEL_TXT = "lightgbm_multislice_model.txt"
SCALER_JOBLIB = "scaler.joblib"

#MODEL_TXT = r"C:\Users\NABIRA\OneDrive\Desktop\mri\24\lightgbm_multislice_model.txt"
#SCALER_JOBLIB = r"C:\Users\NABIRA\OneDrive\Desktop\mri\24\scaler (2).joblib"

AXIAL = [-2, -1, 0, 1, 2]
CORONAL = [-2, -1, 0, 1, 2]
SAGITTAL = [-2, -1, 0, 1, 2]

EFF_INPUT = (224, 224)

# -----------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_TXT):
        st.error(f"LightGBM model missing: {MODEL_TXT}")
        st.stop()
    if not os.path.exists(SCALER_JOBLIB):
        st.error(f"Scaler missing: {SCALER_JOBLIB}")
        st.stop()

    model = lgb.Booster(model_file=MODEL_TXT)
    scaler = joblib.load(SCALER_JOBLIB)
    effnet = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    return model, scaler, effnet

model, scaler, effnet = load_models()

# -----------------------------------------------------
# FEATURE FUNCTIONS
# -----------------------------------------------------
def normalize01(img):
    img = img.astype(np.float32)
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)

def safe_slice(data, axis, idx):
    idx = np.clip(idx, 0, data.shape[axis] - 1)
    if axis == 2:
        return data[:, :, idx]
    if axis == 1:
        return data[:, idx, :]
    return data[idx, :, :]

def compute_glcm(img_uint8):
    glcm = graycomatrix(img_uint8, [1], [0], 256, symmetric=True, normed=True)
    out = {}
    for p in ["contrast","dissimilarity","homogeneity","energy","correlation","ASM"]:
        try:
            out[f"glcm_{p}"] = float(graycoprops(glcm, p).mean())
        except:
            out[f"glcm_{p}"] = 0.0
    return out

def compute_hist(img_uint8):
    hist, _ = np.histogram(img_uint8.flatten(), bins=20, range=(0,255))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
    data = {f"hist_bin_{i}": float(v) for i, v in enumerate(hist)}
    data["hist_mean"] = float(img_uint8.mean())
    data["hist_std"] = float(img_uint8.std())
    data["hist_skew"] = float(pd.Series(img_uint8.flatten()).skew())
    data["hist_kurt"] = float(pd.Series(img_uint8.flatten()).kurt())
    return data

def compute_dwt(img):
    coeffs = pywt.wavedec2(img, "db1", level=2)
    out = {}
    for i, c in enumerate(coeffs):
        if i == 0:
            out[f"dwt_LL_energy_0"] = float(np.sum(np.square(c)))
        else:
            for j, sub in enumerate(c):
                out[f"dwt_energy_{i}_{j}"] = float(np.sum(np.square(sub)))
    return out

def effnet_feat(img_uint8):
    rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, EFF_INPUT)
    arr = image.img_to_array(rgb)[None]
    arr = preprocess_input(arr)
    return effnet.predict(arr, verbose=0)[0]

# -----------------------------------------------------
# MULTISLICE EXTRACTION
# -----------------------------------------------------
def extract_multislice(nifti_path):
    vol = nib.load(nifti_path).get_fdata()

    cx, cy, cz = np.array(vol.shape) // 2
    planes = [
        (2, cz, AXIAL),
        (1, cy, CORONAL),
        (0, cx, SAGITTAL)
    ]

    slice_feats = []

    for axis, center, offs in planes:
        for off in offs:
            sl = safe_slice(vol, axis, center + off)
            sln = normalize01(sl)
            u8 = (sln * 255).astype(np.uint8)

            feats = {}
            feats.update(compute_glcm(u8))
            feats.update(compute_hist(u8))
            feats.update(compute_dwt(sln))

            deep = effnet_feat(u8)
            for i, v in enumerate(deep):
                feats[f"deep_{i}"] = float(v)

            slice_feats.append(feats)

    # aggregate (mean & std)
    numeric = list(slice_feats[0].keys())
    agg = {}
    for k in numeric:
        vals = [s[k] for s in slice_feats]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg

# -----------------------------------------------------
# STREAMLIT UI
# -----------------------------------------------------
st.title("🧠 Multislice MRI Schizophrenia Classifier")

uploaded = st.file_uploader("Upload MRI (.nii or .nii.gz)", type=["nii","nii.gz"])

if uploaded:
    st.info("Processing MRI… This may take 20–60 seconds.")
    
    # SAFE Windows-compatible temp writing
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
    tmp_path = tmp.name
    tmp.close()

    with open(tmp_path, "wb") as f:
        f.write(uploaded.read())

    try:
        agg = extract_multislice(tmp_path)
    except Exception as e:
        st.error(f"Error reading NIfTI: {e}")
        st.stop()

    keys = sorted(agg.keys())
    X = np.array([agg[k] for k in keys]).reshape(1, -1)

    expected = scaler.mean_.shape[0]
    if X.shape[1] != expected:
        st.error(f"❌ Feature mismatch: {X.shape[1]} vs expected {expected}")
        st.stop()

    Xs = scaler.transform(X)
    prob = float(model.predict(Xs)[0])
    label = "Schizophrenia" if prob >= 0.5 else "Healthy Brain (Control)"

    st.subheader("Prediction Result")
    if prob >= 0.5:
        st.error(f"🧠 {label} — Probability: {prob:.4f}")
    else:
        st.success(f"🧠 {label} — Probability: {prob:.4f}")

    # Cleanup
    try:
        os.remove(tmp_path)
    except:
        pass

st.write("---")
st.caption("EfficientNet + GLCM + Histogram + DWT | Multislice Aggregation | LightGBM Classifier")
