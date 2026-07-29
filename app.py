import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Brain Tumor Detection", page_icon="\U0001F9E0", layout="centered")

MODEL_PATH = "brain_tumor_model.joblib"
SCALER_PATH = "brain_tumor_scaler.joblib"
FEATURES_PATH = "brain_tumor_features.joblib"

@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features

model, scaler, feature_cols = load_artifacts()

st.title("\U0001F9E0 Brain Tumor Detection (Tabular Features)")
st.write(
    "Enter extracted MRI image features below (first-order statistics and "
    "GLCM texture features) to predict tumor presence."
)

st.warning(
    "\u26A0\uFE0F **Academic demo only \u2014 not a medical device.** "
    "Predictions are based on a model trained on a public dataset and should "
    "never be used for real diagnostic decisions."
)

if model is None:
    st.error(
        "Model artifacts not found. Place `brain_tumor_model.joblib`, "
        "`brain_tumor_scaler.joblib`, and `brain_tumor_features.joblib` "
        "next to `app.py` before running."
    )
    st.stop()

st.markdown("---")

# Build an input form dynamically from the saved feature list
with st.form("prediction_form"):
    st.subheader("Input Features")
    input_values = {}
    cols = st.columns(2)
    for i, feat in enumerate(feature_cols):
        with cols[i % 2]:
            input_values[feat] = st.number_input(feat, value=0.0, format="%.6f")
    submitted = st.form_submit_button("\U0001F50D Predict")

if submitted:
    X_input = pd.DataFrame([input_values])[feature_cols]
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    confidence = proba[int(prediction)] * 100

    st.markdown("---")
    if prediction == 1:
        st.error("\U0001F9E0 **Brain Tumor Detected**")
    else:
        st.success("\u2705 **No Brain Tumor Detected**")

    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
    st.progress(int(confidence))

st.markdown("---")
st.caption("Developed as an academic project | Tabular ML Brain Tumor Detection System")
