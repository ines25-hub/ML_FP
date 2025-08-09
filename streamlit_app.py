# streamlit_app.py
"""
Streamlit app that:
- loads saved model + scaler + metadata from saved_model/
- allows user to upload CSV for prediction OR fill manual inputs
- shows predicted fault types and, if true labels present in uploaded test CSV, shows metrics and confusion matrices
Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import os

SAVE_DIR = 'saved_model'
MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.joblib')
IMPUTER_PATH = os.path.join(SAVE_DIR, 'imputer.joblib')
SCALER_PATH = os.path.join(SAVE_DIR, 'scaler.joblib')
META_PATH = os.path.join(SAVE_DIR, 'metadata.json')

@st.cache_data
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)
    return model, imputer, scaler, metadata

def preprocess_user_df(df_user, imputer, scaler, num_cols):
    X_num = df_user[num_cols]
    X_imp = imputer.transform(X_num)
    X_scaled = scaler.transform(X_imp)
    return X_scaled

st.set_page_config(page_title="Steel Plate Faults — Demo", layout="wide")
st.title("Steel Plate Faults Detection — Demo")

if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
    st.error(f"Saved model artifacts not found in {SAVE_DIR}. Run the training script first.")
    st.stop()

model, imputer, scaler, metadata = load_artifacts()
label_cols = metadata['label_cols']
num_cols = metadata['num_cols']

st.sidebar.header("Input")
mode = st.sidebar.selectbox("Mode", ['Upload CSV', 'Manual input'])

if mode == 'Upload CSV':
    uploaded = st.sidebar.file_uploader("Upload CSV (features + optional label columns)", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Uploaded sample (first 10 rows):")
        st.dataframe(df.head(10))

        # Ensure necessary feature columns are present
        missing_feats = [c for c in num_cols if c not in df.columns]
        if missing_feats:
            st.error(f"Missing expected feature columns: {missing_feats}")
        else:
            X_proc = preprocess_user_df(df, imputer, scaler, num_cols)
            preds = model.predict(X_proc)
            pred_df = pd.DataFrame(preds, columns=label_cols)
            st.subheader("Predictions (first 20 rows)")
            st.dataframe(pred_df.head(20))

            # If true labels present, compute metrics
            if all([c in df.columns for c in label_cols]):
                y_true = df[label_cols].values
                y_pred = preds
                p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
                micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
                st.write("**Micro F1:**", round(micro_f1, 4))
                # Per label
                tab = pd.DataFrame({
                    'label': label_cols,
                    'precision': p,
                    'recall': r,
                    'f1': f1
                })
                st.dataframe(tab)
                # Confusion matrices per label
                cms = multilabel_confusion_matrix(y_true, y_pred, labels=[0,1])
                st.subheader("Confusion Matrices (per label)")
                for idx, lbl in enumerate(label_cols):
                    cm = multilabel_confusion_matrix(y_true, y_pred)[idx]
                    fig, ax = plt.subplots()
                    ax.matshow(cm, cmap=plt.cm.Blues)
                    for (i, j), val in np.ndenumerate(cm):
                        ax.text(j, i, int(val), ha='center', va='center')
                    ax.set_title(lbl)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)

else:  # Manual input
    st.sidebar.write("Enter values for features (numeric).")
    manual_vals = {}
    for col in num_cols:
        # Provide a reasonable numeric input default
        manual_vals[col] = st.sidebar.number_input(col, value=0.0)
    if st.sidebar.button("Predict"):
        df_manual = pd.DataFrame([manual_vals])
        X_proc = preprocess_user_df(df_manual, imputer, scaler, num_cols)
        pred = model.predict(X_proc)[0]
        pred_labels = [label_cols[i] for i, v in enumerate(pred) if v == 1]
        if not pred_labels:
            st.success("No faults predicted (all zero).")
        else:
            st.success("Predicted faults: " + ", ".join(pred_labels))

st.markdown("---")
st.info("Notes: This app loads the saved best model from the training step. If you retrain and save a new model, overwrite the files in the `saved_model/` folder.")
