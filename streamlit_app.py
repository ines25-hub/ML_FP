# app.py — Streamlit app for Steel Plate Fault Detection
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# Label columns
label_cols = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

st.title("🛠️ Steel Plate Fault Detection")
st.write("Upload steel plate measurements to detect possible faults.")

# Show model performance table
st.subheader("📊 Model Performance")
metrics_df = pd.read_csv("model_metrics.csv")
st.dataframe(metrics_df)

# File upload section
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Scale and predict
    data_scaled = scaler.transform(data)
    predictions = model.predict(data_scaled)
    predictions_df = pd.DataFrame(predictions, columns=label_cols)
    st.subheader("🔍 Predictions")
    st.dataframe(predictions_df)

# Manual input section
st.subheader("✏️ Enter Feature Values Manually")
manual_data = {}
feature_names = [col for col in pd.read_csv("Steel_Plates_Faults_Cleaned.csv").columns if col not in label_cols]

for feature in feature_names:
    manual_data[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict from Manual Input"):
    manual_df = pd.DataFrame([manual_data])
    manual_scaled = scaler.transform(manual_df)
    manual_prediction = model.predict(manual_scaled)
    manual_result = pd.DataFrame(manual_prediction, columns=label_cols)
    st.subheader("🔍 Prediction Result")
    st.dataframe(manual_result)
