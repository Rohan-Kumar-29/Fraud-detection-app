import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Load the saved model
model = joblib.load("fraud_model.pkl")

st.title("🚨 Fraud Detection App")

# Upload CSV file
uploaded = st.file_uploader("Upload a CSV file with transactions", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    
    # Predict
    proba = model.predict_proba(df)[:, 1]
    df["fraud_proba"] = proba
    df["fraud_pred"] = (proba > 0.5).astype(int)
    
    # Show results
    st.subheader("Predictions")
    st.dataframe(df.head())
    
    # Download button
    st.download_button(
        "Download Full Predictions",
        df.to_csv(index=False),
        "fraud_predictions.csv",
        "text/csv"
    )
