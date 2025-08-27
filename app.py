import streamlit as st
import pandas as pd
import cloudpickle

st.set_page_config(page_title="Fraud Detection App", layout="wide")

# Load the model
MODEL_PATH = "fraud_model1.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = cloudpickle.load(f)
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

st.title("🚨 Fraud Detection App")

# Upload CSV
uploaded = st.file_uploader("Upload a CSV file with transactions", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    # Make predictions
    try:
        proba = model.predict_proba(df)[:, 1]
        df["fraud_proba"] = proba
        df["fraud_pred"] = (proba > 0.5).astype(int)

        st.subheader("🔍 Predictions")
        st.dataframe(df.head())

        st.download_button(
            "⬇️ Download Predictions",
            df.to_csv(index=False),
            "fraud_predictions.csv",
            "text/csv",
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
