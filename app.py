import streamlit as st
import pandas as pd
import cloudpickle

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

MODEL_PATH = "fraud_model1.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return cloudpickle.load(f)


try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

st.title("🚨 Fraud Detection Dashboard")
st.caption("Upload transaction data to automatically flag suspicious activity.")

# --- Controls ---
with st.sidebar:
    st.header("Settings")
    threshold = st.slider(
        "Fraud decision threshold",
        0.0,
        1.0,
        0.5,
        0.01,
        help="Transactions scoring above this are flagged as fraud. "
        "Lower it to catch more fraud (more false alarms); "
        "raise it to be stricter.",
    )

uploaded = st.file_uploader("Upload a CSV file with transactions", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)

    try:
        proba = model.predict_proba(df)[:, 1]
        df["fraud_proba"] = proba
        df["fraud_pred"] = (proba > threshold).astype(int)

        # Human-readable risk labels
        def risk_band(p):
            if p >= 0.80:
                return "High"
            elif p >= 0.40:
                return "Medium"
            return "Low"

        df["risk_level"] = df["fraud_proba"].apply(risk_band)
        df["verdict"] = df["fraud_pred"].map(
            {1: "⚠️ Likely Fraud", 0: "✅ Looks Legit"}
        )

        # --- KPI summary for non-technical stakeholders ---
        n_total = len(df)
        n_flagged = int(df["fraud_pred"].sum())
        fraud_rate = 100 * n_flagged / n_total if n_total else 0
        amount_at_risk = (
            df.loc[df["fraud_pred"] == 1, "amount"].sum()
            if "amount" in df.columns
            else 0
        )

        st.subheader("📊 Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions analysed", f"{n_total:,}")
        c2.metric("Flagged as fraud", f"{n_flagged:,}")
        c3.metric("Fraud rate", f"{fraud_rate:.2f}%")
        c4.metric("Amount at risk", f"{amount_at_risk:,.0f}")

        # --- Risk breakdown chart ---
        st.subheader("🎯 Risk Breakdown")
        band_counts = (
            df["risk_level"]
            .value_counts()
            .reindex(["Low", "Medium", "High"])
            .fillna(0)
        )
        st.bar_chart(band_counts)

        # --- Flagged transactions, most suspicious first ---
        st.subheader("🔍 Flagged Transactions (highest risk first)")
        flagged = df[df["fraud_pred"] == 1].sort_values(
            "fraud_proba", ascending=False
        )

        if flagged.empty:
            st.info("No transactions exceeded the fraud threshold.")
        else:
            # Shade rows red by risk (no matplotlib dependency).
            def shade_risk(row):
                p = row["fraud_proba"]
                color = (
                    "background-color: #f8d7da"
                    if p >= 0.80
                    else "background-color: #fff3cd"
                    if p >= 0.40
                    else ""
                )
                return [color] * len(row)

            styled = flagged.style.apply(shade_risk, axis=1).format(
                {"fraud_proba": "{:.1%}"}
            )
            st.dataframe(styled, use_container_width=True)

        st.download_button(
            "⬇️ Download Full Results",
            df.to_csv(index=False),
            "fraud_predictions.csv",
            "text/csv",
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
