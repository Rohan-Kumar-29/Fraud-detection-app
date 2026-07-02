# 🚨 Fraud Detection App

An end-to-end machine learning project that detects fraudulent financial
transactions and serves the predictions through an interactive **Streamlit
dashboard** built for non-technical stakeholders.

## Overview

The project covers the full ML lifecycle:

1. **Model training** (`fraud_detection_model.ipynb`) — EDA, data cleaning,
   feature engineering, model comparison, and hyperparameter tuning on the
   PaySim mobile-money dataset.
2. **Deployment** (`app.py`) — a Streamlit dashboard that scores uploaded
   transactions and presents the results clearly.
3. **Packaging** (`Dockerfile`, `requirements.txt`) — containerized for easy
   deployment.

## The Model

- Compared **Logistic Regression, Random Forest, Gradient Boosting, and
  XGBoost** inside a scikit-learn `Pipeline` (median imputation +
  one-hot encoding).
- **Random Forest** was selected as the best model (chosen by **PR-AUC**,
  the right metric for a highly imbalanced dataset where fraud is ~0.13%).
- Engineered features include balance residuals (`orig_residual`,
  `dest_residual`), `log_amount`, and a merchant-account flag.
- Handles class imbalance with `class_weight="balanced"`.
- Final model saved as `fraud_model1.pkl`.

## The Dashboard

Upload a CSV of transactions and the app returns:

- **Summary KPIs** — transactions analysed, number flagged, fraud rate, and
  total amount at risk.
- **Risk breakdown chart** — Low / Medium / High risk counts.
- **Plain-language verdicts** — "⚠️ Likely Fraud" / "✅ Looks Legit".
- **Highlighted table** of flagged transactions, most suspicious first.
- **Adjustable decision threshold** to trade off recall vs. false alarms.
- **Downloadable results** as CSV.

## Getting Started

### 1. Set up the environment

```bash
python -m venv fraud-venv
# Windows
fraud-venv\Scripts\activate
# macOS/Linux
source fraud-venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### 3. Try it out

Upload the included **`sample_transactions.csv`** to see the dashboard in
action — no external data needed.

> **Note:** the uploaded CSV must contain the same feature columns the model
> was trained on (see `sample_transactions.csv` for the exact format).

## Run with Docker

```bash
docker build -t fraud-detection .
docker run -p 10000:10000 fraud-detection
```

## Project Structure

```
.
├── app.py                       # Streamlit dashboard
├── fraud_detection_model.ipynb  # Model training & analysis
├── fraud_model1.pkl             # Trained model
├── sample_transactions.csv      # Sample input for a quick demo
├── requirements.txt             # Pinned dependencies
├── Dockerfile                   # Container setup
└── README.md
```

## Tech Stack

Python · scikit-learn · XGBoost · pandas · Streamlit · Docker

> **Reproducibility note:** the saved model was trained with
> `scikit-learn==1.5.1` and should be loaded with that version (as pinned in
> `requirements.txt`) to avoid unpickling errors on newer releases.
