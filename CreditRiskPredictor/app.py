"""
app.py
Streamlit dashboard for the Credit Card Default Risk & Limit Decisioning
system. Two ML models feed a non-ML decision layer:

  Model 1 (classifier)  -> probability of default (PD)
  Model 2 (regressor)   -> predicted future credit utilization
  Decision engine        -> recommended credit limit, from expected profit

Lets a user input a client's profile, see the default-risk score with a
SHAP explanation, and see the resulting limit recommendation - including a
side-by-side comparison against a naive utilization assumption, to show
that Model 2 actually changes the outcome rather than existing for show.
"""

import sys
import json
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, "src")
from data_pipeline import engineer_features, FEATURE_COLUMNS_ENGINEERED, PAY_COLS
from utilization_pipeline import UTILIZATION_FEATURE_COLUMNS
from decision_engine import recommend_limit, compare_naive_vs_model_utilization, breakeven_utilization

st.set_page_config(page_title="Credit Risk & Limit Decisioning", layout="wide")

# --- Styling: restrained fintech palette matching .streamlit/config.toml ---
# Ink #10223D, Slate #475569, Accent teal #0F766E, Border #D8DEE6.
# Kept deliberately flat (no shadows, no gradient washes, one border-radius
# used sparingly) rather than the generic "SaaS card kit" look.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; max-width: 1100px; }
    h1 { font-weight: 650; letter-spacing: -0.01em; color: #10223D; }
    h2 { font-weight: 600; color: #10223D; border-bottom: 1px solid #D8DEE6; padding-bottom: 0.4rem; }
    h3 { font-weight: 600; color: #10223D; }
    [data-testid="stMetricValue"] { color: #10223D; font-weight: 650; }
    [data-testid="stMetricLabel"] { color: #475569; }
    .stButton button[kind="primary"] {
        background-color: #0F766E; border-color: #0F766E;
    }
    .stButton button[kind="primary"]:hover {
        background-color: #0C5F58; border-color: #0C5F58;
    }
    [data-testid="stCaptionContainer"] { color: #475569; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    # best_model.pkl (raw) is used only for SHAP - CalibratedClassifierCV
    # wraps the model in a way TreeExplainer can't see inside.
    raw_model = joblib.load("models/best_model.pkl")
    # calibrated_model.pkl is used for the actual risk score, the decision
    # threshold, and the decision engine's PD input - see README for why
    # raw probabilities from this model type weren't reliable enough to use
    # directly in an expected-profit calculation.
    calibrated_model = joblib.load("models/calibrated_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    with open("models/metadata.json") as f:
        metadata = json.load(f)
    explainer = shap.TreeExplainer(raw_model) if metadata["model_name"] in ("XGBoost", "RandomForest") \
        else shap.Explainer(raw_model)

    util_model = joblib.load("models/utilization_model.pkl")
    with open("models/utilization_metadata.json") as f:
        util_metadata = json.load(f)

    return raw_model, calibrated_model, scaler, metadata, explainer, util_model, util_metadata


raw_model, calibrated_model, scaler, metadata, explainer, util_model, util_metadata = load_artifacts()
FEATURES = metadata["feature_columns"]
THRESHOLD = metadata["cost_based_threshold"]

st.title("Credit Risk & Limit Decisioning")
st.caption(
    f"Model 1 (default risk): {metadata['model_name']}, test ROC-AUC {metadata['test_auc']:.3f}  |  "
    f"Model 2 (utilization): {util_metadata['model_name']}, test R\u00b2 {util_metadata['test_r2']:.3f}"
)

st.markdown(
    "Enter a client's profile to see their default-risk score, an explanation "
    "of that score, and a recommended credit limit derived from both models' "
    "outputs by a separate decision rule."
)

st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Credit & Demographics")
    limit_bal = st.number_input("Current credit limit (NT$)", min_value=10000, max_value=1000000, value=150000, step=10000)
    age = st.slider("Age", 21, 79, 35)
    sex = st.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox("Education", options=[1, 2, 3, 4],
                              format_func=lambda x: {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}[x])
    marriage = st.selectbox("Marital status", options=[1, 2, 3],
                             format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x])

with col2:
    st.subheader("Repayment History (last 6 months)")
    st.caption("-1/0 = paid on time, 1+ = months delayed")
    pay_0 = st.slider("Most recent month (PAY_0)", -2, 8, 0)
    pay_2 = st.slider("1 month ago (PAY_2)", -2, 8, 0)
    pay_3 = st.slider("2 months ago (PAY_3)", -2, 8, 0)
    pay_4 = st.slider("3 months ago (PAY_4)", -2, 8, 0)
    pay_5 = st.slider("4 months ago (PAY_5)", -2, 8, 0)
    pay_6 = st.slider("5 months ago (PAY_6)", -2, 8, 0)

with col3:
    st.subheader("Bills & Payments (avg, last 6 months)")
    avg_bill = st.number_input("Average monthly bill (NT$)", min_value=0, max_value=1000000, value=30000, step=1000)
    avg_pay_amt = st.number_input("Average monthly payment made (NT$)", min_value=0, max_value=1000000, value=5000, step=500)

if st.button("Run Full Assessment", type="primary"):
    row = {
        "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
        "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
    }
    for i in range(1, 7):
        row[f"BILL_AMT{i}"] = avg_bill
        row[f"PAY_AMT{i}"] = avg_pay_amt

    raw_df = pd.DataFrame([row])
    engineered_df = engineer_features(raw_df)

    # --- Model 1: default probability (calibrated model - see load_artifacts) ---
    X_input = engineered_df[FEATURES]
    X_model = pd.DataFrame(scaler.transform(X_input), columns=FEATURES) if metadata["uses_scaled_input"] else X_input
    pd_score = float(calibrated_model.predict_proba(X_model)[0, 1])
    risk_label = "Elevated risk — likely to default" if pd_score >= THRESHOLD else "Low risk — likely to repay"

    # --- Model 2: predicted utilization (demographics + repayment behavior only) ---
    X_util = engineered_df[UTILIZATION_FEATURE_COLUMNS]
    predicted_util = float(np.clip(util_model.predict(X_util)[0], 0, 3))

    st.divider()
    st.header("1. Default Risk (Model 1)")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric("Default Probability", f"{pd_score:.1%}")
        if pd_score >= THRESHOLD:
            st.error(risk_label)
        else:
            st.success(risk_label)
        st.caption(f"Decision threshold: {THRESHOLD:.1%} (cost-tuned, 5:1 FN penalty)")

    with res_col2:
        st.subheader("What drove this score")
        shap_values = explainer(X_input)
        # Some tree models (RandomForest) return per-class SHAP values with
        # shape (n_samples, n_features, n_classes); others (XGBoost's binary
        # objective) return (n_samples, n_features) directly. Slice to the
        # "default" class (index 1) only when that extra axis is present, so
        # the waterfall plot works regardless of which model won training.
        explanation = shap_values[0]
        if explanation.values.ndim == 2:
            explanation = explanation[:, 1]
        fig = plt.figure(figsize=(9, 4.5))
        shap.plots.waterfall(explanation, max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.header("2. Predicted Utilization (Model 2)")
    st.caption(
        "Predicted from demographics + repayment behavior only (not bill/payment "
        "amounts), so it doesn't just re-derive utilization from its own definition."
    )
    st.metric("Predicted Utilization", f"{predicted_util:.1%}")

    st.divider()
    st.header("3. Recommended Credit Limit (Decision Engine — not ML)")
    st.caption(
        "Not a prediction: this is an expected-profit rule that takes PD and "
        "predicted utilization as inputs. See src/decision_engine.py for the formula."
    )

    comparison = compare_naive_vs_model_utilization(
        current_limit=limit_bal, pd=pd_score, model_util=predicted_util
    )

    dec_col1, dec_col2 = st.columns(2)
    with dec_col1:
        st.subheader("Naive assumption (flat 50% utilization)")
        st.metric("Recommended Limit", f"NT$ {comparison['naive_recommended_limit']:,.0f}")
        st.caption(f"Expected annual profit: NT$ {comparison['naive_expected_profit']:,.0f}")

    with dec_col2:
        st.subheader("Using Model 2's prediction")
        st.metric("Recommended Limit", f"NT$ {comparison['model_recommended_limit']:,.0f}")
        st.caption(f"Expected annual profit: NT$ {comparison['model_expected_profit']:,.0f}")

    if comparison["decision_flipped"]:
        st.warning(
            f"Model 2 changed the decision for this client. Breakeven utilization "
            f"is {comparison['breakeven_utilization']:.1%} — the naive 50% assumption "
            f"and the model's {predicted_util:.1%} prediction fall on opposite sides "
            f"of it, moving the recommended limit from "
            f"NT$ {comparison['naive_recommended_limit']:,.0f} to "
            f"NT$ {comparison['model_recommended_limit']:,.0f}."
        )
    else:
        st.info(
            f"This client's breakeven utilization is {comparison['breakeven_utilization']:.1%}. "
            f"The naive assumption and Model 2's prediction land on the same side "
            f"of it here, so the recommended limit is unchanged — only the expected "
            f"profit estimate differs."
        )

    st.divider()
    st.caption(
        "Demonstration project, not a production credit decisioning system. "
        "Interest rate, loss-given-default, and cost-ratio assumptions are "
        "illustrative — see README for details and a sensitivity analysis."
    )

