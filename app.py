"""
app.py
Streamlit dashboard for the Credit Card Default Risk & Limit Decisioning
system. Two ML models feed a non-ML decision layer:

  Model 1 (classifier)  -> probability of default (PD)
  Model 2 (regressor)   -> predicted future credit utilization
  Decision engine       -> recommended credit limit, from expected profit

Inputs live in a sidebar form; results fill the main area. Quick-start
presets let a visitor see a meaningful result in one click.
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
from data_pipeline import engineer_features
from utilization_pipeline import UTILIZATION_FEATURE_COLUMNS
from decision_engine import compare_naive_vs_model_utilization

st.set_page_config(page_title="Credit Risk & Limit Decisioning", layout="wide")

# --- Styling: restrained palette matching .streamlit/config.toml ---
# Ink #10223D, Slate #475569, Accent teal #0F766E, Border #D8DEE6.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }
    h1 { font-weight: 650; letter-spacing: -0.01em; color: #10223D; font-size: 2rem; }
    h2, h3 { font-weight: 600; color: #10223D; }
    [data-testid="stSidebar"] { min-width: 24rem; max-width: 24rem; }
    [data-testid="stSidebar"] h3 { font-size: 1.05rem; margin-top: 0.4rem; }
    [data-testid="stMetricValue"] { color: #10223D; font-weight: 650; }
    [data-testid="stMetricLabel"] { color: #475569; }
    .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
        background-color: #0F766E; border-color: #0F766E;
    }
    .stButton button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover {
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
    # calibrated_model.pkl drives the risk score, decision threshold, and the
    # decision engine's PD input (calibration matters because PD is used as a
    # literal probability in the expected-profit formula - see README).
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

# --------------------------------------------------------------------------
# Input definitions
# --------------------------------------------------------------------------
SEX_LABELS = {1: "Male", 2: "Female"}
EDUCATION_LABELS = {1: "Graduate school", 2: "University", 3: "High school", 4: "Other"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}

# Dataset codes for monthly repayment status. Listed in the order a person
# would think about them: on time first, then increasing lateness.
PAY_OPTIONS = [0, -1, -2, 1, 2, 3, 4, 5, 6, 7, 8]
PAY_LABELS = {0: "Paid on time", -1: "Paid in full", -2: "No spending"}
PAY_LABELS.update({n: f"{n} month{'s' if n > 1 else ''} late" for n in range(1, 9)})

PAY_FIELDS = [  # (session key, label)
    ("pay_0", "Latest month"),
    ("pay_2", "1 month ago"),
    ("pay_3", "2 months ago"),
    ("pay_4", "3 months ago"),
    ("pay_5", "4 months ago"),
    ("pay_6", "5 months ago"),
]

DEFAULTS = {
    "limit_bal": 150000, "age": 35, "sex": 1, "education": 2, "marriage": 2,
    "pay_0": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
    "avg_bill": 40000, "avg_pay": 5000,
}

PRESETS = {
    "Reliable payer": {
        "limit_bal": 200000, "age": 42, "sex": 2, "education": 2, "marriage": 1,
        "pay_0": -1, "pay_2": -1, "pay_3": 0, "pay_4": -1, "pay_5": 0, "pay_6": 0,
        "avg_bill": 20000, "avg_pay": 20000,
    },
    "Typical client": dict(DEFAULTS),
    "Struggling payer": {
        "limit_bal": 100000, "age": 29, "sex": 1, "education": 3, "marriage": 2,
        "pay_0": 2, "pay_2": 2, "pay_3": 2, "pay_4": 3, "pay_5": 2, "pay_6": 2,
        "avg_bill": 90000, "avg_pay": 1000,
    },
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Sidebar: inputs
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Client profile")
    st.caption("Start from an example, then adjust anything.")
    preset_cols = st.columns(3)
    for col, name in zip(preset_cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

    with st.form("client_form"):
        st.markdown("**About the client**")
        st.number_input("Credit limit", min_value=10000, max_value=1000000, step=10000, key="limit_bal")
        age_col, sex_col = st.columns(2)
        age_col.number_input("Age", min_value=21, max_value=79, step=1, key="age")
        sex_col.selectbox("Sex", options=list(SEX_LABELS), format_func=SEX_LABELS.get, key="sex")
        edu_col, mar_col = st.columns(2)
        edu_col.selectbox("Education", options=list(EDUCATION_LABELS),
                          format_func=EDUCATION_LABELS.get, key="education")
        mar_col.selectbox("Marital status", options=list(MARRIAGE_LABELS),
                          format_func=MARRIAGE_LABELS.get, key="marriage")

        st.markdown("**Repayment history**")
        st.caption("How the client paid each of the last 6 months.")
        left, right = st.columns(2)
        for i, (key, label) in enumerate(PAY_FIELDS):
            (left if i % 2 == 0 else right).selectbox(
                label, options=PAY_OPTIONS, format_func=PAY_LABELS.get, key=key
            )

        st.markdown("**Typical monthly amounts**")
        bill_col, pay_col = st.columns(2)
        bill_col.number_input("Average bill", min_value=0, max_value=1000000, step=1000, key="avg_bill")
        pay_col.number_input("Average payment", min_value=0, max_value=1000000, step=500, key="avg_pay")

        submitted = st.form_submit_button("Run assessment", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# Run the pipeline when the form is submitted; keep results across reruns
# --------------------------------------------------------------------------
if submitted:
    s = st.session_state
    row = {
        "LIMIT_BAL": s.limit_bal, "SEX": s.sex, "EDUCATION": s.education, "MARRIAGE": s.marriage, "AGE": s.age,
        "PAY_0": s.pay_0, "PAY_2": s.pay_2, "PAY_3": s.pay_3, "PAY_4": s.pay_4, "PAY_5": s.pay_5, "PAY_6": s.pay_6,
    }
    for i in range(1, 7):
        row[f"BILL_AMT{i}"] = s.avg_bill
        row[f"PAY_AMT{i}"] = s.avg_pay

    engineered_df = engineer_features(pd.DataFrame([row]))

    # Model 1: default probability (calibrated model)
    X_input = engineered_df[FEATURES]
    X_model = pd.DataFrame(scaler.transform(X_input), columns=FEATURES) if metadata["uses_scaled_input"] else X_input
    pd_score = float(calibrated_model.predict_proba(X_model)[0, 1])

    # Model 2: predicted utilization (demographics + repayment behavior only)
    X_util = engineered_df[UTILIZATION_FEATURE_COLUMNS]
    predicted_util = float(np.clip(util_model.predict(X_util)[0], 0, 3))

    # SHAP explanation from the raw model
    shap_values = explainer(X_input)
    # RandomForest returns (n_samples, n_features, n_classes); XGBoost binary
    # returns (n_samples, n_features). Slice to the "default" class only when
    # the extra axis is present.
    explanation = shap_values[0]
    if explanation.values.ndim == 2:
        explanation = explanation[:, 1]
    fig = plt.figure(figsize=(9, 4.5))
    shap.plots.waterfall(explanation, max_display=8, show=False)
    plt.tight_layout()

    comparison = compare_naive_vs_model_utilization(
        current_limit=s.limit_bal, pd=pd_score, model_util=predicted_util
    )

    st.session_state["result"] = {
        "pd_score": pd_score,
        "predicted_util": predicted_util,
        "comparison": comparison,
        "limit_bal": s.limit_bal,
        "fig": fig,
    }

# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("Credit Risk & Limit Decisioning")
st.caption(
    f"Default-risk model: {metadata['model_name']}, ROC-AUC {metadata['test_auc']:.3f}   |   "
    f"Utilization model: {util_metadata['model_name']}, R\u00b2 {util_metadata['test_r2']:.3f}"
)

result = st.session_state.get("result")

if result is None:
    st.markdown(
        "Describe a client in the sidebar and select **Run assessment**. "
        "You will get a default-risk score, the reasons behind it, and a "
        "recommended credit limit."
    )
    st.info("No client assessed yet. Pick an example profile in the sidebar to see a result in one click.")
    st.markdown("#### How it works")
    st.markdown(
        "- **Default-risk model** estimates the chance this client defaults.\n"
        "- **Utilization model** estimates how much of their limit they will use.\n"
        "- **Decision rule** (not ML) combines both into an expected-profit limit recommendation."
    )
else:
    pd_score = result["pd_score"]
    predicted_util = result["predicted_util"]
    comparison = result["comparison"]
    current_limit = result["limit_bal"]
    recommended = comparison["model_recommended_limit"]
    elevated = pd_score >= THRESHOLD

    # --- Verdict first, in plain language ---
    if elevated:
        st.error("**Elevated risk.** This client is likely to default.")
    else:
        st.success("**Low risk.** This client is likely to repay.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Chance of default", f"{pd_score:.1%}", help=f"Flagged as elevated above {THRESHOLD:.1%} (cost-tuned threshold, 5:1 penalty for missed defaulters).")
    m2.metric("Expected credit use", f"{predicted_util:.1%}", help="Predicted from demographics and repayment behavior only, not bill or payment amounts, so it does not simply re-derive utilization from its own definition.")
    m3.metric("Recommended limit", f"{recommended:,.0f}", delta=f"{recommended - current_limit:+,.0f} vs current", delta_color="off")

    tab_why, tab_limit = st.tabs(["Why this score", "How the limit was chosen"])

    with tab_why:
        st.caption("Each bar shows how much a factor pushed this client's default probability up (red) or down (blue).")
        st.pyplot(result["fig"], use_container_width=True)

    with tab_limit:
        st.caption(
            "The limit is not predicted by a model. It comes from an expected-profit rule "
            "that takes the default probability and expected credit use as inputs "
            "(see src/decision_engine.py)."
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**If we assumed flat 50% credit use**")
            st.metric("Recommended limit", f"{comparison['naive_recommended_limit']:,.0f}")
            st.caption(f"Expected annual profit: {comparison['naive_expected_profit']:,.0f}")
        with c2:
            st.markdown("**Using the utilization model's prediction**")
            st.metric("Recommended limit", f"{comparison['model_recommended_limit']:,.0f}")
            st.caption(f"Expected annual profit: {comparison['model_expected_profit']:,.0f}")

        if comparison["decision_flipped"]:
            st.warning(
                f"The utilization model changed the decision for this client. The breakeven "
                f"utilization is {comparison['breakeven_utilization']:.1%}; the flat 50% "
                f"assumption and the predicted {predicted_util:.1%} fall on opposite sides "
                f"of it, moving the recommended limit from "
                f"{comparison['naive_recommended_limit']:,.0f} to "
                f"{comparison['model_recommended_limit']:,.0f}."
            )
        else:
            st.info(
                f"The breakeven utilization for this client is {comparison['breakeven_utilization']:.1%}. "
                f"The flat assumption and the model's prediction land on the same side of it, "
                f"so the recommended limit is unchanged. Only the expected profit estimate differs."
            )

st.divider()
st.caption(
    "Demonstration project, not a production credit decisioning system. Amounts are in the "
    "dataset's own currency units (Taiwan, 2005). Interest rate, loss-given-default, and "
    "cost-ratio assumptions are illustrative; see the README for details and a sensitivity analysis."
)
