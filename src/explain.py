"""
explain.py
SHAP-based explainability for the trained XGBoost model - both global
feature importance and per-client explanations, as used in the Streamlit app.
"""

import json
import joblib
import shap
import pandas as pd


def load_artifacts(model_dir: str = "models"):
    model = joblib.load(f"{model_dir}/best_model.pkl")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    with open(f"{model_dir}/metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def get_explainer(model, metadata):
    # TreeExplainer is exact and fast for tree-based models (XGBoost/RandomForest).
    # Falls back to a general Explainer for Logistic Regression if that ever wins.
    if metadata["model_name"] in ("XGBoost", "RandomForest"):
        return shap.TreeExplainer(model)
    return shap.Explainer(model)


def explain_client(model, explainer, X_row: pd.DataFrame, feature_columns):
    """
    Returns (shap_values, base_value) for a single client row, plus the
    top contributing features sorted by |impact|, for display in the UI.

    Note: some tree models (RandomForest) return per-class SHAP values with
    an extra trailing axis (n_samples, n_features, n_classes); others
    (XGBoost's binary objective) return (n_samples, n_features) directly.
    This slices to the "default" class (index 1) only when that extra axis
    is present, so the result is consistent regardless of which model won.
    """
    shap_values = explainer(X_row[feature_columns])

    values = shap_values.values[0]
    if values.ndim == 2:
        values = values[:, 1]
    contributions = sorted(
        zip(feature_columns, values, X_row[feature_columns].values[0]),
        key=lambda t: abs(t[1]),
        reverse=True,
    )
    return shap_values, contributions


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_pipeline import get_dataset

    model, scaler, metadata = load_artifacts()
    X, y, df = get_dataset(engineered=True)
    feature_columns = metadata["feature_columns"]

    explainer = get_explainer(model, metadata)

    sample = X.iloc[[0]]
    shap_values, contributions = explain_client(model, explainer, sample, feature_columns)

    print("Top 5 contributing features for client 0:")
    for name, val, raw in contributions[:5]:
        direction = "increases" if val > 0 else "decreases"
        print(f"  {name:25s} (value={raw:>10.2f})  {direction} risk by {abs(val):.4f}")
