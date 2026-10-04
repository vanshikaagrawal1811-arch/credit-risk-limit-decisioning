"""
train.py
Compares Logistic Regression, Random Forest, and XGBoost for predicting
credit card default, under class imbalance (~22% positive rate).
Selects the best model via 5-fold cross-validated ROC-AUC, tunes the
cost-based decision threshold on a held-out VALIDATION split, then reports
final performance on a separate, untouched TEST split.

Why three splits (60/20/20) instead of two: tuning the threshold on the
same data you report final metrics on leaks information from that data
into the "final" numbers, quietly inflating them. Val and test are kept
strictly separate so the reported recall/precision reflect what the
threshold would actually do on data it never influenced.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from xgboost import XGBClassifier

from data_pipeline import get_dataset

RANDOM_STATE = 42


def build_models(scale_pos_weight: float):
    """
    Returns the three candidate models. Each handles class imbalance
    differently:
      - LogisticRegression: class_weight='balanced' reweights the loss
      - RandomForest: class_weight='balanced' does the same via bootstrap
        sample reweighting
      - XGBoost: scale_pos_weight upweights the minority (default) class
        directly in the gradient boosting objective
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def find_cost_based_threshold(y_true, y_proba, cost_fn: float = 5.0, cost_fp: float = 1.0):
    """
    Sweeps candidate thresholds and picks the one minimizing total expected
    cost, where a false negative (approving a client who defaults) is
    weighted `cost_fn` times worse than a false positive (flagging a client
    who would have repaid). Default 5:1 ratio reflects that missing a real
    default is materially costlier than an unnecessary manual review.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold, best_cost = 0.5, float("inf")

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = cost_fn * fn + cost_fp * fp
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = t

    return best_threshold, best_cost


def main():
    X, y, df = get_dataset(engineered=True)

    # Split 1: carve off the test set (20%), never touched again until the
    # very final evaluation below.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # Split 2: carve the remaining 80% into train (60% of the total) and
    # validation (20% of the total) - validation is used ONLY for threshold
    # tuning below, never for fitting or for the final reported metrics.
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(scale_pos_weight)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    print("=" * 60)
    print("5-FOLD CROSS-VALIDATED ROC-AUC (on training data)")
    print("=" * 60)
    for name, model in models.items():
        # Tree models handle raw scale fine; LR benefits from scaled features
        X_cv = X_train_scaled if name == "LogisticRegression" else X_train
        scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {"cv_auc_mean": scores.mean(), "cv_auc_std": scores.std()}
        print(f"{name:20s}  AUC = {scores.mean():.4f}  (+/- {scores.std():.4f})")

    best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
    print(f"\nSelected model: {best_name}")

    best_model = models[best_name]
    X_fit = X_train_scaled if best_name == "LogisticRegression" else X_train
    X_val_eval = X_val_scaled if best_name == "LogisticRegression" else X_val
    X_test_eval = X_test_scaled if best_name == "LogisticRegression" else X_test
    best_model.fit(X_fit, y_train)

    # --- Raw-probability calibration check (diagnostic only) ---
    # ROC-AUC only cares about ranking order, not whether a "70%" prediction
    # is actually right 70% of the time - and the decision engine's expected-
    # profit formula needs the latter. Checking this on validation data (not
    # test) keeps the test set clean for the final numbers below.
    y_val_proba_raw = best_model.predict_proba(X_val_eval)[:, 1]
    raw_brier = brier_score_loss(y_val, y_val_proba_raw)
    print(f"Raw model Brier score (validation): {raw_brier:.4f}")

    # --- Fix it: probability calibration on top of the already-fitted model ---
    # FrozenEstimator wraps best_model so CalibratedClassifierCV treats it as
    # already trained and fits ONLY the calibration map (an isotonic
    # regressor) on top of it, using the validation split - no refitting or
    # cloning of the base model itself. (FrozenEstimator replaces the older
    # cv="prefit" option, removed in recent scikit-learn versions.) This
    # both avoids leaking train/test into the calibration step and keeps the
    # saved artifact small: ordinary cv=5 cross-fitting would instead train
    # and store 5 separate clones of the base model internally - for a
    # 300-tree Random Forest, that's the difference between an ~8MB and an
    # ~40MB file for no accuracy benefit at this dataset size.
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method="isotonic")
    calibrated_model.fit(X_val_eval, y_val)

    # Note: checking Brier score on X_val here would be circular, since the
    # calibrator was just fit on exactly that data - it would look better
    # than it really is. The honest before/after comparison is done on the
    # untouched test set further down (raw_test_brier vs cal_test_brier).

    # --- Threshold tuning happens ONLY on validation data, using the
    # calibrated probabilities (since those are what the app and the
    # decision engine actually use) ---
    y_val_proba_cal = calibrated_model.predict_proba(X_val_eval)[:, 1]
    threshold, val_cost = find_cost_based_threshold(y_val, y_val_proba_cal)
    print(f"Cost-based threshold (tuned on validation, calibrated probabilities): {threshold:.3f} (5:1 FN:FP cost ratio)")

    # --- Final reporting happens ONLY on the untouched test set ---
    y_test_proba_raw = best_model.predict_proba(X_test_eval)[:, 1]
    y_test_proba_cal = calibrated_model.predict_proba(X_test_eval)[:, 1]

    test_auc = roc_auc_score(y_test, y_test_proba_cal)
    print(f"Held-out test ROC-AUC (calibrated probabilities): {test_auc:.4f}")

    preds = (y_test_proba_cal >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print(f"Confusion matrix @ threshold={threshold:.3f} (test set): TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Recall (catching real defaulters): {recall:.3f}")
    print(f"Precision: {precision:.3f}")

    raw_test_brier = brier_score_loss(y_test, y_test_proba_raw)
    cal_test_brier = brier_score_loss(y_test, y_test_proba_cal)
    print(f"Test Brier score - raw: {raw_test_brier:.4f}, calibrated: {cal_test_brier:.4f}")

    frac_pos, mean_pred = calibration_curve(y_test, y_test_proba_cal, n_bins=10, strategy="quantile")
    calibration_bins = [
        {"mean_predicted": float(mp), "observed_fraction": float(fp_)}
        for mp, fp_ in zip(mean_pred, frac_pos)
    ]

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(calibrated_model, "models/calibrated_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    with open("models/metadata.json", "w") as f:
        json.dump({
            "model_name": best_name,
            "cv_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            "test_auc": float(test_auc),
            "cost_based_threshold": float(threshold),
            "threshold_tuned_on": "validation_split_calibrated_probabilities",
            "calibration_method": "isotonic_on_frozen_estimator_fit_on_validation_split",
            "test_recall": float(recall),
            "test_precision": float(precision),
            "raw_brier_score_test": float(raw_test_brier),
            "calibrated_brier_score_test": float(cal_test_brier),
            "calibration_bins": calibration_bins,
            "uses_scaled_input": best_name == "LogisticRegression",
            "feature_columns": list(X.columns),
            "probability_source": (
                "models/calibrated_model.pkl is used for the reported risk "
                "score, the decision threshold, and the decision engine's PD "
                "input. models/best_model.pkl (uncalibrated) is used only "
                "for SHAP explanations, since CalibratedClassifierCV wraps "
                "the model in a way SHAP's TreeExplainer can't look inside."
            ),
        }, f, indent=2)

    print("\nSaved: models/best_model.pkl, models/calibrated_model.pkl, models/scaler.pkl, models/metadata.json")


if __name__ == "__main__":
    main()
