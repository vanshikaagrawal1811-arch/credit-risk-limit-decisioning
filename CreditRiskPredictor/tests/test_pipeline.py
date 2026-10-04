"""
test_pipeline.py
Quick sanity tests for the feature engineering and cost-based threshold logic.
Run with: pytest tests/test_pipeline.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from data_pipeline import engineer_features, load_raw
from train import find_cost_based_threshold


def test_engineered_features_present():
    df = load_raw()
    engineered = engineer_features(df.head(100))
    for col in ["avg_utilization", "repayment_consistency", "months_delayed",
                "avg_pay_delay", "pay_to_bill_ratio", "credit_limit_log"]:
        assert col in engineered.columns


def test_repayment_consistency_range():
    df = load_raw()
    engineered = engineer_features(df)
    assert engineered["repayment_consistency"].between(0, 1).all()


def test_utilization_capped():
    df = load_raw()
    engineered = engineer_features(df)
    assert engineered["avg_utilization"].between(0, 3).all()
    assert engineered["max_utilization"].between(0, 3).all()


def test_perfect_payer_has_no_delay():
    # Synthetic client who always pays on time and never carries a balance
    row = {
        "LIMIT_BAL": 100000, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 1, "AGE": 30,
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    }
    for i in range(1, 7):
        row[f"BILL_AMT{i}"] = 0
        row[f"PAY_AMT{i}"] = 0
    df = pd.DataFrame([row])
    engineered = engineer_features(df)
    assert engineered["repayment_consistency"].iloc[0] == 1.0
    assert engineered["months_delayed"].iloc[0] == 0


def test_cost_based_threshold_penalizes_false_negatives():
    # 10 true positives, 10 true negatives, model scores them perfectly
    y_true = np.array([1] * 10 + [0] * 10)
    y_proba = np.array([0.9] * 10 + [0.1] * 10)
    threshold, cost = find_cost_based_threshold(y_true, y_proba, cost_fn=5.0, cost_fp=1.0)
    # With a perfect separation, any threshold between 0.1 and 0.9 gives zero cost
    assert 0.1 < threshold < 0.9
    assert cost == 0


def test_cost_based_threshold_shifts_with_cost_ratio():
    # Imperfect scores where the threshold choice actually matters
    np.random.seed(0)
    y_true = np.array([1] * 30 + [0] * 70)
    y_proba = np.concatenate([
        np.random.beta(5, 2, 30),  # positives skew high
        np.random.beta(2, 5, 70),  # negatives skew low
    ])
    low_fn_cost, _ = find_cost_based_threshold(y_true, y_proba, cost_fn=1.0, cost_fp=1.0)
    high_fn_cost, _ = find_cost_based_threshold(y_true, y_proba, cost_fn=10.0, cost_fp=1.0)
    # Penalizing false negatives more heavily should push the threshold DOWN
    # (catch more positives, tolerate more false alarms)
    assert high_fn_cost <= low_fn_cost


def test_calibration_improves_brier_score():
    # Regression test locking in the calibration fix: the calibrated model's
    # test Brier score must not be worse than the raw model's. Reads
    # models/metadata.json rather than retraining, so this only runs
    # meaningfully after `python3 src/train.py` has been run at least once.
    import json
    import os
    meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "metadata.json")
    if not os.path.exists(meta_path):
        import pytest
        pytest.skip("models/metadata.json not found - run src/train.py first")
    with open(meta_path) as f:
        metadata = json.load(f)
    assert metadata["calibrated_brier_score_test"] <= metadata["raw_brier_score_test"]


def test_threshold_not_tuned_on_test_set():
    # Guards against reintroducing the leakage this project used to have:
    # the threshold must be documented as tuned on validation, not test.
    import json
    import os
    meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "metadata.json")
    if not os.path.exists(meta_path):
        import pytest
        pytest.skip("models/metadata.json not found - run src/train.py first")
    with open(meta_path) as f:
        metadata = json.load(f)
    assert "validation" in metadata["threshold_tuned_on"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
