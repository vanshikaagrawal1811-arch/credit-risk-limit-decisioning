"""
test_decision_engine.py
Tests for Model 2's feature/target setup (no leakage) and the decision
engine's breakeven logic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utilization_pipeline import get_utilization_dataset, UTILIZATION_FEATURE_COLUMNS
from decision_engine import (
    breakeven_utilization,
    expected_profit,
    recommend_limit,
    compare_naive_vs_model_utilization,
)
from sensitivity_analysis import CLIENT


def test_utilization_features_exclude_leakage_columns():
    # avg_utilization is defined as BILL_AMT / LIMIT_BAL, so none of those
    # (or ratios built from them) should be in Model 2's feature set.
    leaky = {"LIMIT_BAL", "credit_limit_log", "avg_utilization", "max_utilization",
             "pay_to_bill_ratio", *(f"BILL_AMT{i}" for i in range(1, 7)),
             *(f"PAY_AMT{i}" for i in range(1, 7))}
    assert leaky.isdisjoint(set(UTILIZATION_FEATURE_COLUMNS))


def test_utilization_dataset_shapes_match():
    X, y, df = get_utilization_dataset()
    assert len(X) == len(y) == len(df)
    assert list(X.columns) == UTILIZATION_FEATURE_COLUMNS


def test_breakeven_increases_with_pd():
    # Riskier clients need higher utilization to justify more credit.
    low_pd_breakeven = breakeven_utilization(0.05)
    high_pd_breakeven = breakeven_utilization(0.20)
    assert high_pd_breakeven > low_pd_breakeven


def test_recommend_limit_approves_max_when_util_above_breakeven():
    breakeven = breakeven_utilization(0.05)
    limit, profit, be = recommend_limit(current_limit=100000, pd=0.05, expected_util=breakeven + 0.2)
    assert limit == 150000  # max_multiple default = 1.5
    assert profit > 0


def test_recommend_limit_floors_when_util_below_breakeven():
    breakeven = breakeven_utilization(0.10)
    limit, profit, be = recommend_limit(current_limit=100000, pd=0.10, expected_util=max(breakeven - 0.2, 0))
    assert limit == 50000  # min_multiple default = 0.5


def test_recommend_limit_always_floors_above_decline_pd():
    # Even with sky-high utilization, a near-certain defaulter gets the floor.
    limit, profit, be = recommend_limit(current_limit=100000, pd=0.9, expected_util=3.0)
    assert limit == 50000


def test_naive_vs_model_can_flip_decision():
    # Regression test matching the PD=0.10 example in decision_engine.py's
    # own demo: naive 0.5 assumption approves max, a lower model prediction
    # flips it to the floor.
    result = compare_naive_vs_model_utilization(current_limit=100000, pd=0.10, model_util=0.30)
    assert result["decision_flipped"] is True
    assert result["model_recommended_limit"] < result["naive_recommended_limit"]


def test_naive_vs_model_same_side_does_not_flip():
    result = compare_naive_vs_model_utilization(current_limit=100000, pd=0.05, model_util=0.65)
    assert result["decision_flipped"] is False
    assert result["naive_recommended_limit"] == result["model_recommended_limit"]


def test_sensitivity_example_client_crosses_breakeven_across_pd_range():
    # Locks in the specific finding reported in the README/sensitivity
    # output: the fixed example client (45% utilization) is above breakeven
    # at PD=0.08 and below it at PD=0.12, i.e. the decision genuinely
    # depends on the assumptions, not just the headline demo numbers.
    low_pd_limit, _, low_breakeven = recommend_limit(
        CLIENT["current_limit"], 0.08, CLIENT["expected_util"]
    )
    high_pd_limit, _, high_breakeven = recommend_limit(
        CLIENT["current_limit"], 0.12, CLIENT["expected_util"]
    )
    assert low_breakeven < CLIENT["expected_util"]
    assert high_breakeven > CLIENT["expected_util"]
    assert low_pd_limit > high_pd_limit


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
