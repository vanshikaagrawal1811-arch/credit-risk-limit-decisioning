"""
utilization_pipeline.py
Builds the feature set and target for Model 2: predicting a client's
average credit utilization from demographics and repayment BEHAVIOR only
(not from bill/payment dollar amounts).

Why exclude BILL_AMT*, PAY_AMT*, LIMIT_BAL, and pay_to_bill_ratio:
utilization itself is defined as BILL_AMT / LIMIT_BAL, so including any of
those columns (or ratios built from them) would let the model "predict"
utilization by nearly re-deriving it from its own inputs - a leakage
shortcut that would inflate performance without learning anything real.
Instead this model answers a genuinely different, decision-relevant
question: "based on who this client is and how reliably they pay, how much
of whatever limit they're given will they likely use?" That is exactly the
input the limit decision engine needs for a NEW or re-evaluated limit,
where the bill history under a hypothetical new limit doesn't exist yet.
"""

import pandas as pd
import numpy as np

from data_pipeline import load_raw, engineer_features, PAY_COLS

# Deliberately behavior/demographic only - see module docstring.
UTILIZATION_FEATURE_COLUMNS = [
    "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *PAY_COLS,
    "repayment_consistency", "months_delayed", "avg_pay_delay",
]

UTILIZATION_TARGET = "avg_utilization"


def get_utilization_dataset():
    """
    Returns (X, y, df) for the utilization regression task.
    y = avg_utilization (already computed by engineer_features), clipped
    to [0, 3] as in the main pipeline.
    """
    df = load_raw()
    df = engineer_features(df)

    X = df[UTILIZATION_FEATURE_COLUMNS].copy()
    y = df[UTILIZATION_TARGET].copy()
    return X, y, df


if __name__ == "__main__":
    X, y, df = get_utilization_dataset()
    print(f"Rows: {len(df)}, Features: {X.shape[1]}")
    print(f"Target (avg_utilization) stats:\n{y.describe()}")
