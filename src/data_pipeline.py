"""
data_pipeline.py
Loads the UCI Default of Credit Card Clients dataset and engineers features
that go beyond the raw columns: repayment consistency and credit utilization.

Raw dataset: 30,000 clients, 23 explanatory variables + binary target
(default payment next month). Source: Yeh & Lien (2009), UCI ML Repository.
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/UCI_Credit_Card.csv"

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"default.payment.next.month": "default"})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features on top of the raw columns:

    - avg_utilization: average (bill amount / credit limit) across the 6 months,
      capped at [0, 3] to tame extreme/negative-balance outliers. Utilization is
      a well-known strong predictor of default risk in real credit scoring.

    - max_utilization: worst single-month utilization, since a single spike can
      matter more than the average.

    - repayment_consistency: fraction of the 6 months where the client paid on
      time or early (PAY_x <= 0). Higher = more consistent payer.

    - months_delayed: count of months with any payment delay (PAY_x > 0).

    - avg_pay_delay: average delay severity across months (PAY_x values,
      clipped at 0 for on-time months) - captures *how late*, not just *how often*.

    - pay_to_bill_ratio: average (amount paid / bill amount) across months,
      capped at [0, 5]. Low ratio -> paying much less than owed -> risk signal.

    - credit_limit_log: log-transformed LIMIT_BAL, since raw credit limits are
      heavily right-skewed.
    """
    df = df.copy()

    limit = df["LIMIT_BAL"].replace(0, np.nan)

    # --- Utilization features ---
    util_matrix = df[BILL_COLS].div(limit, axis=0)
    util_matrix = util_matrix.clip(lower=0, upper=3)
    df["avg_utilization"] = util_matrix.mean(axis=1)
    df["max_utilization"] = util_matrix.max(axis=1)

    # --- Repayment consistency features ---
    pay_matrix = df[PAY_COLS]
    df["repayment_consistency"] = (pay_matrix <= 0).sum(axis=1) / len(PAY_COLS)
    df["months_delayed"] = (pay_matrix > 0).sum(axis=1)
    df["avg_pay_delay"] = pay_matrix.clip(lower=0).mean(axis=1)

    # --- Payment-to-bill ratio ---
    bill_safe = df[BILL_COLS].replace(0, np.nan).values
    pay_amt = df[PAY_AMT_COLS].values
    ratio = np.divide(pay_amt, bill_safe, out=np.zeros_like(pay_amt, dtype=float), where=bill_safe != 0)
    ratio = np.clip(ratio, 0, 5)
    df["pay_to_bill_ratio"] = ratio.mean(axis=1)

    # --- Scale transform ---
    df["credit_limit_log"] = np.log1p(df["LIMIT_BAL"])

    df["avg_utilization"] = df["avg_utilization"].fillna(0)
    df["max_utilization"] = df["max_utilization"].fillna(0)
    df["pay_to_bill_ratio"] = df["pay_to_bill_ratio"].fillna(0)

    return df


FEATURE_COLUMNS_RAW = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    *PAY_COLS, *BILL_COLS, *PAY_AMT_COLS,
]

FEATURE_COLUMNS_ENGINEERED = FEATURE_COLUMNS_RAW + [
    "avg_utilization", "max_utilization",
    "repayment_consistency", "months_delayed", "avg_pay_delay",
    "pay_to_bill_ratio", "credit_limit_log",
]


def get_dataset(engineered: bool = True):
    df = load_raw()
    if engineered:
        df = engineer_features(df)
        feature_cols = FEATURE_COLUMNS_ENGINEERED
    else:
        feature_cols = FEATURE_COLUMNS_RAW

    X = df[feature_cols]
    y = df["default"]
    return X, y, df


if __name__ == "__main__":
    X, y, df = get_dataset(engineered=True)
    print(f"Rows: {len(df)}, Features: {X.shape[1]}")
    print(f"Default rate: {y.mean():.3%}")
    print(df[["avg_utilization", "repayment_consistency", "months_delayed"]].describe())
