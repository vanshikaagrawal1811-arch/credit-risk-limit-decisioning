# Credit Risk Predictor & Limit Decisioning

Two ML models feeding an explicit decision rule, built on the UCI *Default of Credit Card Clients* dataset (30,000 clients, Yeh & Lien 2009) and deployed as a Streamlit app.

**Live demo:** _add your Streamlit Community Cloud URL here_

| Component | What it does |
|---|---|
| **Model 1: default classifier** | Predicts probability of default. Compares 3 models, tunes a cost-based threshold on a validation split, and calibrates probabilities so they can be used in a profit formula, not just for ranking. |
| **Model 2: utilization regressor** | Predicts future credit utilization from demographics and repayment *behavior* only. Bill and payment amounts are excluded on purpose, since utilization is defined from them and including them would leak the target. |
| **Decision engine (rule, not ML)** | Combines both outputs into a recommended credit limit using an expected-profit rule. |

Individual predictions are explained with SHAP, and the rule's assumptions are stress-tested with a sensitivity analysis.

## Design decisions

**Why not just regress `LIMIT_BAL`?** That column is the bank's *past policy*, not a ground-truth optimal limit, so a model on it would only imitate that policy, blind spots included. Instead, ML predicts things verifiable against real outcomes (default, utilization), and an auditable formula sets the limit. Every recommendation traces back to an equation.

**Why does Model 2 exist?** The optimal limit depends on whether predicted utilization is above a PD-dependent breakeven. A flat assumption (e.g. "everyone uses 50%") can land on the wrong side of that line. The app shows the naive and Model 2 recommendations side by side and flags when they differ.

## Two bugs found and fixed

Both are covered by regression tests.

1. **Threshold leakage.** The threshold was originally tuned on the test set, inflating reported recall and precision. Fixed with a 60/20/20 train/validation/test split: the threshold is tuned on validation only, and all reported metrics come from test. (`test_threshold_not_tuned_on_test_set`)
2. **Poor calibration.** The raw model's Brier score (0.173) barely beat the base-rate baseline (0.172) and it was overconfident (clients scored ~86% defaulted ~69% of the time), despite a fine ROC-AUC. Isotonic calibration fit on the validation split (`CalibratedClassifierCV` + `FrozenEstimator`) cut the test Brier score to **0.136**. `FrozenEstimator` also keeps the saved model ~8 MB instead of ~40 MB from cross-fitting. (`test_calibration_improves_brier_score`)

SHAP's `TreeExplainer` can't read inside the calibrated wrapper, so the app uses the **calibrated** model for risk score, threshold, and decision input, and the **raw** model only for the SHAP explanation.

## Results

**Model 1: default classification** (22.1% default rate)

| Model | 5-fold CV ROC-AUC |
|---|---|
| Logistic Regression | 0.7610 |
| **Random Forest (selected)** | **0.7855** |
| XGBoost | 0.7832 |

- Held-out test ROC-AUC: **0.7777**
- Cost-based threshold: **0.170** (tuned on validation, 5:1 penalty for false negatives vs. false positives)
- At that threshold on the test set: **76.1% recall, 36.1% precision**
- Random Forest and XGBoost are effectively tied; the meaningful result is that both clearly beat Logistic Regression.

**Model 2: utilization regression** (mean utilization 0.373)

| Model | 5-fold CV R² |
|---|---|
| Linear Regression | 0.436 |
| **Gradient Boosting (selected)** | **0.555** |

- Test R² **0.565**, MAE **0.169**, RMSE **0.230**
- R² is deliberately modest: this is the harder, leakage-free version of the problem.

All numbers come from real runs of `train.py` and `train_utilization.py` on the full dataset.

## Decision engine

```
expected_profit(L)        = (1 - PD) * interest_rate * expected_util * L - PD * LGD * L
breakeven_utilization(PD) = (PD * LGD) / ((1 - PD) * interest_rate)
```

Profit is linear in the limit `L`, so only one thing matters: is predicted utilization above breakeven? If yes, extend up to a policy ceiling; if no, floor the limit. This bang-bang behavior is intentional and resembles real underwriting.

**Sensitivity analysis** (`src/sensitivity_analysis.py`) varies the three assumed constants (18% interest, 75% LGD, 5:1 cost ratio). A recommendation only flips when a change pushes breakeven across the client's predicted utilization. For the example client (PD 0.08, utilization 45%), cutting interest to 12% or raising LGD to 95% flips the decision, while moderate changes do not.

## Features

- **Model 1:** 23 raw columns plus `avg_utilization`, `max_utilization`, `repayment_consistency`, `months_delayed`, `avg_pay_delay`, `pay_to_bill_ratio`, and `credit_limit_log`. SHAP ranks `PAY_0`, `avg_pay_delay`, and `repayment_consistency` among the top predictors.
- **Model 2:** `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, the six `PAY_x` columns, `repayment_consistency`, `months_delayed`, `avg_pay_delay`.
- **Class imbalance:** `class_weight='balanced'` (LR, RF) and `scale_pos_weight` (XGBoost), avoiding oversampling that distorts calibration.

## Project structure

```
├── app.py                        # Streamlit dashboard
├── src/
│   ├── data_pipeline.py          # loading + feature engineering (Model 1)
│   ├── train.py                  # Model 1: comparison, calibration, threshold
│   ├── explain.py                # SHAP helpers
│   ├── utilization_pipeline.py   # leakage-free setup (Model 2)
│   ├── train_utilization.py      # Model 2 training
│   ├── decision_engine.py        # PD + utilization -> limit
│   └── sensitivity_analysis.py
├── tests/                        # pipeline, calibration, leakage, decision engine
├── data/UCI_Credit_Card.csv
├── models/                       # saved models + metadata (generated)
├── .github/workflows/tests.yml   # CI
└── CODE_WALKTHROUGH.md           # line-by-line explanation
```

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 src/train.py               # Model 1
PYTHONPATH=src python3 src/train_utilization.py   # Model 2
PYTHONPATH=src pytest tests/ -v                   # tests
PYTHONPATH=src python3 src/sensitivity_analysis.py
streamlit run app.py                              # http://localhost:8501
```

Run from the project root. Trained models (~17 MB) and data are small enough to commit directly, so no Git LFS is needed.

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (`models/*.pkl` included, so no training at deploy time).
2. At [share.streamlit.io](https://share.streamlit.io), click **New app**, select the repo, and set the main file to `app.py`.
3. Deploy. `requirements.txt`, `runtime.txt`, and `.streamlit/config.toml` are picked up automatically.

If the first build fails on memory while installing `shap`/`xgboost` (free tier is 1 GB), just retry.

## Limitations

- A portfolio project, not a production system: no fair-lending compliance, no live data pipeline, no validation on a live population.
- Data is from Taiwan, 2005, so patterns may not transfer to other markets or periods.
- The 5:1 cost ratio, 18% interest, and 75% LGD are illustrative assumptions, not real bank data (see sensitivity analysis).
- The engine anchors on current `LIMIT_BAL` because the dataset has no income field. A real system would use income or affordability.
- Isotonic calibration can overfit on small datasets. On a smaller sample, prefer Platt scaling (`method="sigmoid"`).
