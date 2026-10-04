# Credit Risk Predictor & Limit Decisioning

An end-to-end ML system with two models feeding a business decision layer,
using the UCI "Default of Credit Card Clients" dataset (Yeh & Lien, 2009):

1. **Model 1 (classifier):** predicts probability of default, compares
   three models under class imbalance, tunes a cost-based decision
   threshold on a held-out validation split, and calibrates its
   probabilities so they can be used in a profit calculation, not just for
   ranking.
2. **Model 2 (regressor):** predicts future credit utilization from
   demographics and repayment *behavior* only — deliberately excluding
   bill/payment dollar amounts, since utilization is defined from those and
   including them would let the model leak the answer from its own inputs.
3. **Decision engine (not ML):** combines both models' outputs into a
   recommended credit limit via an expected-profit rule. There's no
   ground-truth "correct limit" to fit — the dataset's existing limits just
   reflect a bank's past policy — so this is a rule, not a third model.

Individual predictions are explained with SHAP, assumptions behind the
decision rule are stress-tested with a sensitivity analysis, and everything
is deployed as an interactive Streamlit app.

**Live demo:** _add your Streamlit Community Cloud URL here after deploying_

---

## Why two models feeding a decision layer, not one model predicting a limit

An earlier, simpler design would train a regression model directly on the
dataset's `LIMIT_BAL` column. That's tempting but weak: `LIMIT_BAL` is
whatever the bank's *existing* policy assigned, not a ground-truth optimal
limit, so a model trained on it just learns to imitate that historical
policy (including its blind spots) — there's no way to know if the target
itself is even good decisioning to imitate.

Instead, this project uses ML for what it can actually verify — predicting
default risk and predicting utilization, both against real observed
outcomes — and applies an explicit, auditable decision rule on top for the
limit itself. That mirrors how real credit systems separate a statistical
risk model from a policy layer, and it means every recommended limit can be
traced back to a formula, not a black box.

## Why Model 2 exists at all — proof it isn't just for show

Two models can look forced if the second doesn't change anything. Here it
does: the decision engine's optimal limit depends on whether predicted
utilization is above or below a **breakeven utilization** that rises with
PD (see `src/decision_engine.py`). A flat, naive utilization assumption
(e.g. "assume everyone uses 50%") can land on the wrong side of that
breakeven point for a given client — the app runs both the naive assumption
and Model 2's real prediction side by side and flags it when they disagree
enough to flip the recommended limit. That's the concrete, demonstrable
reason Model 2 earns its place instead of padding the project.

---

## Two correctness issues found and fixed during development

Both of these were caught by deliberately checking for them, not by
accident, and both are now covered by regression tests so they can't
silently come back.

**1. Threshold tuning was leaking the test set.**
An earlier version picked the cost-based decision threshold by sweeping it
directly against the test set, then reported recall/precision on that same
test set at that threshold — which quietly inflates the reported numbers,
since the threshold was chosen *to* fit that data. Fixed by splitting the
data three ways (60% train / 20% validation / 20% test): the threshold is
now tuned only on validation, and every reported metric comes from the
test set, which the threshold search never saw.
(`tests/test_pipeline.py::test_threshold_not_tuned_on_test_set` guards
against this regressing.)

**2. The model's probabilities were poorly calibrated.**
Checking this wasn't optional: the decision engine's expected-profit
formula treats PD as a literal probability, not just a ranking score, so a
model that says "70%" needs to actually be right about 70% of the time.
The raw model's Brier score on the test set (0.173) was barely better than
always predicting the base rate (0.172) — it looked fine on ROC-AUC (which
only cares about ranking) but was quietly overconfident at every
probability level (e.g. clients scored at ~86% actually defaulted ~69% of
the time). Fixed with isotonic calibration fit on top of the already-trained model
(`CalibratedClassifierCV` wrapping a `FrozenEstimator`, fit only on the
validation split — never train or test): the calibrated test Brier score
dropped to 0.136, and the calibration curve now tracks closely across all
probability bins. Using `FrozenEstimator` instead of ordinary k-fold
cross-fitting also keeps the saved model small — cross-fitting would store
5 separate clones of the 300-tree Random Forest internally (~40MB) for no
accuracy benefit at this dataset size, versus ~8MB by calibrating the one
already-trained model directly.
(`tests/test_pipeline.py::test_calibration_improves_brier_score` guards
against this regressing.)

One consequence worth noting: the raw model is kept and saved separately,
because SHAP's `TreeExplainer` can't look inside a `CalibratedClassifierCV`
wrapper. The app uses the **calibrated** model for the reported risk score,
the decision threshold, and the decision engine's input — and the **raw**
model only to generate the SHAP explanation shown alongside it.

---

## Results (from an actual run of this code)

**Model 1 — Default classification**

| Model | 5-fold CV ROC-AUC |
|---|---|
| Logistic Regression | 0.7610 |
| **Random Forest (selected)** | **0.7855** |
| XGBoost | 0.7832 |

- **Held-out test ROC-AUC: 0.7777**
- **Class imbalance:** 22.1% of clients defaulted (a realistic, moderately
  imbalanced target — not a toy 50/50 split)
- **Cost-based threshold: 0.170**, tuned on a validation split (never the
  test set) by minimizing total cost under a 5:1 penalty for false
  negatives (approving a client who defaults) vs. false positives
  (flagging a client who would have repaid)
- At that threshold, on the untouched test set: **76.1% recall** on real
  defaulters, **36.1% precision**
- **Calibration:** Brier score 0.136 (calibrated) vs. 0.173 (raw model) —
  see the section above

Random Forest and XGBoost are close enough (0.7855 vs. 0.7832 CV AUC) that
which one wins can shift slightly with the exact train/validation/test
split — the comparison itself, and the fact that both clearly beat Logistic
Regression's 0.761, is the meaningful result here, not a specific ranking
between the two tree models.

**Model 2 — Utilization regression**

| Model | 5-fold CV R² |
|---|---|
| Linear Regression | 0.436 |
| **Gradient Boosting (selected)** | **0.555** |

- **Held-out test R²: 0.565, MAE: 0.169, RMSE: 0.230**
  (average utilization in the data is 0.373, so an MAE of 0.169 is a
  meaningfully informative prediction, not noise)
- Deliberately trained *without* bill/payment amounts or credit limit, to
  avoid leaking the target's own definition (see `src/utilization_pipeline.py`)

These numbers come from actually running `train.py` and
`train_utilization.py` against the real 30,000-row dataset — not simulated
or estimated.

---

## Sensitivity analysis: how much do the decision engine's assumptions matter?

The decision engine's profit formula rests on three assumed constants:
interest rate (18%), loss-given-default (75%), and Model 1's 5:1
false-negative cost ratio. None are derived from real bank loss data.
`src/sensitivity_analysis.py` varies each one across a realistic range for
a fixed example client and shows exactly when the recommendation changes:

```
python3 src/sensitivity_analysis.py
```

The finding: because the decision is a breakeven comparison (see below),
the recommended limit for a given client only moves when an assumption
shift pushes the breakeven line across that client's *actual* predicted
utilization — small shifts that don't cross that line leave the decision
unchanged. For the example client (PD=0.08, predicted utilization=45%),
dropping the interest rate to 12% or raising LGD to 95% is enough to flip
the recommendation; moderate changes within that range are not. The same
client's recommendation also flips somewhere between PD=0.08 and PD=0.10,
which is a cleaner way to see how sensitive the decision is to the
underlying risk model than looking at the profit number alone.

---

## Project Structure

```
CreditRiskPredictor/
├── .streamlit/
│   └── config.toml                # theme (see "Design" below)
├── .github/workflows/
│   └── tests.yml                  # CI: runs the test suite on every push
├── data/
│   └── UCI_Credit_Card.csv        # 30,000 rows, 23 features + target
├── src/
│   ├── data_pipeline.py           # loading + feature engineering for Model 1
│   ├── train.py                   # Model 1: comparison, selection, calibration, threshold tuning
│   ├── explain.py                 # SHAP explainability helpers
│   ├── utilization_pipeline.py    # leakage-free feature/target setup for Model 2
│   ├── train_utilization.py       # Model 2: utilization regression
│   ├── decision_engine.py         # non-ML decision rule: PD + utilization -> limit
│   └── sensitivity_analysis.py    # stress-tests the decision engine's assumptions
├── tests/
│   ├── test_pipeline.py           # Model 1 feature engineering, threshold, and calibration tests
│   └── test_decision_engine.py    # Model 2 leakage checks + decision engine tests
├── models/                        # saved models, scaler, metadata (generated)
├── app.py                         # Streamlit dashboard (both models + decision engine)
├── requirements.txt                # pinned to tested versions
├── runtime.txt                     # Python version for Streamlit Community Cloud
├── LICENSE
└── CODE_WALKTHROUGH.md            # line-by-line explanation of every file
```

## Engineered Features (Model 1)

Beyond the 23 raw columns (credit limit, demographics, 6 months of payment
status/bills/payments), the pipeline derives:

- **`avg_utilization`** / **`max_utilization`** — bill amount ÷ credit limit,
  averaged and peak across 6 months. Utilization is one of the strongest
  known predictors in real credit scoring.
- **`repayment_consistency`** — fraction of the 6 months paid on time or early.
- **`months_delayed`** — count of months with any payment delay.
- **`avg_pay_delay`** — average delay *severity* (not just count).
- **`pay_to_bill_ratio`** — how much of what's owed actually gets paid.
- **`credit_limit_log`** — log-transformed credit limit (raw values are
  heavily right-skewed).

In the SHAP analysis, `PAY_0` (most recent payment status), `avg_pay_delay`,
and `repayment_consistency` consistently rank among the top predictors for
individual clients — confirming the engineered features carry real signal,
not just the raw columns.

## Features for Model 2 (utilization)

`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, the 6 `PAY_x` repayment-status
columns, plus `repayment_consistency`, `months_delayed`, `avg_pay_delay` —
i.e. who the client is and how reliably they've paid, nothing derived from
bill or payment dollar amounts. See the docstring in
`src/utilization_pipeline.py` for why those are excluded.

## The Decision Engine

```
expected_profit(L) = (1 - PD) * interest_rate * expected_util * L
                      - PD * loss_given_default * L
```

Both terms scale linearly with the candidate limit `L`, so the sign of
marginal profit per dollar of limit doesn't depend on `L` — only on whether
predicted utilization clears a **breakeven utilization** that rises with PD:

```
breakeven_utilization(PD) = (PD * LGD) / ((1 - PD) * interest_rate)
```

Above breakeven, extending more limit is always better (approve up to a
policy ceiling); below it, more limit only adds risk (floor the limit).
This "bang-bang" shape isn't a bug — real underwriting behaves similarly,
extending maximum room to profitable, engaged, low-risk clients and
minimizing exposure elsewhere, rather than picking timid limits in between.
Interest rate (18%) and loss-given-default (75%) are illustrative industry
rule-of-thumb assumptions, stated explicitly — same as the 5:1 cost ratio
in Model 1's threshold tuning. See the sensitivity analysis above for how
much those assumptions actually matter.

---

## Setup (local)

```bash
pip install -r requirements.txt
```

### 1. Train both models

```bash
PYTHONPATH=src python3 src/train.py               # Model 1: default risk (comparison + calibration + threshold)
PYTHONPATH=src python3 src/train_utilization.py    # Model 2: utilization
```

Run from the project root (both scripts expect `data/` and `models/`
relative to it). This produces `models/best_model.pkl` (raw, for SHAP),
`models/calibrated_model.pkl` (calibrated, used everywhere else),
`models/scaler.pkl`, `models/metadata.json`,
`models/utilization_model.pkl`, and `models/utilization_metadata.json`.

### 2. Run the tests

```bash
PYTHONPATH=src pytest tests/ -v
```

### 3. Check the decision engine's assumptions

```bash
PYTHONPATH=src python3 src/sensitivity_analysis.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter a client's profile to get a default
probability with a SHAP waterfall explanation, a predicted utilization, and
a recommended credit limit — shown alongside what a naive utilization
assumption would have recommended, so you can see when Model 2 actually
changes the outcome.

---

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (see "Putting this on GitHub" below) —
   `models/*.pkl` are committed, so no training step is needed at deploy time.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick this repository and branch, and set the main
   file path to `app.py`.
4. Streamlit Cloud reads `requirements.txt` and `runtime.txt` automatically
   — no extra configuration needed. `.streamlit/config.toml` is picked up
   automatically too, so the app deploys with its intended theme.
5. Deploy. The first build takes a few minutes (installing `xgboost`,
   `shap`, etc.); subsequent deploys from new commits are faster.
6. Once live, put the app's URL at the top of this README.

If the build fails on memory during `shap`/`xgboost` installation, Streamlit
Community Cloud's free tier has a 1 GB resource limit — retry the deploy
rather than changing dependencies, as this is usually a transient build
issue, not a project problem.

## Putting this on GitHub

```bash
git init
git add .
git commit -m "Credit risk prediction and limit decisioning system"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

`models/*.pkl` (~17 MB combined — `best_model.pkl` and `calibrated_model.pkl`
are each ~8 MB, `utilization_model.pkl` is much smaller) and
`data/UCI_Credit_Card.csv` (~2.8 MB) are all comfortably under GitHub's
100 MB per-file limit and small enough to commit directly — no Git LFS
needed.

---

## Design

The Streamlit theme (`.streamlit/config.toml`) uses a deliberately
restrained palette rather than Streamlit's default red accent, suited to a
risk/underwriting tool: ink-navy text (`#10223D`), a muted slate for
secondary text (`#475569`), a soft off-white background (`#F7F8FA`) instead
of stark white, and a muted teal accent (`#0F766E`) for primary actions.
Semantic colors (risk warnings, success states) are left to Streamlit's
built-in `st.error`/`st.success`/`st.warning` components rather than
overridden, so risk signals stay visually distinct from the primary accent.

---

## Design Notes

- **Why compare three models instead of just using one:** the comparison
  itself is the point — it demonstrates that the two tree-based models
  (Random Forest, XGBoost) clearly beat Logistic Regression (0.785/0.783 vs.
  0.761 CV AUC), while being close enough to each other that the choice
  between them isn't the interesting result. A resume claim of "selected
  model X" is only meaningful if there's an actual comparison behind it.

- **Why the threshold is tuned on validation, not test:** explained above
  under "Two correctness issues found and fixed."

- **Why calibration matters here specifically:** explained above. The
  short version is that ROC-AUC and a cost-based threshold only need the
  model to rank clients correctly, but the decision engine's profit formula
  multiplies PD directly into a dollar calculation — so an overconfident
  model doesn't just rank well and score well, it actively produces wrong
  limit recommendations.

- **Why SHAP over simpler feature importance:** SHAP explains *individual*
  predictions ("why was *this* client flagged"), which is what's actually
  required in regulated lending contexts — a bank has to be able to explain
  an individual credit decision, not just say "utilization matters on
  average."

- **Class imbalance handling:** each model handles the ~22%/78% split
  differently — `class_weight='balanced'` for Logistic Regression and
  Random Forest, `scale_pos_weight` for XGBoost — rather than naively
  training on the raw imbalanced data or oversampling, which can distort
  the probability calibration this project specifically checks for.

- **Why Model 2 is lighter than Model 1:** it's a supporting signal for the
  decision engine, not the centerpiece of the project — one baseline vs.
  gradient boosting comparison, one evaluation pass, no threshold tuning
  (regression has no threshold to tune). Giving it equal weight to Model 1
  would be effort spent for its own sake rather than where it earns value.

---

## Honest Limitations

- This is a portfolio/demonstration project, not a production credit
  decisioning system — it doesn't account for regulatory fair-lending
  requirements, doesn't have a real-time data pipeline, and hasn't been
  validated against a live population.
- The dataset is from Taiwan in 2005 — patterns here (which features matter,
  what "normal" utilization looks like) don't necessarily transfer directly
  to other countries or time periods.
- The 5:1 false-negative-to-false-positive cost ratio, the 18% interest
  rate, and the 75% loss-given-default are illustrative assumptions, not
  derived from a real bank's actual loss data — the sensitivity analysis
  above shows how much they actually move the final recommendation.
- The decision engine uses each client's *current* `LIMIT_BAL` as a
  capacity anchor, since the dataset has no verified income field — a real
  system would anchor on income/affordability instead.
- Model 2's R² (0.565) reflects a genuinely harder, leakage-free problem —
  predicting utilization from behavior alone rather than from the bill
  amounts utilization is literally defined from — so it's intentionally
  more modest than Model 1's AUC, not a weaker effort.
- Calibration was fixed with isotonic regression on this dataset; isotonic
  calibration can overfit on much smaller datasets than this one (30,000
  rows) — worth switching to Platt scaling (`method="sigmoid"`) if reusing
  this approach on a smaller sample.
