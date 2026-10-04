# Code Walkthrough — Credit Risk Predictor & Limit Decisioning

This explains every file in the project, block by block. Read it alongside
the actual `.py` files — line numbers below match those files exactly.

---

## 1. `src/data_pipeline.py` — loads data, builds Model 1's features

```python
1   """
2   data_pipeline.py
...
8   """
10  import pandas as pd
11  import numpy as np
```
Docstring explains the file's purpose. `pandas` for the dataframe, `numpy`
for the numeric operations (log, clip, division) used below.

```python
13  RAW_PATH = "data/UCI_Credit_Card.csv"
15  PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
16  BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
17  PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
```
Constants so column names are typed once, in one place, and reused
everywhere else (`train.py`, `utilization_pipeline.py`, `app.py`) instead
of being retyped and risking a typo. `PAY_COLS` is oddly named `PAY_0` then
`PAY_2..PAY_6` — that's not a bug, it's how the original UCI dataset names
its columns (there's no `PAY_1`). `BILL_COLS`/`PAY_AMT_COLS` are built with
a list comprehension (`f"BILL_AMT{i}"` for `i` 1 through 6) instead of
typed out, since the pattern is regular.

```python
20  def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
21      df = pd.read_csv(path)
22      df = df.rename(columns={"default.payment.next.month": "default"})
23      return df
```
Reads the CSV. Line 22 renames the awkward original column name
(`default.payment.next.month`, with dots in it — inconvenient for
attribute-style pandas access) to a clean `default`, which every other
file then refers to as the target.

```python
26  def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
```
This is the core feature-engineering function. It takes the raw dataframe
and adds 7 new columns. The big docstring (lines 27–50) documents what
each new feature means and why — read it once, it's the "why" for
everything below.

```python
51      df = df.copy()
```
Copies the input dataframe so this function never mutates the caller's
original `df` in place (a common source of subtle bugs in pandas code).

```python
53      limit = df["LIMIT_BAL"].replace(0, np.nan)
```
Pulls out the credit-limit column, but replaces any `0` values with `NaN`
first. This matters because utilization = bill ÷ limit, and dividing by a
real `0` would produce `inf`/`-inf`, poisoning downstream stats; dividing
by `NaN` instead cleanly produces `NaN`, which gets handled explicitly
later (line 77).

```python
55      # --- Utilization features ---
56      util_matrix = df[BILL_COLS].div(limit, axis=0)
57      util_matrix = util_matrix.clip(lower=0, upper=3)
58      df["avg_utilization"] = util_matrix.mean(axis=1)
59      df["max_utilization"] = util_matrix.max(axis=1)
```
- Line 56: divides all 6 `BILL_AMT` columns by `limit` at once.
  `axis=0` tells pandas to align the division down each row (each client's
  6 bills divided by *that client's* limit), not across columns.
- Line 57: clips the result to `[0, 3]`. Some bill amounts are negative
  (a credit balance) or absurdly large relative to a small limit
  (data artifacts), so this caps outliers rather than letting them
  distort the mean/max.
- Lines 58–59: `avg_utilization` = mean utilization across the 6 months;
  `max_utilization` = the single worst month. `axis=1` here means "across
  columns, for each row" — the opposite direction from line 56.

```python
61      # --- Repayment consistency features ---
62      pay_matrix = df[PAY_COLS]
63      df["repayment_consistency"] = (pay_matrix <= 0).sum(axis=1) / len(PAY_COLS)
64      df["months_delayed"] = (pay_matrix > 0).sum(axis=1)
65      df["avg_pay_delay"] = pay_matrix.clip(lower=0).mean(axis=1)
```
In this dataset, `PAY_x <= 0` means "paid on time or early" and `PAY_x > 0`
means "paid N months late."
- Line 63: `(pay_matrix <= 0)` produces a table of `True`/`False`.
  `.sum(axis=1)` counts how many of the 6 months were `True` per row, then
  dividing by 6 turns that count into a fraction between 0 and 1.
- Line 64: same idea, but counting the *delayed* months instead.
- Line 65: `.clip(lower=0)` zeroes out the on-time months (which have
  negative or zero values that don't represent "delay severity"), then
  averages what's left — so this captures *how late*, not just *how often*.

```python
67      # --- Payment-to-bill ratio ---
68      bill_safe = df[BILL_COLS].replace(0, np.nan).values
69      pay_amt = df[PAY_AMT_COLS].values
70      ratio = np.divide(pay_amt, bill_safe, out=np.zeros_like(pay_amt, dtype=float), where=bill_safe != 0)
71      ratio = np.clip(ratio, 0, 5)
72      df["pay_to_bill_ratio"] = ratio.mean(axis=1)
```
- Lines 68–69: pull the bill and payment columns out as raw numpy arrays
  (`.values`) for a lower-level, more controlled division.
- Line 70: `np.divide(..., out=..., where=...)` is numpy's "safe divide" —
  it computes `pay_amt / bill_safe` everywhere `bill_safe != 0`, and just
  writes `0` (the pre-filled `out` array) everywhere it isn't, instead of
  raising a divide-by-zero warning or producing `NaN`/`inf`.
- Line 71: caps the ratio at 5 (paying 5x the bill is already an extreme
  case; anything beyond that is noise, not signal).
- Line 72: averages across the 6 months into one number per client.

```python
74      # --- Scale transform ---
75      df["credit_limit_log"] = np.log1p(df["LIMIT_BAL"])
```
`log1p(x)` = `log(1 + x)`. Credit limits range from 10,000 to 1,000,000 —
heavily right-skewed — and tree/linear models often work better with a
compressed, more evenly-spread version of such a feature. `log1p` (rather
than plain `log`) safely handles a limit of 0 without producing `-inf`.

```python
77      df["avg_utilization"] = df["avg_utilization"].fillna(0)
78      df["max_utilization"] = df["max_utilization"].fillna(0)
79      df["pay_to_bill_ratio"] = df["pay_to_bill_ratio"].fillna(0)
81      return df
```
Cleans up the `NaN`s that line 53's zero-replacement could have produced
(a client with `LIMIT_BAL == 0` would otherwise have `NaN` utilization) —
filled with 0 as a reasonable default, then returns the fully-engineered
dataframe.

```python
84  FEATURE_COLUMNS_RAW = [
85      "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
86      *PAY_COLS, *BILL_COLS, *PAY_AMT_COLS,
87  ]
89  FEATURE_COLUMNS_ENGINEERED = FEATURE_COLUMNS_RAW + [
90      "avg_utilization", "max_utilization",
91      "repayment_consistency", "months_delayed", "avg_pay_delay",
92      "pay_to_bill_ratio", "credit_limit_log",
93  ]
```
Two named feature lists: the 23 raw columns alone, or those plus the 7
engineered ones. The `*PAY_COLS` syntax unpacks the list in place (so this
becomes one flat list of strings, not a list containing a nested list).
Model 1 always ends up using `FEATURE_COLUMNS_ENGINEERED`.

```python
96  def get_dataset(engineered: bool = True):
97      df = load_raw()
98      if engineered:
99          df = engineer_features(df)
100         feature_cols = FEATURE_COLUMNS_ENGINEERED
101     else:
102         feature_cols = FEATURE_COLUMNS_RAW
104     X = df[feature_cols]
105     y = df["default"]
106     return X, y, df
```
The single entry point every training script calls. Loads raw data,
optionally engineers features, and splits it into `X` (features), `y`
(target), and `df` (the full frame, kept around in case a caller wants
extra columns).

```python
109 if __name__ == "__main__":
110     X, y, df = get_dataset(engineered=True)
111     print(f"Rows: {len(df)}, Features: {X.shape[1]}")
112     print(f"Default rate: {y.mean():.3%}")
113     print(df[["avg_utilization", "repayment_consistency", "months_delayed"]].describe())
```
Lets you run `python3 src/data_pipeline.py` directly as a quick sanity
check — prints row/feature counts, the overall default rate, and summary
stats for a few engineered columns. This block never runs when the file is
*imported* by another script (only when executed directly), which is why
every file in this project has one.

---

## 2. `src/train.py` — trains, calibrates, and selects Model 1

```python
16  import json
17  import joblib
18  import numpy as np
19  import pandas as pd
20  from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
21  from sklearn.preprocessing import StandardScaler
22  from sklearn.linear_model import LogisticRegression
23  from sklearn.ensemble import RandomForestClassifier
24  from sklearn.metrics import roc_auc_score, confusion_matrix, brier_score_loss
25  from sklearn.calibration import calibration_curve, CalibratedClassifierCV
26  from sklearn.frozen import FrozenEstimator
27  from xgboost import XGBClassifier
29  from data_pipeline import get_dataset
31  RANDOM_STATE = 42
```
`joblib` saves/loads trained models to disk (`.pkl` files). `json` saves
the metadata file. `brier_score_loss` and `calibration_curve` support the
calibration check explained below. `CalibratedClassifierCV` and
`FrozenEstimator` implement the calibration fix itself.
`RANDOM_STATE = 42` is fixed once and reused everywhere random
splitting/shuffling happens, so re-running this script reproduces the
exact same numbers every time.

```python
34  def build_models(scale_pos_weight: float):
44      return {
45          "LogisticRegression": LogisticRegression(
46              max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
47          ),
48          "RandomForest": RandomForestClassifier(
49              n_estimators=300, max_depth=8, class_weight="balanced",
50              random_state=RANDOM_STATE, n_jobs=-1
51          ),
52          "XGBoost": XGBClassifier(
53              n_estimators=300, max_depth=4, learning_rate=0.05,
54              subsample=0.8, colsample_bytree=0.8,
55              scale_pos_weight=scale_pos_weight,
56              eval_metric="auc", random_state=RANDOM_STATE, n_jobs=-1
57          ),
58      }
```
Returns a dictionary of the three candidate models, each configured to
handle the ~22%/78% class imbalance in its own way: `class_weight="balanced"`
for the two sklearn models automatically reweights the loss function so the
minority class (defaults) counts more; `scale_pos_weight` does the
equivalent for XGBoost, computed from the actual training-set ratio
(line 103) rather than hardcoded. `n_jobs=-1` uses all available CPU cores.

```python
61  def find_cost_based_threshold(y_true, y_proba, cost_fn: float = 5.0, cost_fp: float = 1.0):
69      thresholds = np.linspace(0.05, 0.95, 181)
70      best_threshold, best_cost = 0.5, float("inf")
72      for t in thresholds:
73          preds = (y_proba >= t).astype(int)
74          tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
75          total_cost = cost_fn * fn + cost_fp * fp
76          if total_cost < best_cost:
77              best_cost = total_cost
78              best_threshold = t
80      return best_threshold, best_cost
```
Unchanged from the original design: builds 181 evenly-spaced candidate
thresholds, converts probabilities to binary predictions at each one,
computes `5 × false_negatives + 1 × false_positives` as the total cost, and
keeps whichever threshold minimizes it. What changed is *what data this
function gets called on* — see below.

```python
83  def main():
84      X, y, df = get_dataset(engineered=True)
87      X_trainval, X_test, y_trainval, y_test = train_test_split(
88          X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
89      )
94      X_train, X_val, y_train, y_val = train_test_split(
95          X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE
96      )
```
This is the fix for the project's first correctness issue (see README):
an earlier version split only into train/test and tuned the threshold
directly on the test set, which leaks test-set information into the
reported metrics. Now there are two splits in sequence: line 87 carves off
20% as `X_test`/`y_test` — untouched until the very end of the function.
Line 94 then splits the *remaining* 80% again, `test_size=0.25` of that
80% works out to 20% of the original total, giving a 60/20/20
train/validation/test split overall. `stratify=y` (and `stratify=y_trainval`)
on both splits keeps the ~22% default rate consistent across all three
pieces.

```python
98      scaler = StandardScaler()
99      X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
100     X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
101     X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
```
`StandardScaler` rescales every feature to mean 0, standard deviation 1.
`fit_transform` on the *training* set learns the scaling parameters and
applies them; `transform` (no `fit`) applies those same already-learned
parameters to validation and test — fitting the scaler on either of those
would leak their statistics into training. Logistic Regression needs this;
the tree models don't, which is why scaled versions are only used
selectively further down.

```python
103     scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
104     models = build_models(scale_pos_weight)
106     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
107     results = {}
112     for name, model in models.items():
114         X_cv = X_train_scaled if name == "LogisticRegression" else X_train
115         scores = cross_val_score(model, X_cv, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
116         results[name] = {"cv_auc_mean": scores.mean(), "cv_auc_std": scores.std()}
```
Unchanged from the original design, just now run only on the 60% training
split rather than 80%: 5-fold cross-validation for each of the three
models, training on 4 folds and scoring ROC-AUC on the 5th, five times
over. One real consequence of training on less data: in this project's
actual runs, Random Forest's CV AUC (0.7855) now edges out XGBoost's
(0.7832) — close enough that either could plausibly win depending on the
exact split, which is itself evidence the two are genuinely comparable
rather than one being clearly better.

```python
119     best_name = max(results, key=lambda k: results[k]["cv_auc_mean"])
122     best_model = models[best_name]
123     X_fit = X_train_scaled if best_name == "LogisticRegression" else X_train
124     X_val_eval = X_val_scaled if best_name == "LogisticRegression" else X_val
125     X_test_eval = X_test_scaled if best_name == "LogisticRegression" else X_test
126     best_model.fit(X_fit, y_train)
```
Picks the model with the highest mean CV AUC, then refits it on the full
60% training split (cross-validation only ever trained on 4/5 of that at a
time). This `best_model` is the one later saved as `best_model.pkl` and
used for SHAP — notice it is **only ever fit on `X_train`**, never on
validation or test.

```python
133     y_val_proba_raw = best_model.predict_proba(X_val_eval)[:, 1]
134     raw_brier = brier_score_loss(y_val, y_val_proba_raw)
135     print(f"Raw model Brier score (validation): {raw_brier:.4f}")
```
This is the project's second correctness check: Brier score measures how
far predicted probabilities are from reality (lower is better, 0 is
perfect) — unlike ROC-AUC, which only cares about *ranking* clients
correctly, not whether a "70%" prediction is actually right 70% of the
time. Checked on validation (not test) so test stays clean for the final
numbers. In this project's actual run, the raw model's Brier score (0.173)
turned out to be barely better than always predicting the base rate
(0.172) — the model looked fine by ROC-AUC alone but was quietly
overconfident, which matters a lot once PD feeds into the decision
engine's profit formula as a literal probability.

```python
148     calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(best_model), method="isotonic")
149     calibrated_model.fit(X_val_eval, y_val)
```
The fix. `FrozenEstimator(best_model)` wraps the already-fitted model and
tells `CalibratedClassifierCV` "don't refit this, just calibrate on top of
it." `method="isotonic"` fits a monotonic step-function mapping from raw
predicted probability to a corrected probability, learned entirely from
`X_val`/`y_val` — data `best_model` itself never trained on. This is
deliberately *not* `CalibratedClassifierCV`'s default `cv=5` behavior,
which would instead train 5 full clones of the base model internally
(cross-fitting) — for a 300-tree Random Forest, that turns an ~8MB saved
file into ~40MB for no real accuracy benefit at this dataset's size, which
is why `FrozenEstimator` is used instead.

```python
159     y_val_proba_cal = calibrated_model.predict_proba(X_val_eval)[:, 1]
160     threshold, val_cost = find_cost_based_threshold(y_val, y_val_proba_cal)
```
Threshold tuning now happens on validation-set predictions *from the
calibrated model* — since that's the model the app and decision engine
actually use, the threshold needs to be tuned against the same
probabilities it will later be applied to.

```python
164     y_test_proba_raw = best_model.predict_proba(X_test_eval)[:, 1]
165     y_test_proba_cal = calibrated_model.predict_proba(X_test_eval)[:, 1]
167     test_auc = roc_auc_score(y_test, y_test_proba_cal)
170     preds = (y_test_proba_cal >= threshold).astype(int)
171     tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
172     recall = tp / (tp + fn)
173     precision = tp / (tp + fp)
178     raw_test_brier = brier_score_loss(y_test, y_test_proba_raw)
179     cal_test_brier = brier_score_loss(y_test, y_test_proba_cal)
```
This is the only block in the whole file that touches `X_test`/`y_test`,
and it happens exactly once, after every tuning decision (model choice,
calibration, threshold) has already been locked in using only train and
validation data. ROC-AUC, the confusion matrix, recall, precision, and the
raw-vs-calibrated Brier comparison are all computed here, on data none of
those decisions were allowed to see — these are the numbers reported in
the README.

```python
182     frac_pos, mean_pred = calibration_curve(y_test, y_test_proba_cal, n_bins=10, strategy="quantile")
183     calibration_bins = [
184         {"mean_predicted": float(mp), "observed_fraction": float(fp_)}
185         for mp, fp_ in zip(mean_pred, frac_pos)
186     ]
```
`calibration_curve` buckets test-set predictions into 10 quantile-sized
bins and, for each bin, reports the average predicted probability versus
the actual observed fraction of defaults in that bin. A perfectly
calibrated model would have these two numbers equal in every bin; storing
them lets you inspect exactly where (if anywhere) the calibrated model's
probabilities are still off, rather than trusting a single Brier-score
summary.

```python
188     joblib.dump(best_model, "models/best_model.pkl")
189     joblib.dump(calibrated_model, "models/calibrated_model.pkl")
190     joblib.dump(scaler, "models/scaler.pkl")
191     with open("models/metadata.json", "w") as f:
192         json.dump({...}, f, indent=2)
```
Saves **two** model files instead of one — a deliberate split explained in
the `probability_source` field written into the metadata itself: the raw
`best_model.pkl` is kept only because SHAP's `TreeExplainer` can't look
inside a `CalibratedClassifierCV` wrapper; every actual prediction shown to
a user (the risk score, the threshold decision, the decision engine's PD
input) goes through `calibrated_model.pkl` instead. The metadata also
records `threshold_tuned_on` and `calibration_method` as explicit strings
— not for the code's own use, but so anyone reading `metadata.json` later
(including the tests) can verify what data each number actually came from.

```python
218 if __name__ == "__main__":
219     main()
```
Standard entry point — running `python3 src/train.py` calls `main()`.

---

## 3. `src/explain.py` — SHAP explainability

```python
7   import json
8   import joblib
9   import shap
10  import pandas as pd
13  def load_artifacts(model_dir: str = "models"):
14      model = joblib.load(f"{model_dir}/best_model.pkl")
15      scaler = joblib.load(f"{model_dir}/scaler.pkl")
16      with open(f"{model_dir}/metadata.json") as f:
17          metadata = json.load(f)
18      return model, scaler, metadata
```
Loads the three artifacts `train.py` saved. Parameterized by `model_dir`
so it's not hardcoded, though every caller in this project uses the
default `"models"`.

```python
21  def get_explainer(model, metadata):
24      if metadata["model_name"] in ("XGBoost", "RandomForest"):
25          return shap.TreeExplainer(model)
26      return shap.Explainer(model)
```
SHAP has specialized, exact, fast explainers for tree-based models
(`TreeExplainer`) and a slower general-purpose one for anything else
(`shap.Explainer`, used as a fallback if Logistic Regression had won
instead). The check on line 24 picks the right one based on which model
actually won training.

```python
29  def explain_client(model, explainer, X_row: pd.DataFrame, feature_columns):
34      shap_values = explainer(X_row[feature_columns])
36      values = shap_values.values[0]
37      contributions = sorted(
38          zip(feature_columns, values, X_row[feature_columns].values[0]),
39          key=lambda t: abs(t[1]),
40          reverse=True,
41      )
42      return shap_values, contributions
```
- Line 34: runs the SHAP explainer on a single client row (a one-row
  dataframe), producing one SHAP value per feature — how much that
  feature pushed the prediction up or down from the average.
- Line 36: `shap_values.values[0]` pulls out the raw array of per-feature
  SHAP values for that one row.
- Lines 37–41: `zip` pairs each feature name with its SHAP value and its
  actual raw value, then `sorted(..., key=lambda t: abs(t[1]), reverse=True)`
  sorts those triples by the *absolute size* of the SHAP value, biggest
  impact first — regardless of whether the impact pushed the prediction up
  or down.

```python
45  if __name__ == "__main__":
46      import sys
47      sys.path.insert(0, ".")
48      from data_pipeline import get_dataset
50      model, scaler, metadata = load_artifacts()
51      X, y, df = get_dataset(engineered=True)
52      feature_columns = metadata["feature_columns"]
54      explainer = get_explainer(model, metadata)
56      sample = X.iloc[[0]]
57      shap_values, contributions = explain_client(model, explainer, sample, feature_columns)
59      print("Top 5 contributing features for client 0:")
60      for name, val, raw in contributions[:5]:
61          direction = "increases" if val > 0 else "decreases"
62          print(f"  {name:25s} (value={raw:>10.2f})  {direction} risk by {abs(val):.4f}")
```
A runnable demo: loads the saved model, grabs the very first row of the
dataset (`X.iloc[[0]]` — double brackets keep it as a one-row dataframe,
not a Series), explains it, and prints the top 5 features that moved that
client's risk score, in plain English ("X increases/decreases risk by Y").

---

## 4. `src/utilization_pipeline.py` — Model 2's feature/target setup

```python
15  import pandas as pd
16  import numpy as np
18  from data_pipeline import load_raw, engineer_features, PAY_COLS
```
Reuses `load_raw`/`engineer_features` from Model 1's pipeline rather than
duplicating that logic — `avg_utilization` is already computed there, so
this file just needs to pick the right *input features* around it.

```python
21  UTILIZATION_FEATURE_COLUMNS = [
22      "SEX", "EDUCATION", "MARRIAGE", "AGE",
23      *PAY_COLS,
24      "repayment_consistency", "months_delayed", "avg_pay_delay",
25  ]
27  UTILIZATION_TARGET = "avg_utilization"
```
This is the single most important design decision in the whole file, and
it's explained in the module docstring above it (lines 1–13): the feature
list deliberately **excludes** `BILL_AMT*`, `PAY_AMT*`, `LIMIT_BAL`,
`credit_limit_log`, `max_utilization`, and `pay_to_bill_ratio` — every
column that `avg_utilization` (the target) is directly computed from, or
strongly derived alongside. Including any of those would let the model
"cheat" by nearly re-deriving the answer from its own definition instead
of learning a real pattern. What's left is purely demographic and
repayment-*behavior* information — a genuinely different, harder, but
decision-relevant question.

```python
30  def get_utilization_dataset():
34      df = load_raw()
35      df = engineer_features(df)
37      X = df[UTILIZATION_FEATURE_COLUMNS].copy()
38      y = df[UTILIZATION_TARGET].copy()
39      return X, y, df
```
Loads and engineers the data exactly like Model 1 does (so `avg_utilization`
exists to use as a target), but slices out only the leakage-free feature
columns for `X`. `.copy()` avoids `SettingWithCopyWarning` pitfalls later
if the caller modifies `X` or `y`.

```python
42  if __name__ == "__main__":
43      X, y, df = get_utilization_dataset()
44      print(f"Rows: {len(df)}, Features: {X.shape[1]}")
45      print(f"Target (avg_utilization) stats:\n{y.describe()}")
```
Same pattern as `data_pipeline.py` — a runnable sanity check.

---

## 5. `src/train_utilization.py` — trains Model 2

```python
14  import json
15  import joblib
16  import numpy as np
17  from sklearn.model_selection import train_test_split, cross_val_score
18  from sklearn.linear_model import LinearRegression
19  from sklearn.ensemble import GradientBoostingRegressor
20  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
22  from utilization_pipeline import get_utilization_dataset
24  RANDOM_STATE = 42
```
Structurally a lighter mirror of `train.py`: two regressors instead of
three classifiers, and regression metrics (MAE/RMSE/R²) instead of
classification ones.

```python
27  def main():
28      X, y, df = get_utilization_dataset()
30      X_train, X_test, y_train, y_test = train_test_split(
31          X, y, test_size=0.2, random_state=RANDOM_STATE
32      )
```
80/20 split. No `stratify` here — stratification is a classification
concept (preserving class proportions); it doesn't apply to a continuous
target.

```python
35      models = {
36          "LinearRegression": LinearRegression(),
37          "GradientBoosting": GradientBoostingRegressor(
38              n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
39          ),
40      }
```
A simple linear baseline against a gradient-boosted tree ensemble — the
same "compare a baseline against something stronger" pattern as Model 1,
just with regression models.

```python
44      results = {}
45      for name, model in models.items():
46          scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
47          results[name] = {"cv_r2_mean": scores.mean(), "cv_r2_std": scores.std()}
48          print(f"{name:20s}  R^2 = {scores.mean():.4f}  (+/- {scores.std():.4f})")
```
Plain 5-fold CV (`cv=5`, no explicit `KFold` object needed since there's
no stratification to configure), scored by R² (fraction of variance in
utilization explained by the model).

```python
50      best_name = max(results, key=lambda k: results[k]["cv_r2_mean"])
53      best_model = models[best_name]
54      best_model.fit(X_train, y_train)
56      y_pred = np.clip(best_model.predict(X_test), 0, 3)
57      mae = mean_absolute_error(y_test, y_pred)
58      rmse = mean_squared_error(y_test, y_pred) ** 0.5
59      r2 = r2_score(y_test, y_pred)
```
Selects the higher-scoring model, refits it on the full training set, then
predicts on the untouched test set. Line 56 clips predictions to `[0, 3]`
— the same valid range utilization was clipped to during feature
engineering — since a regression model can technically output any number,
including a nonsensical negative utilization. `mean_squared_error(...) **
0.5` computes RMSE by hand (older scikit-learn versions don't have a
`squared=False` option in every release, so this is the portable way).

```python
66      joblib.dump(best_model, "models/utilization_model.pkl")
67      with open("models/utilization_metadata.json", "w") as f:
68          json.dump({...}, f, indent=2)
```
Same save pattern as `train.py`, into separate files so Model 1's and
Model 2's artifacts never collide.

---

## 6. `src/decision_engine.py` — the non-ML decision layer

The module docstring (lines 1–41) is the most important part of this file
conceptually — it explains *why* this is a formula, not a model, and
derives the "breakeven utilization" idea. The code below is the
implementation of that math.

```python
43  DEFAULT_INTEREST_RATE = 0.18
44  DEFAULT_LGD = 0.75
45  DEFAULT_MIN_MULTIPLE = 0.5
46  DEFAULT_MAX_MULTIPLE = 1.5
47  DEFAULT_DECLINE_PD = 0.5
```
Every assumption the formula depends on, named as constants with the
values it currently uses, at the top of the file where they're easy to
find and change. Every function below accepts these as optional
parameters (with these as defaults), so a caller can override any one of
them without touching the function bodies.

```python
50  def breakeven_utilization(pd, interest_rate=DEFAULT_INTEREST_RATE, lgd=DEFAULT_LGD):
56      if pd >= 1:
57          return float("inf")
58      return (pd * lgd) / ((1 - pd) * interest_rate)
```
Implements the formula from the docstring directly. Line 56 guards against
a `pd` of exactly 1 (certain default), which would otherwise divide by
`(1 - 1) = 0`; returning infinity correctly represents "no utilization
could ever justify more credit to a certain defaulter."

```python
61  def expected_profit(limit, pd, expected_util, interest_rate=DEFAULT_INTEREST_RATE, lgd=DEFAULT_LGD):
63      return (1 - pd) * interest_rate * expected_util * limit - pd * lgd * limit
```
A direct, one-line translation of the profit formula: expected income
(probability of repaying × interest rate × how much of the limit gets
used × the limit itself) minus expected loss (probability of default ×
loss-given-default × the limit).

```python
66  def recommend_limit(
67      current_limit, pd, expected_util,
68      interest_rate=DEFAULT_INTEREST_RATE, lgd=DEFAULT_LGD,
69      min_multiple=DEFAULT_MIN_MULTIPLE, max_multiple=DEFAULT_MAX_MULTIPLE,
70      decline_pd=DEFAULT_DECLINE_PD,
71  ):
85      breakeven = breakeven_utilization(pd, interest_rate, lgd)
87      if pd >= decline_pd or expected_util <= breakeven:
88          best_limit = current_limit * min_multiple
89      else:
90          best_limit = current_limit * max_multiple
92      best_limit = round(float(best_limit), -3)
93      best_profit = expected_profit(best_limit, pd, expected_util, interest_rate, lgd)
95      return best_limit, float(best_profit), float(breakeven)
```
The actual decision. Because the profit formula is linear in `limit` (no
diminishing returns or increasing risk built in), the best possible limit
is always at one extreme of the allowed range — line 87 decides which
extreme: the floor (`min_multiple × current_limit`) if the client is
either too risky outright (`pd >= decline_pd`) or their predicted
utilization doesn't clear breakeven; otherwise the ceiling
(`max_multiple × current_limit`). `round(..., -3)` rounds to the nearest
1,000 (the `-3` means "3 digits before the decimal point," i.e. thousands)
so recommended limits look like realistic numbers instead of
`123456.789`. It then reports the expected profit at that chosen limit and
the breakeven value itself, so callers (like `app.py`) can display *why*
the decision landed where it did.

```python
98  def compare_naive_vs_model_utilization(current_limit, pd, model_util, naive_util=0.5, **kwargs):
108     naive_limit, naive_profit, breakeven = recommend_limit(current_limit, pd, naive_util, **kwargs)
109     model_limit, model_profit, _ = recommend_limit(current_limit, pd, model_util, **kwargs)
110     return {
111         "breakeven_utilization": breakeven,
112         "naive_utilization_assumed": naive_util,
113         "naive_recommended_limit": naive_limit,
114         "naive_expected_profit": naive_profit,
115         "model_predicted_utilization": model_util,
116         "model_recommended_limit": model_limit,
117         "model_expected_profit": model_profit,
118         "limit_delta": model_limit - naive_limit,
119         "decision_flipped": (naive_limit != model_limit),
120     }
```
Calls `recommend_limit` twice for the *same client* — once pretending you
only had a flat, naive utilization guess (`naive_util=0.5` by default),
once using Model 2's actual per-client prediction (`model_util`) — and
packages both outcomes into one dictionary. `**kwargs` forwards any extra
overrides (like a custom interest rate) to both calls identically, so the
only thing that differs between the two calls is the utilization input.
`decision_flipped` is `True` exactly when the two calls landed on
different sides of the breakeven line and thus produced different
recommended limits — the concrete evidence that Model 2 changed the
outcome, not just the profit estimate.

```python
123 if __name__ == "__main__":
126     result_a = compare_naive_vs_model_utilization(current_limit=100000, pd=0.05, model_util=0.65)
133     result_b = compare_naive_vs_model_utilization(current_limit=100000, pd=0.10, model_util=0.30)
```
Two hand-picked runnable examples: Example A (low risk, high utilization)
shows a case where naive and model-based decisions agree; Example B (a
PD/utilization combination straddling the breakeven point) shows a real
flip. These aren't random — they were chosen to demonstrate both outcomes
on purpose.

---

## 7. `app.py` — the Streamlit dashboard tying it all together

```python
18  sys.path.insert(0, "src")
19  from data_pipeline import engineer_features, FEATURE_COLUMNS_ENGINEERED, PAY_COLS
20  from utilization_pipeline import UTILIZATION_FEATURE_COLUMNS
21  from decision_engine import recommend_limit, compare_naive_vs_model_utilization, breakeven_utilization
```
Line 18 adds `src/` to Python's import path so the rest of the app can
import from `data_pipeline`, `utilization_pipeline`, and `decision_engine`
directly, without needing them installed as a package.

```python
28  @st.cache_resource
29  def load_artifacts():
30      model = joblib.load("models/best_model.pkl")
31      scaler = joblib.load("models/scaler.pkl")
32      with open("models/metadata.json") as f:
33          metadata = json.load(f)
34      explainer = shap.TreeExplainer(model) if metadata["model_name"] in ("XGBoost", "RandomForest") \
35          else shap.Explainer(model)
37      util_model = joblib.load("models/utilization_model.pkl")
38      with open("models/utilization_metadata.json") as f:
39          util_metadata = json.load(f)
41      return model, scaler, metadata, explainer, util_model, util_metadata
```
`@st.cache_resource` tells Streamlit to run this function once and cache
the result across reruns — without it, every single UI interaction
(moving a slider, clicking a button) would reload both models and rebuild
the SHAP explainer from scratch, which is slow. Loads both models' full
sets of artifacts in one function.

```python
44  model, scaler, metadata, explainer, util_model, util_metadata = load_artifacts()
45  FEATURES = metadata["feature_columns"]
46  THRESHOLD = metadata["cost_based_threshold"]
```
Runs at module load time (this executes top-to-bottom every time
Streamlit reruns the script, but the `@st.cache_resource` above makes the
actual loading only happen once). Pulls the model's expected feature order
and the tuned decision threshold out of the saved metadata, rather than
hardcoding them here.

```python
64  col1, col2, col3 = st.columns(3)
66  with col1:
67      st.subheader("Credit & Demographics")
68      limit_bal = st.number_input(...)
...
```
Builds a 3-column input form. Each `st.number_input`/`st.slider`/
`st.selectbox` both renders a widget and returns whatever value the user
currently has it set to — these become plain Python variables
(`limit_bal`, `age`, `sex`, etc.) used below.

```python
92  if st.button("Run Full Assessment", type="primary"):
```
Everything below this line only runs when the button is clicked (on
click, Streamlit reruns the whole script top-to-bottom with the button's
state now `True`).

```python
93      row = {
94          "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age,
95          "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3, "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
96      }
97      for i in range(1, 7):
98          row[f"BILL_AMT{i}"] = avg_bill
99          row[f"PAY_AMT{i}"] = avg_pay_amt
101     raw_df = pd.DataFrame([row])
102     engineered_df = engineer_features(raw_df)
```
Assembles a single client's inputs into the same raw column shape the
model was trained on. Since the UI only asks for one *average* bill and
payment amount (not 6 separate months, to keep the form simple), lines
97–99 repeat that single average across all 6 monthly columns. Line 101
wraps the dict in a one-row dataframe (`[row]` — a list containing one
dict — is how pandas builds a single-row frame from a dict). Line 102 runs
it through the *exact same* `engineer_features` function used during
training, guaranteeing the engineered columns are computed identically at
inference time.

```python
105     X_input = engineered_df[FEATURES]
106     X_model = pd.DataFrame(scaler.transform(X_input), columns=FEATURES) if metadata["uses_scaled_input"] else X_input
107     pd_score = float(model.predict_proba(X_model)[0, 1])
108     risk_label = "HIGH RISK — likely default" if pd_score >= THRESHOLD else "LOW RISK — likely to repay"
```
Model 1's inference: selects the exact feature columns/order the model
expects (line 105), scales them only if the winning model actually needs
scaling (line 106 — the same conditional pattern as `train.py`), predicts
a probability (`[0, 1]` picks the "probability of class 1/default" value
for this single row), and labels it against the tuned threshold rather
than the sklearn default of 0.5.

```python
111     X_util = engineered_df[UTILIZATION_FEATURE_COLUMNS]
112     predicted_util = float(np.clip(util_model.predict(X_util)[0], 0, 3))
```
Model 2's inference, using the leakage-free feature subset defined in
`utilization_pipeline.py`. `[0]` takes the single prediction out of the
one-element array `.predict()` returns; clipped to `[0, 3]` for the same
reason as during training.

```python
128     shap_values = explainer(X_input)
129     fig, ax = plt.subplots(figsize=(8, 4))
130     shap.plots.waterfall(shap_values[0], max_display=8, show=False)
131     st.pyplot(fig, use_container_width=True)
132     plt.close(fig)
```
Runs SHAP on this specific client's row, draws a waterfall chart (each bar
shows one feature pushing the prediction up or down from the baseline,
`max_display=8` keeps only the 8 biggest), and renders that matplotlib
figure inside the Streamlit page. `show=False` stops SHAP from also
trying to pop open its own plot window; `plt.close(fig)` frees the figure
from memory afterward so repeated clicks don't leak memory.

```python
143     comparison = compare_naive_vs_model_utilization(
144         current_limit=limit_bal, pd=pd_score, model_util=predicted_util
145     )
```
This is where both models' outputs actually converge into the decision
layer — `pd_score` from Model 1 and `predicted_util` from Model 2 both
feed into the same function from `decision_engine.py` used in the
command-line examples.

```python
147     dec_col1, dec_col2 = st.columns(2)
148     with dec_col1:
149         st.subheader("Naive assumption (flat 50% utilization)")
150         st.metric("Recommended Limit", f"NT$ {comparison['naive_recommended_limit']:,.0f}")
...
157     if comparison["decision_flipped"]:
158         st.warning(...)
159     else:
164         st.info(...)
```
Displays both scenarios side by side, then conditionally shows either a
warning (if `decision_flipped` is `True` — the concrete case where Model 2
mattered) or an informational note (if the two scenarios agreed) —
directly surfacing, in the UI itself, the evidence for why Model 2 is in
the project.

---

## 8. `src/sensitivity_analysis.py` — stress-testing the decision engine's assumptions

```python
18  from decision_engine import recommend_limit, breakeven_utilization
21  CLIENT = {"current_limit": 100000, "pd": 0.08, "expected_util": 0.45}
```
Imports the two functions it needs from `decision_engine.py` directly
(this file does no training, no data loading — it's pure analysis on top
of functions that already exist). `CLIENT` is one fixed, hand-picked
example held constant across every scenario below, so the only thing that
changes between runs is the assumption being tested — not the client.

```python
24  def interest_rate_sensitivity():
29      for rate in [0.12, 0.15, 0.18, 0.21, 0.24, 0.30]:
30          breakeven = breakeven_utilization(CLIENT["pd"], interest_rate=rate)
31          limit, profit, _ = recommend_limit(
32              CLIENT["current_limit"], CLIENT["pd"], CLIENT["expected_util"], interest_rate=rate
33          )
34          print(f"{rate:>14.0%} {breakeven:>15.1%} {limit:>18,.0f}")
```
Holds LGD at its default (0.75) and sweeps the interest rate across a
realistic range (12%–30%), printing the resulting breakeven utilization and
recommended limit at each one. `lgd_sensitivity()` (lines 37–45) does the
mirror image — holds interest rate fixed, sweeps LGD instead.

```python
48  def pd_crossover_point():
53      for pd in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
54          breakeven = breakeven_utilization(pd)
55          limit, profit, _ = recommend_limit(CLIENT["current_limit"], pd, CLIENT["expected_util"])
56          cleared = "above breakeven -> max limit" if CLIENT["expected_util"] > breakeven else "below breakeven -> floor"
57          print(f"  PD={pd:>5.0%}  breakeven={breakeven:>6.1%}  {cleared:30s}  limit={limit:>10,.0f}")
```
This is the most informative of the three scenarios: it holds the client's
utilization fixed at 45% and sweeps PD instead, directly showing *at what
risk level* this specific client's recommendation flips. In the project's
actual run, that crossover happens between PD=0.08 (still above breakeven,
gets the max limit) and PD=0.10 (now below breakeven, gets the floor) —
line 56 computes and prints which side of that line the client falls on at
each PD, rather than just printing raw numbers and leaving the reader to
work it out.

```python
61  if __name__ == "__main__":
62      interest_rate_sensitivity()
63      lgd_sensitivity()
64      pd_crossover_point()
65      print(
66          "\nTakeaway: the recommended limit for a given client only changes "
...
71      )
```
Running all three scenarios in sequence and ending with a one-paragraph,
hardcoded takeaway rather than leaving the reader to infer the pattern
from raw numbers: the decision only moves when an assumption shift pushes
the breakeven line across the client's actual utilization — small shifts
that don't cross that line leave the recommendation unchanged. This is a
direct, visible demonstration of the "bang-bang" shape documented in
`decision_engine.py`'s module docstring, using real numbers instead of
just asserting it.

---

## 9. Tests — what they actually check

**`tests/test_pipeline.py`** (Model 1):
- `test_engineered_features_present` — after running `engineer_features`,
  all 7 new columns actually exist.
- `test_repayment_consistency_range` — that fraction is always between 0
  and 1 (a sanity bound check).
- `test_utilization_capped` — utilization never exceeds the `[0, 3]` clip.
- `test_perfect_payer_has_no_delay` — builds a synthetic client who always
  pays on time with a zero balance, and checks the engineered features
  correctly show 100% consistency and zero delayed months — a targeted
  check with a hand-built, known-correct example, rather than only
  checking the real dataset.
- `test_cost_based_threshold_penalizes_false_negatives` — with perfectly
  separated fake scores, confirms the threshold search finds a
  zero-cost threshold.
- `test_cost_based_threshold_shifts_with_cost_ratio` — with imperfect fake
  scores, confirms that penalizing false negatives more heavily
  (`cost_fn=10` vs `cost_fn=1`) pushes the chosen threshold *down* (casting
  a wider net to catch more real defaulters), which is the whole point of
  cost-based tuning.
- `test_calibration_improves_brier_score` — reads the saved
  `models/metadata.json` and asserts the calibrated model's test Brier
  score is no worse than the raw model's, locking in the calibration fix
  described in `train.py` so a future change can't silently reintroduce
  the original overconfidence problem. Skips (rather than fails) if
  `train.py` hasn't been run yet, since it depends on that output file.
- `test_threshold_not_tuned_on_test_set` — a narrower, intent-focused
  check: asserts the saved metadata's `threshold_tuned_on` field mentions
  "validation", guarding specifically against the original test-set
  leakage bug reappearing even if someone restructures the splits later.

**`tests/test_decision_engine.py`** (Model 2 + decision engine):
- `test_utilization_features_exclude_leakage_columns` — directly asserts
  that none of the leakage-prone columns (`LIMIT_BAL`, `BILL_AMT*`, etc.)
  ever sneak into Model 2's feature list, even if someone edits
  `utilization_pipeline.py` later.
- `test_utilization_dataset_shapes_match` — `X`, `y`, and `df` all have the
  same number of rows, and `X`'s columns match the declared feature list
  exactly.
- `test_breakeven_increases_with_pd` — confirms riskier clients need
  higher utilization to justify more credit, matching the formula's
  intended behavior.
- `test_recommend_limit_approves_max_when_util_above_breakeven` /
  `test_recommend_limit_floors_when_util_below_breakeven` — directly test
  both branches of the bang-bang decision.
- `test_recommend_limit_always_floors_above_decline_pd` — confirms the
  hard override for near-certain defaulters holds even with a maxed-out
  utilization input.
- `test_naive_vs_model_can_flip_decision` / `test_naive_vs_model_same_side_does_not_flip`
  — regression tests locking in the exact two demo scenarios from
  `decision_engine.py`'s own `__main__` block, so future edits can't
  silently break the flip that justifies Model 2's inclusion.
- `test_sensitivity_example_client_crosses_breakeven_across_pd_range` —
  imports the fixed `CLIENT` dict directly from `sensitivity_analysis.py`
  (rather than redefining it) and asserts that client's recommendation
  genuinely flips between PD=0.08 and PD=0.12, locking in the specific
  finding the sensitivity analysis reports rather than just checking the
  underlying formula in isolation.
