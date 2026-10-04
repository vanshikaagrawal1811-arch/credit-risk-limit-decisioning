"""
train_utilization.py
Trains Model 2: a regression model predicting average credit utilization
from demographics + repayment behavior (see utilization_pipeline.py for why
bill/payment dollar amounts are excluded).

This is intentionally a lighter-weight companion to train.py's default
model: one comparison (Linear Regression baseline vs. Gradient Boosting),
one round of evaluation, no threshold tuning. Its job is to feed a usable
utilization estimate into the decision engine, not to be the centerpiece
of the project.
"""

import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utilization_pipeline import get_utilization_dataset

RANDOM_STATE = 42


def main():
    X, y, df = get_utilization_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    print("=" * 60)
    print("5-FOLD CROSS-VALIDATED R^2 (on training data)")
    print("=" * 60)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        results[name] = {"cv_r2_mean": scores.mean(), "cv_r2_std": scores.std()}
        print(f"{name:20s}  R^2 = {scores.mean():.4f}  (+/- {scores.std():.4f})")

    best_name = max(results, key=lambda k: results[k]["cv_r2_mean"])
    print(f"\nSelected model: {best_name}")

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    y_pred = np.clip(best_model.predict(X_test), 0, 3)  # utilization can't be negative
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"Held-out test MAE:  {mae:.4f}")
    print(f"Held-out test RMSE: {rmse:.4f}")
    print(f"Held-out test R^2:  {r2:.4f}")
    print(f"(for reference, average utilization in the data is {y.mean():.3f})")

    joblib.dump(best_model, "models/utilization_model.pkl")
    with open("models/utilization_metadata.json", "w") as f:
        json.dump({
            "model_name": best_name,
            "cv_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
            "test_mae": float(mae),
            "test_rmse": float(rmse),
            "test_r2": float(r2),
            "target_mean": float(y.mean()),
            "feature_columns": list(X.columns),
        }, f, indent=2)

    print("\nSaved: models/utilization_model.pkl, models/utilization_metadata.json")


if __name__ == "__main__":
    main()
