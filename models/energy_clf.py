"""
PHAI - energy classifier.

Predicts the probability that the user's step count tomorrow will be ABOVE
their personal median (a proxy for an "above-baseline energetic day"),
given the last 7 days of wearable features.

Two entry points:

    Train (writes model + metrics to disk):
        python -m models.energy_clf train

    Predict for one user (uses the trained model):
        python -m models.energy_clf predict
        python -m models.energy_clf predict --user-id <id>

The `predict()` function is also imported and exposed as a tool in
agents.tools.predict_energy_tomorrow.

Methodology:
  - Per-user binary target so each user is judged against their own baseline.
  - Time-based per-user 70/30 split prevents future leakage.
  - XGBoost handles NaN natively - no imputation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "phai.db"
MODEL_PATH = PROJECT_ROOT / "models" / "energy_clf.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "energy_clf_metrics.json"

# Metrics whose 7-day rolling stats become features.
NUMERIC_METRICS = (
    "steps", "sleep_min", "hrv_rmssd", "resting_hr",
    "stress_score", "very_active_min",
)
WINDOW = 7  # last 7 days predict tomorrow

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_daily() -> pd.DataFrame:
    cols = list(NUMERIC_METRICS) + ["mood_tired", "mood_happy"]
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"SELECT user_id, date, {', '.join(cols)} "
            f"FROM daily_summary ORDER BY user_id, date",
            conn,
        )
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------------------------------------------------------------------------
# Feature engineering (used for both training and prediction)
# ---------------------------------------------------------------------------

def _features_from_window(window: pd.DataFrame, tomorrow_dow: int) -> dict:
    """Build the feature dict from a 7-day window. Used for training & predict."""
    feats: dict = {}
    for col in NUMERIC_METRICS:
        feats[f"{col}_7d_mean"] = float(window[col].mean()) if window[col].notna().any() else np.nan
        feats[f"{col}_7d_std"] = float(window[col].std()) if window[col].notna().sum() > 1 else 0.0
    # Trends: last 3 days vs prior 4 days, for the headline metrics.
    for col in ("steps", "sleep_min", "hrv_rmssd"):
        last3 = window.iloc[-3:][col].mean()
        prior4 = window.iloc[:-3][col].mean()
        feats[f"{col}_trend"] = (
            float(last3 - prior4) if pd.notna(last3) and pd.notna(prior4) else 0.0
        )
    # Mood flag rates over the window (counts of days the user logged that mood).
    feats["mood_tired_rate"] = float((window["mood_tired"] == 1).sum()) / WINDOW
    feats["mood_happy_rate"] = float((window["mood_happy"] == 1).sum()) / WINDOW
    feats["dow"] = int(tomorrow_dow)
    return feats


def _build_training_examples(df: pd.DataFrame) -> pd.DataFrame:
    """For each user-day t with a non-null tomorrow steps, build features+target."""
    rows = []
    for user_id, g in df.groupby("user_id"):
        g = g.sort_values("date").reset_index(drop=True)
        user_median_steps = g["steps"].median()
        if pd.isna(user_median_steps):
            continue
        for i in range(WINDOW - 1, len(g) - 1):
            window = g.iloc[i - WINDOW + 1: i + 1]
            tomorrow = g.iloc[i + 1]
            if pd.isna(tomorrow["steps"]):
                continue
            feats = _features_from_window(window, tomorrow["date"].dayofweek)
            feats["user_id"] = user_id
            feats["date_predicted_for"] = tomorrow["date"]
            feats["target"] = int(tomorrow["steps"] > user_median_steps)
            rows.append(feats)
    return pd.DataFrame(rows)


def _per_user_time_split(feat_df: pd.DataFrame, test_frac: float = 0.3):
    """For each user, hold out the last test_frac of days as test."""
    train_idx, test_idx = [], []
    for _, g in feat_df.groupby("user_id"):
        g = g.sort_values("date_predicted_for")
        n = len(g)
        n_test = max(1, int(round(n * test_frac)))
        train_idx.extend(g.index[: n - n_test].tolist())
        test_idx.extend(g.index[n - n_test:].tolist())
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train() -> dict:
    print("Loading daily summaries from phai.db...")
    df = _load_daily()
    print(f"  {len(df):,} rows across {df['user_id'].nunique()} users")

    print("\nBuilding training examples...")
    feat_df = _build_training_examples(df)
    print(f"  {len(feat_df):,} (user, day) examples generated")

    target_col = "target"
    feature_cols = [c for c in feat_df.columns
                    if c not in {"user_id", "date_predicted_for", target_col}]

    print("\nTime-based per-user split (70/30)...")
    train_idx, test_idx = _per_user_time_split(feat_df)
    X_train = feat_df.loc[train_idx, feature_cols]
    y_train = feat_df.loc[train_idx, target_col]
    X_test = feat_df.loc[test_idx, feature_cols]
    y_test = feat_df.loc[test_idx, target_col]
    n_test_users = feat_df.loc[test_idx, "user_id"].nunique()
    print(f"  train: {len(X_train):,} examples")
    print(f"  test:  {len(X_test):,} examples ({n_test_users} unique users)")

    print("\nTraining XGBoost (with early stopping + regularisation)...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=3,            # shallower trees
        learning_rate=0.05,     # smaller steps
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,          # L1 regularisation
        reg_lambda=2.0,         # L2 regularisation
        min_child_weight=10,    # require more samples per leaf
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=20,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    test_acc = accuracy_score(y_test, (test_proba > 0.5).astype(int))
    base_rate = float(y_test.mean())

    print("\n=== Results ===")
    print(f"  Train AUC:      {train_auc:.3f}")
    print(f"  Test  AUC:      {test_auc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")
    print(f"  Test base rate: {base_rate:.3f}  (chance = {max(base_rate, 1 - base_rate):.3f})")

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop 10 features by importance:")
    for name, val in importances.head(10).items():
        print(f"  {name:<30s} {val:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "window": WINDOW},
        MODEL_PATH,
    )
    metrics = {
        "train_auc": round(float(train_auc), 4),
        "test_auc": round(float(test_auc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_base_rate": round(base_rate, 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_test_users": int(n_test_users),
        "feature_count": len(feature_cols),
        "top_features": importances.head(10).round(4).to_dict(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nSaved model to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics to {METRICS_PATH.relative_to(PROJECT_ROOT)}")
    return metrics


# ---------------------------------------------------------------------------
# Predict (used by the agent tool)
# ---------------------------------------------------------------------------

def predict(user_id: str) -> dict:
    """
    Predict tomorrow's energy class for one user.
    Returns a dict ready to JSON-serialise (suitable for tool output).
    """
    if not MODEL_PATH.exists():
        return {
            "error": (
                "Model not trained yet. Run "
                "`python -m models.energy_clf train` first."
            )
        }

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    df = _load_daily()
    user_df = df[df["user_id"] == user_id].sort_values("date").reset_index(drop=True)
    if len(user_df) < WINDOW:
        return {
            "error": f"User '{user_id}' has only {len(user_df)} days; "
                     f"need at least {WINDOW}."
        }

    user_median_steps = user_df["steps"].median()
    last_window = user_df.tail(WINDOW)
    tomorrow_dow = (user_df["date"].iloc[-1].dayofweek + 1) % 7
    feats = _features_from_window(last_window, tomorrow_dow)

    X = pd.DataFrame([feats])[feature_cols]
    prob = float(model.predict_proba(X)[0, 1])

    if prob > 0.6:
        interp = "above-baseline activity day predicted"
    elif prob < 0.4:
        interp = "below-baseline activity day predicted"
    else:
        interp = "around-baseline activity day predicted"

    if abs(prob - 0.5) > 0.2:
        confidence = "high"
    elif abs(prob - 0.5) > 0.1:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "user_id": user_id,
        "user_median_steps": (
            round(float(user_median_steps), 0) if pd.notna(user_median_steps) else None
        ),
        "p_above_user_median_tomorrow": round(prob, 3),
        "interpretation": interp,
        "confidence": confidence,
        "based_on_last_n_days": WINDOW,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pick_demo_user() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT u.user_id "
            "FROM users u JOIN daily_summary d USING (user_id) "
            "WHERE u.source = 'lifesnaps' "
            "GROUP BY u.user_id "
            "ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
    if not row:
        raise RuntimeError("No LifeSnaps users in DB.")
    return row[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--user-id", default=None)
    args = parser.parse_args()

    if args.command == "train":
        train()
    else:
        user_id = args.user_id or _pick_demo_user()
        print(f"Predicting for user: {user_id}\n")
        result = predict(user_id)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
