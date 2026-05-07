"""
PHAI ETL - load the LifeSnaps daily CSV into phai.db.

Run from the project root:
    python etl/load_lifesnaps.py

Idempotent: any rows previously inserted with source='lifesnaps' are
deleted first, then re-inserted. Re-run any time the cleaning logic
changes.

Inserts into:
    users           (one row per LifeSnaps participant)
    daily_summary   (one row per user-day)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lifesnaps"
    / "daily_fitbit_sema_df_unprocessed.csv"
)
DB_PATH = PROJECT_ROOT / "phai.db"

# LifeSnaps column -> our daily_summary column.
# Anything not in this map is dropped (intentional; see Step 5 analysis).
COLUMN_MAP = {
    # Activity
    "steps": "steps",
    "distance": "distance",
    "calories": "calories",
    "lightly_active_minutes": "lightly_active_min",
    "moderately_active_minutes": "moderately_active_min",
    "very_active_minutes": "very_active_min",
    "sedentary_minutes": "sedentary_min",
    # Heart
    "bpm": "bpm_avg",
    "resting_hr": "resting_hr",
    "rmssd": "hrv_rmssd",
    "nremhr": "nrem_hr",
    "full_sleep_breathing_rate": "breathing_rate",
    # Sleep
    "minutesAsleep": "sleep_min",
    "sleep_efficiency": "sleep_efficiency",
    "minutesToFallAsleep": "minutes_to_fall_asleep",
    "minutesAwake": "minutes_awake",
    # Wellness scores (Fitbit-derived)
    "stress_score": "stress_score",
    "sleep_points_percentage": "sleep_score_pct",
    "exertion_points_percentage": "exertion_score_pct",
    "responsiveness_points_percentage": "responsiveness_score_pct",
    "nightly_temperature": "nightly_temp",
    # Mood self-report (binary, NULL = user didn't respond that day)
    "ALERT": "mood_alert",
    "HAPPY": "mood_happy",
    "NEUTRAL": "mood_neutral",
    "RESTED/RELAXED": "mood_rested",
    "SAD": "mood_sad",
    "TENSE/ANXIOUS": "mood_tense",
    "TIRED": "mood_tired",
}

# These columns have spurious zeros in the raw data (sensor errors).
# Replace 0 with NULL so the agent doesn't report bogus values.
ZERO_AS_NULL = ("bpm_avg", "nrem_hr", "breathing_rate", "hrv_rmssd")


# ----- helpers --------------------------------------------------------------

def first_non_null(series: pd.Series) -> Any:
    """First non-null value in the series; None if all null."""
    s = series.dropna()
    return s.iloc[0] if len(s) else None


def build_users(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per LifeSnaps user with their demographics."""
    grouped = (
        raw.groupby("id")
        .agg({"age": first_non_null, "gender": first_non_null, "bmi": first_non_null})
        .reset_index()
        .rename(columns={"id": "user_id"})
    )
    # age and bmi arrived as strings - coerce, NaN if unparseable.
    grouped["age"] = pd.to_numeric(grouped["age"], errors="coerce").astype("Int64")
    grouped["bmi"] = pd.to_numeric(grouped["bmi"], errors="coerce")
    grouped["source"] = "lifesnaps"
    grouped["synthetic"] = 0
    return grouped[["user_id", "source", "synthetic", "age", "gender", "bmi"]]


def build_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (user, date) with the metrics we keep."""
    cols = ["id", "date"] + list(COLUMN_MAP.keys())
    df = raw[cols].copy()
    df = df.rename(columns={"id": "user_id", **COLUMN_MAP})

    # ISO date string for clean sort-as-text in SQLite.
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Replace sentinel zeros with NULL.
    for col in ZERO_AS_NULL:
        if col in df.columns:
            df.loc[df[col] == 0, col] = pd.NA

    # Mood columns: nullable integer (0 / 1 / NULL).
    for col in df.columns:
        if col.startswith("mood_"):
            df[col] = df[col].astype("Int64")

    return df


# ----- main -----------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"phai.db not found at {DB_PATH}. Run `python db/init_db.py` first."
        )

    print(f"Loading {CSV_PATH.name}...")
    raw = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"  {len(raw):,} rows, {raw['id'].nunique()} users")

    users = build_users(raw)
    daily = build_daily(raw)
    print(f"  prepared {len(users)} users and {len(daily):,} daily rows")
    print()

    conn = sqlite3.connect(DB_PATH)
    try:
        # Idempotent: clear any prior LifeSnaps load.
        conn.execute(
            "DELETE FROM daily_summary "
            "WHERE user_id IN (SELECT user_id FROM users WHERE source='lifesnaps')"
        )
        conn.execute("DELETE FROM users WHERE source='lifesnaps'")
        conn.commit()

        users.to_sql("users", conn, if_exists="append", index=False)
        daily.to_sql("daily_summary", conn, if_exists="append", index=False)
        conn.commit()

        u_count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE source='lifesnaps'"
        ).fetchone()[0]
        d_count = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]

        print("Loaded into phai.db:")
        print(f"  users (source='lifesnaps'):  {u_count}")
        print(f"  daily_summary rows:          {d_count:,}")
        print()

        # Quick sanity sample.
        print("Sample - 3 random user-days with key metrics:")
        sample = pd.read_sql(
            "SELECT user_id, date, steps, sleep_min, resting_hr, "
            "       hrv_rmssd, stress_score, mood_tired "
            "FROM daily_summary "
            "WHERE steps IS NOT NULL "
            "ORDER BY RANDOM() LIMIT 3",
            conn,
        )
        print(sample.to_string(index=False))
    finally:
        conn.close()

    print("\nETL complete.")


if __name__ == "__main__":
    main()
