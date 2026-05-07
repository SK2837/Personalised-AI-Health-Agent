"""
PHAI ETL - extend cohort from 71 real users to 1000 via stratified resampling.

Run from the project root:
    python etl/extend_cohort.py

For each of 929 synthetic users:
  1. Pick a random real LifeSnaps user as template.
  2. Copy their daily_summary trajectory.
  3. Add per-metric Gaussian noise (proportional to per-column mean).
  4. Occasionally flip mood self-reports (small probability).
  5. Clip values back into plausible physiological ranges.
  6. Generate fresh HWE genotypes from the same panel.

All synthetic users have source='synthetic' and synthetic=1, so the agent
can always tell them from real LifeSnaps participants.

Idempotent: clears any prior synthetic users before re-running.
Deterministic: fixed RNG seed -> same cohort every run.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from etl.load_genes import sample_genotype  # noqa: E402  reuse HWE sampler
from reference.snp_panel import PANEL  # noqa: E402

DB_PATH = PROJECT_ROOT / "phai.db"
TARGET_COHORT_SIZE = 1000
RANDOM_SEED = 42

# Per-metric relative Gaussian noise std (fraction of the column's per-user mean).
# Tuned roughly to within-person day-to-day variability seen in wearables.
NOISE_STD_PCT: dict[str, float] = {
    "steps": 0.15,
    "distance": 0.15,
    "calories": 0.10,
    "lightly_active_min": 0.15,
    "moderately_active_min": 0.20,
    "very_active_min": 0.25,
    "sedentary_min": 0.05,
    "bpm_avg": 0.05,
    "resting_hr": 0.04,
    "hrv_rmssd": 0.10,
    "nrem_hr": 0.05,
    "breathing_rate": 0.05,
    "sleep_min": 0.10,
    "sleep_efficiency": 0.03,
    "minutes_to_fall_asleep": 0.30,
    "minutes_awake": 0.20,
    "stress_score": 0.10,
    "sleep_score_pct": 0.05,
    "exertion_score_pct": 0.05,
    "responsiveness_score_pct": 0.05,
    "nightly_temp": 0.01,
}

MOOD_COLS = (
    "mood_alert", "mood_happy", "mood_neutral", "mood_rested",
    "mood_sad", "mood_tense", "mood_tired",
)
MOOD_FLIP_PROB = 0.05

# Plausible physiological clamps applied after noise.
CLAMP_RANGES: dict[str, tuple[float, float]] = {
    "steps": (0, 50000),
    "distance": (0, 60000),
    "calories": (500, 8000),
    "sleep_min": (0, 1200),
    "sleep_efficiency": (0, 100),
    "stress_score": (0, 100),
    "sleep_score_pct": (0, 1),
    "exertion_score_pct": (0, 1),
    "responsiveness_score_pct": (0, 1),
    "resting_hr": (35, 110),
    "bpm_avg": (40, 200),
    "hrv_rmssd": (5, 200),
    "nrem_hr": (35, 110),
    "breathing_rate": (8, 30),
    "nightly_temp": (30, 40),
    "lightly_active_min": (0, 600),
    "moderately_active_min": (0, 300),
    "very_active_min": (0, 300),
    "sedentary_min": (0, 1440),
    "minutes_to_fall_asleep": (0, 120),
    "minutes_awake": (0, 400),
}


def perturb_demographics(template: pd.Series, rng: np.random.Generator) -> dict:
    """Small perturbations on age and BMI; keep gender."""
    age = template["age"]
    if pd.notna(age):
        age = max(18, min(80, int(age) + int(rng.integers(-2, 3))))
    bmi = template["bmi"]
    if pd.notna(bmi):
        bmi = float(bmi) * (1.0 + float(rng.normal(0, 0.05)))
    return {"age": age, "gender": template["gender"], "bmi": bmi}


def add_noise_and_clamp(
    df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Add Gaussian noise to numeric metrics, flip moods, clamp ranges."""
    out = df.copy()

    # Noise on numeric metrics, scaled to each column's mean.
    for col, pct in NOISE_STD_PCT.items():
        if col not in out.columns:
            continue
        col_mean = out[col].mean()
        if pd.isna(col_mean) or col_mean == 0:
            continue
        std = abs(col_mean * pct)
        noise = rng.normal(0, std, size=len(out))
        mask = out[col].notna()
        out.loc[mask, col] = out.loc[mask, col] + noise[mask.values]

    # Random mood flips.
    for col in MOOD_COLS:
        if col not in out.columns:
            continue
        mask = out[col].notna()
        flips = rng.random(len(out)) < MOOD_FLIP_PROB
        flip_mask = mask & pd.Series(flips, index=out.index)
        out.loc[flip_mask, col] = 1 - out.loc[flip_mask, col]

    # Clamp to plausible ranges.
    for col, (lo, hi) in CLAMP_RANGES.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=lo, upper=hi)

    return out


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"phai.db not found at {DB_PATH}. Run prior ETL steps first."
        )

    conn = sqlite3.connect(DB_PATH)
    try:
        real_count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE source='lifesnaps'"
        ).fetchone()[0]
        if real_count == 0:
            raise RuntimeError(
                "No LifeSnaps users found. Run etl/load_lifesnaps.py first."
            )
        n_synthetic = TARGET_COHORT_SIZE - real_count
        if n_synthetic <= 0:
            print(f"Already have {real_count} >= {TARGET_COHORT_SIZE} users. Nothing to do.")
            return
        print(f"Real users: {real_count}. Generating {n_synthetic} synthetic users...")

        # Idempotent: drop any prior synthetic extensions.
        conn.execute(
            "DELETE FROM user_variants "
            "WHERE user_id IN (SELECT user_id FROM users WHERE source='synthetic')"
        )
        conn.execute(
            "DELETE FROM daily_summary "
            "WHERE user_id IN (SELECT user_id FROM users WHERE source='synthetic')"
        )
        conn.execute("DELETE FROM users WHERE source='synthetic'")
        conn.commit()

        # Pre-load real users + their full daily history once (fast).
        real_users = pd.read_sql(
            "SELECT user_id, age, gender, bmi FROM users WHERE source='lifesnaps'",
            conn,
        )
        all_real_daily = pd.read_sql(
            "SELECT * FROM daily_summary "
            "WHERE user_id IN (SELECT user_id FROM users WHERE source='lifesnaps')",
            conn,
        )
        daily_by_template = {
            uid: g.reset_index(drop=True)
            for uid, g in all_real_daily.groupby("user_id")
        }

        rng = np.random.default_rng(RANDOM_SEED)
        new_users_rows: list[dict] = []
        new_daily_frames: list[pd.DataFrame] = []
        new_variants: list[tuple[str, str, str, str]] = []

        t0 = time.time()
        for i in range(n_synthetic):
            template = real_users.iloc[int(rng.integers(0, len(real_users)))]
            template_uid = template["user_id"]
            new_uid = f"syn_{i:04d}"

            demo = perturb_demographics(template, rng)
            new_users_rows.append({
                "user_id": new_uid,
                "source": "synthetic",
                "synthetic": 1,
                "age": demo["age"],
                "gender": demo["gender"],
                "bmi": demo["bmi"],
            })

            template_daily = daily_by_template[template_uid]
            new_daily = add_noise_and_clamp(template_daily, rng)
            new_daily["user_id"] = new_uid
            new_daily_frames.append(new_daily)

            for snp in PANEL:
                new_variants.append(
                    (new_uid, snp["rsid"], sample_genotype(snp, rng), "synthetic")
                )

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{n_synthetic} users  ({elapsed:.1f}s)")

        # ---- bulk insert ----
        print("\nInserting...")
        users_df = pd.DataFrame(new_users_rows)
        users_df["age"] = users_df["age"].astype("Int64")
        daily_df = pd.concat(new_daily_frames, ignore_index=True)

        # Round mood columns to int (post-flip), preserve nulls.
        for col in MOOD_COLS:
            if col in daily_df.columns:
                daily_df[col] = daily_df[col].round().astype("Int64")

        users_df.to_sql("users", conn, if_exists="append", index=False)
        daily_df.to_sql("daily_summary", conn, if_exists="append", index=False)
        conn.executemany(
            "INSERT INTO user_variants (user_id, rsid, genotype, source) "
            "VALUES (?, ?, ?, ?)",
            new_variants,
        )
        conn.commit()

        # ---- verification ----
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_daily = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
        total_var = conn.execute("SELECT COUNT(*) FROM user_variants").fetchone()[0]
        print()
        print("Final cohort:")
        print(f"  users:         {total_users:,}")
        print(f"  daily_summary: {total_daily:,}")
        print(f"  user_variants: {total_var:,}")

        breakdown = pd.read_sql(
            "SELECT source, COUNT(*) AS n FROM users GROUP BY source ORDER BY source",
            conn,
        )
        print("\nUsers by source:")
        print(breakdown.to_string(index=False))

        # Quick sanity: average steps for real vs synthetic should be similar.
        sanity = pd.read_sql(
            """
            SELECT u.source,
                   AVG(d.steps) AS avg_steps,
                   AVG(d.sleep_min) AS avg_sleep_min,
                   AVG(d.resting_hr) AS avg_resting_hr
            FROM daily_summary d
            JOIN users u USING (user_id)
            GROUP BY u.source
            """,
            conn,
        )
        print("\nSanity - real vs synthetic averages should be close:")
        print(sanity.round(1).to_string(index=False))
    finally:
        conn.close()

    print("\nCohort extension complete.")


if __name__ == "__main__":
    main()
