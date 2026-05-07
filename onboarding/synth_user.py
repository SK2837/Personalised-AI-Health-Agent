"""
PHAI - on-the-fly user synthesis from a short questionnaire.

`create_user_from_questionnaire(profile)` inserts a new user with:
  - source='onboarded', synthetic=1
  - 30 days of plausible daily_summary rows derived from the profile
  - 10-SNP genotypes (mostly HWE, biased for self-reported chronotype + caffeine)
  - NL narratives generated for each day

Returns the new user_id (stable string of the form 'you_YYYYMMDD_HHMMSS').
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "phai.db"


# ---------------------------------------------------------------------------
# Profile -> baseline metric mapping (the "physiological priors")
# ---------------------------------------------------------------------------

BASE_STEPS = {"sedentary": 4500, "moderate": 7500, "active": 10500, "very_active": 13500}
BASE_SLEEP_MIN = {"poor": 360, "fair": 420, "good": 480}
BASE_SLEEP_EFF = {"poor": 80, "fair": 90, "good": 95}
BASE_RESTING_HR = {"sedentary": 72, "moderate": 65, "active": 60, "very_active": 56}
BASE_STRESS = {"low": 35, "moderate": 55, "high": 72}
EXERCISE_HRV_FACTOR = {"sedentary": 0.85, "moderate": 1.0, "active": 1.15, "very_active": 1.3}
BASE_HRV_BASELINE = 35.0


# ---------------------------------------------------------------------------
# Daily synthesis
# ---------------------------------------------------------------------------

def _generate_daily(profile: dict, days: int, rng: np.random.Generator) -> list[dict]:
    """30 plausible daily_summary rows derived from the profile."""
    base_steps = BASE_STEPS[profile["exercise_level"]]
    base_sleep = BASE_SLEEP_MIN[profile["sleep_quality"]]
    base_eff = BASE_SLEEP_EFF[profile["sleep_quality"]]
    base_rhr = BASE_RESTING_HR[profile["exercise_level"]]
    base_stress = BASE_STRESS[profile["stress_level"]]

    age_factor = max(0.5, 1.5 - profile["age"] / 80.0)
    base_hrv = (
        BASE_HRV_BASELINE * age_factor * EXERCISE_HRV_FACTOR[profile["exercise_level"]]
    )

    p_tired = {"poor": 0.55, "fair": 0.30, "good": 0.15}[profile["sleep_quality"]]
    p_happy = {"poor": 0.20, "fair": 0.45, "good": 0.65}[profile["sleep_quality"]]
    p_tense = {"low": 0.05, "moderate": 0.20, "high": 0.45}[profile["stress_level"]]

    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)

    rows: list[dict] = []
    for d in range(days):
        date = start_date + timedelta(days=d)

        steps = max(0, int(rng.normal(base_steps, base_steps * 0.20)))
        sleep_min = max(180, int(rng.normal(base_sleep, 50)))
        sleep_eff = float(max(50, min(100, rng.normal(base_eff, 4))))
        rhr = max(40, int(rng.normal(base_rhr, 3)))
        hrv = float(max(5, rng.normal(base_hrv, base_hrv * 0.15)))
        stress = float(max(0, min(100, rng.normal(base_stress, 12))))

        very_active = max(0, int(steps * 0.001 + rng.normal(0, 4)))
        moderately_active = max(0, int(rng.normal(15, 10)))
        lightly_active = max(0, int(rng.normal(150, 30)))
        sedentary = max(0, 1440 - lightly_active - moderately_active - very_active)

        bpm = float(rhr + rng.normal(15, 4))
        calories = int(1500 + steps * 0.04 + rng.normal(0, 100))

        # Mood self-reports happen on ~35% of days (matches LifeSnaps coverage).
        mood_logged = rng.random() < 0.35

        def _mood(p_yes: float):
            return (1 if rng.random() < p_yes else 0) if mood_logged else None

        rows.append({
            "date": date.isoformat(),
            "steps": steps,
            "distance": float(steps * 0.7),
            "calories": calories,
            "lightly_active_min": lightly_active,
            "moderately_active_min": moderately_active,
            "very_active_min": very_active,
            "sedentary_min": sedentary,
            "bpm_avg": bpm,
            "resting_hr": float(rhr),
            "hrv_rmssd": hrv,
            "nrem_hr": float(max(35, rhr - 5)),
            "breathing_rate": float(rng.normal(15, 1)),
            "sleep_min": sleep_min,
            "sleep_efficiency": sleep_eff,
            "minutes_to_fall_asleep": max(0, int(rng.normal(8, 4))),
            "minutes_awake": max(0, int(rng.normal(50, 15))),
            "stress_score": stress,
            "sleep_score_pct": float(sleep_eff / 100),
            "exertion_score_pct": float(min(1.0, very_active / 30)),
            "responsiveness_score_pct": float(min(1.0, max(0.0, rng.normal(0.65, 0.1)))),
            "nightly_temp": float(rng.normal(33.5, 0.4)),
            "mood_alert":   _mood(1 - p_tired),
            "mood_happy":   _mood(p_happy),
            "mood_neutral": _mood(0.30),
            "mood_rested":  _mood(1 - p_tired),
            "mood_sad":     _mood(p_tired * 0.4),
            "mood_tense":   _mood(p_tense),
            "mood_tired":   _mood(p_tired),
        })
    return rows


# ---------------------------------------------------------------------------
# Genotype synthesis (HWE + biased for chronotype/caffeine self-report)
# ---------------------------------------------------------------------------

def _generate_genotypes(profile: dict, rng: np.random.Generator) -> list[tuple[str, str]]:
    """Per-rsid genotypes. Most via HWE, two biased by self-report."""
    from etl.load_genes import sample_genotype
    from reference.snp_panel import PANEL

    variants: list[tuple[str, str]] = []
    for snp in PANEL:
        rsid = snp["rsid"]

        # CLOCK rs1801260 - bias by chronotype self-report.
        if rsid == "rs1801260":
            ct = profile.get("chronotype")
            if ct == "late":
                genotype = str(rng.choice(["TT", "CT", "CC"], p=[0.15, 0.35, 0.50]))
            elif ct == "early":
                genotype = str(rng.choice(["TT", "CT", "CC"], p=[0.65, 0.30, 0.05]))
            else:
                genotype = sample_genotype(snp, rng)

        # CYP1A2 rs762551 - bias by caffeine sensitivity self-report.
        elif rsid == "rs762551":
            cs = profile.get("caffeine_sensitivity")
            if cs == "sensitive":
                genotype = str(rng.choice(["AA", "AC", "CC"], p=[0.15, 0.35, 0.50]))
            elif cs == "tolerant":
                genotype = str(rng.choice(["AA", "AC", "CC"], p=[0.65, 0.30, 0.05]))
            else:
                genotype = sample_genotype(snp, rng)

        else:
            genotype = sample_genotype(snp, rng)

        variants.append((rsid, genotype))
    return variants


# ---------------------------------------------------------------------------
# DB insert + narrative generation
# ---------------------------------------------------------------------------

def create_user_from_questionnaire(profile: dict, days: int = 30) -> str:
    """Insert a new onboarded user. Returns the new user_id."""
    new_uid = f"you_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    seed = abs(hash(str(sorted(profile.items())))) % (2**31)
    rng = np.random.default_rng(seed)

    daily_rows = _generate_daily(profile, days, rng)
    variants = _generate_genotypes(profile, rng)

    with sqlite3.connect(DB_PATH) as conn:
        # users
        conn.execute(
            "INSERT INTO users (user_id, source, synthetic, age, gender, bmi) "
            "VALUES (?, 'onboarded', 1, ?, ?, ?)",
            (new_uid, int(profile["age"]), profile["sex"], float(profile["bmi"])),
        )
        # daily_summary - bulk insert
        cols = list(daily_rows[0].keys())
        col_str = ", ".join(["user_id"] + cols)
        ph_str = ", ".join("?" * (len(cols) + 1))
        sql = f"INSERT INTO daily_summary ({col_str}) VALUES ({ph_str})"
        conn.executemany(
            sql,
            [(new_uid, *[r[c] for c in cols]) for r in daily_rows],
        )
        # user_variants
        conn.executemany(
            "INSERT INTO user_variants (user_id, rsid, genotype, source) "
            "VALUES (?, ?, ?, 'synthetic')",
            [(new_uid, rsid, gen) for rsid, gen in variants],
        )
        conn.commit()

    # Generate narratives using the same rule engine the rest of the app uses.
    from nl_translator import compute_user_baselines, generate_narrative

    df = pd.DataFrame(daily_rows)
    df["user_id"] = new_uid
    baselines = compute_user_baselines(df)

    narratives: list[tuple[str, str, str, str]] = []
    for _, row in df.iterrows():
        text = generate_narrative(row.to_dict(), baselines.get(new_uid, {}))
        narratives.append((new_uid, row["date"], text, "v1"))

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            "INSERT INTO nl_narratives (user_id, date, text, generator_version) "
            "VALUES (?, ?, ?, ?)",
            narratives,
        )
        conn.commit()

    return new_uid
