"""
PHAI - rule-based natural-language sensor narratives.

Turns a user-day's daily_summary row into one short human-readable sentence
using the user's own historical baseline (mean / std / deciles per metric).

Two ways to use this file:

    # 1. As an importable module (agent / onboarding flow):
    from nl_translator import compute_user_baselines, generate_narrative
    baselines = compute_user_baselines(user_daily_df)
    narrative = generate_narrative(row_dict, baselines[user_id])

    # 2. As a script - populate the nl_narratives table:
    python nl_translator.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "phai.db"

# Metrics for which we cache per-user baselines.
BASELINE_METRICS = (
    "sleep_min",
    "sleep_efficiency",
    "steps",
    "very_active_min",
    "resting_hr",
    "hrv_rmssd",
    "stress_score",
    "sedentary_min",
)


# ----- baselines ------------------------------------------------------------

def compute_user_baselines(
    df: pd.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    For each user, compute mean / std / deciles for our key metrics.

    Returns: baselines[user_id][metric] = {mean, std, p10, p25, p75, p90}.
    """
    baselines: dict[str, dict[str, dict[str, float]]] = {}
    for user_id, g in df.groupby("user_id"):
        user_b: dict[str, dict[str, float]] = {}
        for m in BASELINE_METRICS:
            if m not in g.columns:
                continue
            s = g[m].dropna()
            if len(s) < 5:  # too thin to baseline
                continue
            std = float(s.std()) or 1.0  # avoid div-by-zero downstream
            user_b[m] = {
                "mean": float(s.mean()),
                "std": std,
                "p10": float(s.quantile(0.10)),
                "p25": float(s.quantile(0.25)),
                "p75": float(s.quantile(0.75)),
                "p90": float(s.quantile(0.90)),
            }
        baselines[str(user_id)] = user_b
    return baselines


# ----- rules ----------------------------------------------------------------

def _has(row: dict, key: str) -> bool:
    return key in row and not pd.isna(row.get(key))


def observe_day(row: dict, baseline: dict) -> list[tuple[int, str]]:
    """
    Apply rule families. Each fires (priority, sentence) tuples. Higher
    priority = more salient.
    """
    obs: list[tuple[int, str]] = []

    # ----- sleep duration -----
    sm = baseline.get("sleep_min")
    if sm and _has(row, "sleep_min"):
        v = row["sleep_min"]
        hours = v / 60
        if v < sm["p10"]:
            obs.append((4, f"a notably short night of sleep ({hours:.1f} h, in your bottom decile)"))
        elif v < sm["mean"] - sm["std"]:
            obs.append((3, f"shorter sleep than usual ({hours:.1f} h)"))
        elif v > sm["p90"]:
            obs.append((2, f"a long night of sleep ({hours:.1f} h)"))

    # ----- sleep efficiency -----
    se = baseline.get("sleep_efficiency")
    if se and _has(row, "sleep_efficiency"):
        if row["sleep_efficiency"] < se["p25"]:
            obs.append((2, f"sleep efficiency was lower than usual ({int(row['sleep_efficiency'])}%)"))

    # ----- steps / activity -----
    st = baseline.get("steps")
    if st and _has(row, "steps"):
        v = row["steps"]
        if v > st["p90"]:
            obs.append((3, f"a high-activity day ({int(v):,} steps, in your top decile)"))
        elif v < st["p10"]:
            obs.append((3, f"a low-activity day ({int(v):,} steps, in your bottom decile)"))
        elif v > st["mean"] + st["std"]:
            obs.append((2, f"more steps than usual ({int(v):,})"))
        elif v < st["mean"] - st["std"]:
            obs.append((2, f"fewer steps than usual ({int(v):,})"))

    # ----- HRV / recovery -----
    hr = baseline.get("hrv_rmssd")
    if hr and _has(row, "hrv_rmssd"):
        v = row["hrv_rmssd"]
        if v < hr["p10"]:
            obs.append((4, f"HRV in your bottom decile ({v:.0f} ms) - recovery looks low"))
        elif v < hr["p25"]:
            obs.append((3, "HRV below your usual range"))
        elif v > hr["p90"]:
            obs.append((2, f"HRV in your top decile ({v:.0f} ms) - recovery looks strong"))

    # ----- resting HR -----
    rh = baseline.get("resting_hr")
    if rh and _has(row, "resting_hr"):
        v = row["resting_hr"]
        if v > rh["mean"] + rh["std"]:
            obs.append((3, f"resting heart rate slightly elevated ({v:.0f} bpm)"))
        elif v < rh["mean"] - rh["std"]:
            obs.append((1, f"resting heart rate on the lower end ({v:.0f} bpm)"))

    # ----- stress -----
    ss = baseline.get("stress_score")
    if ss and _has(row, "stress_score"):
        v = row["stress_score"]
        if v > ss["p75"]:
            obs.append((3, f"stress score elevated ({int(v)}/100)"))
        elif v < ss["p25"]:
            obs.append((1, f"stress score low ({int(v)}/100)"))

    # ----- mood self-report -----
    mood_words: list[str] = []
    if row.get("mood_tired") == 1:
        mood_words.append("tired")
    if row.get("mood_sad") == 1:
        mood_words.append("low")
    if row.get("mood_tense") == 1:
        mood_words.append("tense")
    if row.get("mood_happy") == 1:
        mood_words.append("happy")
    if row.get("mood_rested") == 1:
        mood_words.append("rested")
    if row.get("mood_alert") == 1:
        mood_words.append("alert")
    if mood_words:
        obs.append((3, f"you logged feeling {', '.join(mood_words)}"))

    return obs


def craft_narrative(observations: list[tuple[int, str]]) -> str:
    """Take top 2 observations by priority, join into 1-2 sentence narrative."""
    if not observations:
        return "A typical day across your tracked metrics."
    observations.sort(key=lambda x: -x[0])
    top = [s for _, s in observations[:2]]
    # Capitalise the first letter of each sentence-fragment.
    top = [s[0].upper() + s[1:] for s in top]
    return ". ".join(top) + "."


def generate_narrative(row: dict[str, Any], baseline: dict) -> str:
    """End-to-end public helper - observations + crafted narrative."""
    return craft_narrative(observe_day(row, baseline))


# ----- ETL main -------------------------------------------------------------

def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"phai.db not found at {DB_PATH}.")

    print("Loading daily summaries...")
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM daily_summary", conn)
        print(f"  {len(df):,} rows for {df['user_id'].nunique()} users")

        print("\nComputing per-user baselines...")
        baselines = compute_user_baselines(df)
        n_with_baseline = sum(1 for b in baselines.values() if b)
        print(f"  baselines computed for {n_with_baseline} users")

        print("\nGenerating narratives...")
        narratives: list[tuple[str, str, str, str]] = []
        for _, row in df.iterrows():
            uid = str(row["user_id"])
            text = generate_narrative(row.to_dict(), baselines.get(uid, {}))
            narratives.append((uid, str(row["date"]), text, "v1"))

            if len(narratives) % 20000 == 0:
                print(f"  {len(narratives):,}/{len(df):,}")

        print("\nWriting to nl_narratives...")
        conn.execute("DELETE FROM nl_narratives")
        conn.executemany(
            "INSERT INTO nl_narratives (user_id, date, text, generator_version) "
            "VALUES (?, ?, ?, ?)",
            narratives,
        )
        conn.commit()

        n_total = conn.execute("SELECT COUNT(*) FROM nl_narratives").fetchone()[0]
        print(f"  inserted {n_total:,} narratives")
        print()

        # Distribution: how many "typical day" vs interesting?
        typical_count = conn.execute(
            "SELECT COUNT(*) FROM nl_narratives WHERE text LIKE 'A typical day%'"
        ).fetchone()[0]
        print(f"  {typical_count:,} 'typical day' fallbacks "
              f"({typical_count / n_total * 100:.1f}%)")
        print()

        # Sample
        print("Sample narratives (5 random non-typical):")
        sample = pd.read_sql(
            "SELECT user_id, date, text FROM nl_narratives "
            "WHERE text NOT LIKE 'A typical day%' "
            "ORDER BY RANDOM() LIMIT 5",
            conn,
        )
        for _, r in sample.iterrows():
            print(f"  [{r['date']}] {r['text']}")
    finally:
        conn.close()

    print("\nNarrative generation complete.")


if __name__ == "__main__":
    main()
