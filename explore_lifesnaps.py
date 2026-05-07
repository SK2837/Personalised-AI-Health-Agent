"""
PHAI - one-shot exploration of the LifeSnaps daily CSV.

Run from the project root:
    python explore_lifesnaps.py

Prints structure, scale, key columns, and missing-value stats so we can
design the SQLite schema with eyes open. Throwaway script - safe to delete
once Step 6 is done.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_PATH = Path("data/raw/lifesnaps/daily_fitbit_sema_df_unprocessed.csv")

# Wider terminal output so columns don't wrap awkwardly.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)


def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: file not found at {CSV_PATH}")
        print("Make sure the LifeSnaps CSVs are in data/raw/lifesnaps/")
        return

    size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
    print(f"File:      {CSV_PATH}")
    print(f"File size: {size_mb:.1f} MB")
    print()

    print("Loading... (this can take 5-15s for big files)")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print()

    # --- shape ---
    print("=== SHAPE ===")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print()

    # --- all columns, dtype, % missing ---
    print("=== COLUMNS (name, dtype, % missing) ===")
    for col in df.columns:
        dtype = str(df[col].dtype)
        pct_null = df[col].isna().mean() * 100
        print(f"  {col:<45s}  {dtype:<12s}  {pct_null:5.1f}% missing")
    print()

    # --- users ---
    id_candidates = [
        c for c in df.columns
        if c.lower() in {"id", "user_id", "userid", "participant_id"}
    ]
    if id_candidates:
        id_col = id_candidates[0]
        per_user = df.groupby(id_col).size()
        print(f"=== USERS (column: {id_col}) ===")
        print(f"Unique users:   {df[id_col].nunique()}")
        print(f"Rows per user:  min={per_user.min()}, "
              f"median={int(per_user.median())}, max={per_user.max()}")
        print()

    # --- date range ---
    date_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["date", "day", "timestamp"])
    ]
    if date_candidates:
        date_col = date_candidates[0]
        try:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            print(f"=== TIME RANGE (column: {date_col}) ===")
            print(f"From: {dates.min()}")
            print(f"To:   {dates.max()}")
            if pd.notna(dates.min()) and pd.notna(dates.max()):
                span = (dates.max() - dates.min()).days
                print(f"Span: {span} days")
            print()
        except Exception as e:
            print(f"Could not parse {date_col} as date: {e}")
            print()

    # --- first row, transposed (so we see all columns) ---
    print("=== FIRST ROW (transposed) ===")
    print(df.head(1).T)
    print()

    # --- numeric summary ---
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        print("=== NUMERIC COLUMNS - quick stats ===")
        summary = numeric_df.describe(percentiles=[0.25, 0.5, 0.75]).T
        cols_to_show = ["count", "mean", "std", "min", "50%", "max"]
        print(summary[cols_to_show].round(2))
        print()

    print("Done.")


if __name__ == "__main__":
    main()
