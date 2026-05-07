"""
PHAI - initialise the SQLite database from schema.sql.

Run from the project root:
    python db/init_db.py

Creates phai.db in the project root if it doesn't exist, applies the schema,
prints the resulting table list. Safe to re-run - schema uses CREATE TABLE
IF NOT EXISTS, so existing data is preserved.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Project root is the parent of this file's parent (db/init_db.py -> project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "phai.db"
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"


def main() -> None:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema.sql not found at {SCHEMA_PATH}")

    print(f"Schema:   {SCHEMA_PATH}")
    print(f"Database: {DB_PATH}")
    if DB_PATH.exists():
        size_kb = DB_PATH.stat().st_size / 1024
        print(f"          (already exists, {size_kb:.1f} KB - applying schema idempotently)")
    print()

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(schema_sql)
        conn.commit()

        # Verify by listing tables.
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        print(f"Tables in {DB_PATH.name}:")
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:<20s}  ({count} rows)")
        print()
        print(f"Database ready. {len(tables)} tables.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
