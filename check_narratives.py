"""Quick diagnostic - how many users have narratives, by source."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "phai.db"

with sqlite3.connect(DB_PATH) as conn:
    print("=== Narrative coverage by user source ===\n")
    rows = conn.execute(
        """
        SELECT u.source,
               COUNT(DISTINCT u.user_id)         AS total_users,
               COUNT(DISTINCT n.user_id)         AS users_with_narratives,
               COUNT(n.user_id)                  AS total_narratives
        FROM users u
        LEFT JOIN nl_narratives n USING (user_id)
        GROUP BY u.source
        """
    ).fetchall()
    print(f"{'source':<12} {'users':>8} {'with narr':>12} {'total narr':>12}")
    print("-" * 48)
    for source, total, with_narr, n_narr in rows:
        print(f"{source:<12} {total:>8} {with_narr:>12} {n_narr:>12}")
    print()

    # Distribution of "typical day" fallbacks
    n_total = conn.execute("SELECT COUNT(*) FROM nl_narratives").fetchone()[0]
    n_typical = conn.execute(
        "SELECT COUNT(*) FROM nl_narratives WHERE text LIKE 'A typical day%'"
    ).fetchone()[0]
    pct = (n_typical / n_total * 100) if n_total else 0
    print(f"Generic 'typical day' narratives: {n_typical:,}/{n_total:,} ({pct:.1f}%)")
