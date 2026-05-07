"""
PHAI ETL - load gene reference catalogue + sample per-user genotypes.

Run from the project root:
    python etl/load_genes.py

What this does:
  1. Reads the curated 10-SNP panel from reference/snp_panel.py.
  2. Inserts each SNP's plain-English summary into the snp_reference table.
  3. For every existing user, samples a genotype for each of the 10 SNPs
     under Hardy-Weinberg equilibrium using the published minor allele
     frequency. Inserts into user_variants.

Idempotent: clears prior synthetic genotypes and the snp_reference table
before re-inserting. Re-run any time the panel changes.

Determinism: uses a fixed RNG seed so the same user gets the same
genotypes on every re-run (important for reproducible demos).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make `from reference.snp_panel import ...` work when run from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reference.snp_panel import PANEL  # noqa: E402

DB_PATH = PROJECT_ROOT / "phai.db"
RANDOM_SEED = 42


# ----- Hardy-Weinberg sampling ----------------------------------------------

def hwe_probs(maf: float) -> tuple[float, float, float]:
    """Hardy-Weinberg probabilities for (major-major, het, minor-minor)."""
    q = maf
    p = 1.0 - q
    return (p * p, 2.0 * p * q, q * q)


def sample_genotype(snp: dict, rng: np.random.Generator) -> str:
    """Sample one genotype for the given SNP under HWE."""
    major, minor = snp["alleles"]
    probs = hwe_probs(snp["minor_allele_freq"])
    choice = rng.choice(3, p=probs)
    if choice == 0:
        return major + major
    if choice == 1:
        # Always present heterozygous as alphabetical for consistency.
        return "".join(sorted([major, minor]))
    return minor + minor


# ----- ETL ------------------------------------------------------------------

def load_snp_reference(conn: sqlite3.Connection) -> int:
    """Insert each SNP's text summary into the snp_reference table."""
    conn.execute("DELETE FROM snp_reference")
    rows = [
        (
            snp["rsid"],
            snp["gene"],
            snp["trait_summary"],
            snp["lifestyle_implications"],
            snp.get("clinvar_significance"),
            snp.get("citation_url"),
        )
        for snp in PANEL
    ]
    conn.executemany(
        "INSERT INTO snp_reference "
        "(rsid, gene, trait_summary, lifestyle_implications, "
        " clinvar_significance, citation_url) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def generate_user_variants(conn: sqlite3.Connection) -> int:
    """Sample a genotype per (user, SNP) and insert into user_variants."""
    user_ids = [row[0] for row in conn.execute("SELECT user_id FROM users")]
    if not user_ids:
        raise RuntimeError(
            "No users in the database. Run etl/load_lifesnaps.py first."
        )

    conn.execute("DELETE FROM user_variants")

    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[tuple[str, str, str, str]] = []
    for user_id in user_ids:
        for snp in PANEL:
            rows.append((user_id, snp["rsid"], sample_genotype(snp, rng), "synthetic"))

    conn.executemany(
        "INSERT INTO user_variants (user_id, rsid, genotype, source) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


# ----- main -----------------------------------------------------------------

def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"phai.db not found at {DB_PATH}. Run db/init_db.py and "
            f"etl/load_lifesnaps.py first."
        )

    conn = sqlite3.connect(DB_PATH)
    try:
        n_snps = load_snp_reference(conn)
        print(f"snp_reference: {n_snps} SNPs inserted.")

        n_variants = generate_user_variants(conn)
        n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        print(
            f"user_variants: {n_variants:,} rows "
            f"({n_users} users x {n_snps} SNPs)."
        )
        print()

        # Sanity: show one user's full panel.
        sample_user = conn.execute(
            "SELECT user_id FROM users ORDER BY user_id LIMIT 1"
        ).fetchone()[0]
        print(f"Sample - genotypes for user {sample_user}:")
        df = pd.read_sql(
            "SELECT v.rsid, r.gene, v.genotype "
            "FROM user_variants v "
            "JOIN snp_reference r ON v.rsid = r.rsid "
            "WHERE v.user_id = ? "
            "ORDER BY r.gene",
            conn,
            params=(sample_user,),
        )
        print(df.to_string(index=False))
        print()

        # Sanity: cohort-level genotype distribution for one SNP, vs HWE expectation.
        rsid = PANEL[0]["rsid"]
        gene = PANEL[0]["gene"]
        maf = PANEL[0]["minor_allele_freq"]
        expected = hwe_probs(maf)
        print(
            f"Cohort distribution for {gene} ({rsid}, MAF={maf:.2f}) "
            f"vs HWE expectation:"
        )
        df = pd.read_sql(
            "SELECT genotype, COUNT(*) AS n "
            "FROM user_variants WHERE rsid = ? GROUP BY genotype "
            "ORDER BY genotype",
            conn,
            params=(rsid,),
        )
        df["observed_pct"] = (df["n"] / df["n"].sum() * 100).round(1)
        print(df.to_string(index=False))
        print(
            f"  expected ~ {expected[0] * 100:.0f}% / "
            f"{expected[1] * 100:.0f}% / {expected[2] * 100:.0f}%"
        )

    finally:
        conn.close()

    print("\nGene ETL complete.")


if __name__ == "__main__":
    main()
