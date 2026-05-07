"""
PHAI - tool implementations available to agents.

Each tool is a plain Python function that:
  - takes JSON-friendly arguments
  - returns a JSON-friendly dict (no DataFrames, no numpy types)
  - never raises - returns {"error": "..."} on failure

The agent loop (agents.base.run_agent_loop) calls these by name based on
what the LLM decides to invoke.

Metric names passed in by the LLM are validated against ALLOWED_METRICS
to prevent SQL injection or accidental column access.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "phai.db"

# Whitelist of column names the LLM may pass as a `metric` argument.
ALLOWED_METRICS: set[str] = {
    "steps", "distance", "calories",
    "lightly_active_min", "moderately_active_min", "very_active_min", "sedentary_min",
    "bpm_avg", "resting_hr", "hrv_rmssd", "nrem_hr", "breathing_rate",
    "sleep_min", "sleep_efficiency", "minutes_to_fall_asleep", "minutes_awake",
    "stress_score", "sleep_score_pct", "exertion_score_pct",
    "responsiveness_score_pct", "nightly_temp",
    "mood_alert", "mood_happy", "mood_neutral", "mood_rested",
    "mood_sad", "mood_tense", "mood_tired",
}


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _check_metric(name: str) -> str | None:
    """Return None if valid, else an error string."""
    if name not in ALLOWED_METRICS:
        return f"Unknown metric '{name}'. Allowed: {sorted(ALLOWED_METRICS)}"
    return None


def _to_native(v):
    """Convert numpy/pandas scalars to plain Python for clean JSON."""
    if v is None:
        return None
    if pd.isna(v):
        return None
    try:
        return v.item()  # numpy scalar
    except AttributeError:
        return v


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def get_user_profile(user_id: str) -> dict:
    """Return the user's basic profile and gene panel summary."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT user_id, age, gender, bmi, source, synthetic "
            "FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return {"error": f"No user found with id '{user_id}'"}
        n_variants = conn.execute(
            "SELECT COUNT(*) FROM user_variants WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        n_days = conn.execute(
            "SELECT COUNT(*) FROM daily_summary WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
    return {
        "user_id": row[0],
        "age": row[1],
        "gender": row[2],
        "bmi": round(row[3], 1) if row[3] is not None else None,
        "source": row[4],
        "is_synthetic": bool(row[5]),
        "days_of_data": n_days,
        "gene_variants_known": n_variants,
    }


def get_recent_summary(user_id: str, days: int = 30) -> dict:
    """
    Return averages for the user's most recent `days` days, plus the most
    recent narratives. Compact - intended to fit in LLM context easily.
    """
    days = max(1, min(int(days), 90))
    with _conn() as conn:
        max_date = conn.execute(
            "SELECT MAX(date) FROM daily_summary WHERE user_id = ?", (user_id,)
        ).fetchone()[0]
        if not max_date:
            return {"error": f"No daily data for user '{user_id}'"}
        df = pd.read_sql(
            "SELECT * FROM daily_summary WHERE user_id = ? "
            "AND date <= ? ORDER BY date DESC LIMIT ?",
            conn, params=(user_id, max_date, days),
        )
        narr = pd.read_sql(
            "SELECT date, text FROM nl_narratives "
            "WHERE user_id = ? AND date <= ? ORDER BY date DESC LIMIT 7",
            conn, params=(user_id, max_date),
        )

    if df.empty:
        return {"error": f"No daily data for user '{user_id}'"}

    # Per-metric averages, only for whitelisted columns with any data.
    averages = {}
    for col in ALLOWED_METRICS:
        if col in df.columns and df[col].notna().any():
            averages[col] = round(float(df[col].mean()), 2)

    # Coverage: how many of the recent days have data for each headline metric.
    coverage = {
        m: int(df[m].notna().sum())
        for m in ("sleep_min", "steps", "hrv_rmssd", "stress_score", "resting_hr")
        if m in df.columns
    }

    return {
        "user_id": user_id,
        "date_range": {"from": str(df["date"].min()), "to": str(df["date"].max())},
        "n_days_returned": len(df),
        "averages": averages,
        "data_coverage_days": coverage,
        "recent_narratives": narr.to_dict("records"),
    }


def compare_to_population(user_id: str, metric: str) -> dict:
    """
    Compare the user's mean value for a metric against the cohort
    distribution (one mean per user across the cohort).
    """
    err = _check_metric(metric)
    if err:
        return {"error": err}

    with _conn() as conn:
        user_mean = conn.execute(
            f"SELECT AVG({metric}) FROM daily_summary WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        if user_mean is None:
            return {"error": f"No data for {metric} for user '{user_id}'"}
        cohort_means = pd.read_sql(
            f"SELECT AVG({metric}) AS m FROM daily_summary "
            "GROUP BY user_id HAVING m IS NOT NULL",
            conn,
        )["m"]

    if cohort_means.empty:
        return {"error": "No cohort data available for this metric."}

    pct = float((cohort_means < user_mean).mean() * 100)
    return {
        "user_id": user_id,
        "metric": metric,
        "user_average": round(float(user_mean), 2),
        "cohort_size": int(len(cohort_means)),
        "cohort_median": round(float(cohort_means.median()), 2),
        "cohort_p25": round(float(cohort_means.quantile(0.25)), 2),
        "cohort_p75": round(float(cohort_means.quantile(0.75)), 2),
        "user_percentile_in_cohort": round(pct, 1),
    }


def compute_correlation(user_id: str, metric_a: str, metric_b: str) -> dict:
    """
    Pearson correlation between two metrics for one user, using days where
    both metrics are non-null.
    """
    err_a, err_b = _check_metric(metric_a), _check_metric(metric_b)
    if err_a:
        return {"error": err_a}
    if err_b:
        return {"error": err_b}
    if metric_a == metric_b:
        return {"error": "metric_a and metric_b must be different."}

    with _conn() as conn:
        df = pd.read_sql(
            f"SELECT {metric_a}, {metric_b} FROM daily_summary "
            f"WHERE user_id = ? AND {metric_a} IS NOT NULL "
            f"AND {metric_b} IS NOT NULL",
            conn, params=(user_id,),
        )

    if len(df) < 5:
        return {
            "user_id": user_id,
            "metric_a": metric_a,
            "metric_b": metric_b,
            "n_overlapping_days": int(len(df)),
            "warning": "Fewer than 5 overlapping days - correlation not meaningful.",
        }

    corr = float(df[metric_a].corr(df[metric_b]))
    return {
        "user_id": user_id,
        "metric_a": metric_a,
        "metric_b": metric_b,
        "n_overlapping_days": int(len(df)),
        "pearson_r": round(corr, 3),
        "interpretation": _interpret_corr(corr),
    }


def _interpret_corr(r: float) -> str:
    a = abs(r)
    direction = "positive" if r > 0 else "negative"
    if a < 0.10:
        return "no meaningful correlation"
    if a < 0.30:
        return f"weak {direction} correlation"
    if a < 0.50:
        return f"moderate {direction} correlation"
    return f"strong {direction} correlation"


# ---------------------------------------------------------------------------
# Domain Expert tools - genes + knowledge-base RAG
# ---------------------------------------------------------------------------

def get_user_genes(user_id: str) -> dict:
    """
    Return the user's full gene panel: each variant with the user's specific
    genotype, what that genotype means, the trait summary, and the lifestyle
    implications.
    """
    # Local import keeps panel module optional at import time of tools.py.
    from reference.snp_panel import PANEL_BY_RSID

    with _conn() as conn:
        rows = conn.execute(
            "SELECT v.rsid, v.genotype, r.gene, r.trait_summary, "
            "       r.lifestyle_implications, r.clinvar_significance, r.citation_url "
            "FROM user_variants v "
            "JOIN snp_reference r USING (rsid) "
            "WHERE v.user_id = ? "
            "ORDER BY r.gene",
            (user_id,),
        ).fetchall()

    if not rows:
        return {"error": f"No gene data for user '{user_id}'."}

    variants = []
    for rsid, genotype, gene, trait, lifestyle, sig, url in rows:
        snp_info = PANEL_BY_RSID.get(rsid, {})
        meanings = snp_info.get("genotype_meanings", {})
        # Heterozygous genotypes may be stored as 'CT' or 'TC'. Try both
        # original and reversed before giving up.
        interpretation = (
            meanings.get(genotype)
            or meanings.get(genotype[::-1])
            or "Genotype interpretation not in panel."
        )
        variants.append({
            "rsid": rsid,
            "gene": gene,
            "genotype": genotype,
            "interpretation": interpretation,
            "trait_summary": trait,
            "lifestyle_implications": lifestyle,
            "clinvar_significance": sig,
            "citation_url": url,
        })

    return {
        "user_id": user_id,
        "n_variants": len(variants),
        "variants": variants,
    }


def lookup_snp(rsid: str) -> dict:
    """
    Look up a single SNP by rsid (e.g. 'rs762551'). Returns the curated
    summary, lifestyle implications, all genotype meanings, and citation URL.
    """
    from reference.snp_panel import PANEL_BY_RSID

    with _conn() as conn:
        row = conn.execute(
            "SELECT rsid, gene, trait_summary, lifestyle_implications, "
            "       clinvar_significance, citation_url "
            "FROM snp_reference WHERE rsid = ?",
            (rsid,),
        ).fetchone()

    if not row:
        return {"error": f"SNP '{rsid}' not in our panel."}

    snp_info = PANEL_BY_RSID.get(rsid, {})
    return {
        "rsid": row[0],
        "gene": row[1],
        "trait_summary": row[2],
        "lifestyle_implications": row[3],
        "clinvar_significance": row[4],
        "citation_url": row[5],
        "genotype_meanings": snp_info.get("genotype_meanings", {}),
        "alleles": list(snp_info.get("alleles", ())),
        "minor_allele_freq": snp_info.get("minor_allele_freq"),
    }


# Module-level cache for the ChromaDB collection (loading the embedder is slow).
_kb_collection = None


def _get_kb_collection():
    global _kb_collection
    if _kb_collection is not None:
        return _kb_collection

    chroma_dir = PROJECT_ROOT / "chroma_db"
    if not chroma_dir.exists():
        raise FileNotFoundError(
            "Knowledge base not initialised. Run `python kb/ingest.py` first."
        )

    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=str(chroma_dir))
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    )
    _kb_collection = client.get_collection(
        name="phai_kb", embedding_function=embedding_fn
    )
    return _kb_collection


def kb_search(query: str, k: int = 3) -> dict:
    """
    Semantic search the curated knowledge base. Returns top-k snippets with
    text, topic, source, and URL for citation.
    """
    k = max(1, min(int(k), 8))
    try:
        collection = _get_kb_collection()
    except FileNotFoundError as e:
        return {"error": str(e)}

    results = collection.query(query_texts=[query], n_results=k)

    items = []
    for snippet_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append({
            "snippet_id": snippet_id,
            "topic": meta.get("topic", ""),
            "category": meta.get("category", ""),
            "text": doc,
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "distance": round(float(dist), 3),
        })

    return {"query": query, "n_results": len(items), "results": items}


# ---------------------------------------------------------------------------
# ML model tools - energy classifier
# ---------------------------------------------------------------------------

def predict_energy_tomorrow(user_id: str) -> dict:
    """
    Use the trained XGBoost energy classifier to predict whether the user's
    step count tomorrow will be above their personal median.
    Wraps models.energy_clf.predict() so the agent can call it as a tool.
    """
    from models.energy_clf import predict as _predict
    return _predict(user_id)


# ---------------------------------------------------------------------------
# Health Coach tools - plan submission
# ---------------------------------------------------------------------------

def _parse_jsonish(s):
    """Try multiple strategies to parse a JSON-like string. Returns None
    if all fail."""
    import ast

    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return None

    # Strategy 1: standard JSON.
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: ast.literal_eval (handles single quotes, Python literals).
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass

    # Strategy 3: trim to the outermost [ ... ] or { ... } and retry.
    for open_c, close_c in (("[", "]"), ("{", "}")):
        if open_c in s and close_c in s:
            sliced = s[s.index(open_c): s.rindex(close_c) + 1]
            try:
                return json.loads(sliced)
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                return ast.literal_eval(sliced)
            except (ValueError, SyntaxError):
                pass
    return None


def submit_plan(
    user_id: str,
    goal: str,
    why: str,
    steps,
    user_message: str,
    metrics_to_track=None,
    check_in_days=7,
) -> dict:
    """
    Validate a structured plan submitted by the Coach. Does NOT persist -
    persistence is handled by agents.health_coach.run() after the loop.

    Tolerant to small-model quirks:
    - Stringified JSON arrays/objects are coerced back to native types
      via multiple parsing strategies (json -> ast.literal_eval ->
      bracket-trim retry).
    - Too many steps are silently truncated to 8 (never errors out the
      submission). The Coach's "aim for 3-5" lives in the prompt only.
    """
    # ---- Coerce stringified JSON values back to native types ----
    if isinstance(steps, str):
        parsed = _parse_jsonish(steps)
        if parsed is None:
            return {
                "error": (
                    "could not parse steps. Send steps as a JSON array of "
                    "objects with 'title' and 'detail' keys."
                )
            }
        steps = parsed

    if isinstance(metrics_to_track, str):
        parsed = _parse_jsonish(metrics_to_track)
        metrics_to_track = parsed if isinstance(parsed, list) else []

    if isinstance(check_in_days, str):
        try:
            check_in_days = int(check_in_days)
        except (ValueError, TypeError):
            check_in_days = 7

    # ---- Per-step coercion ----
    coerced_steps = []
    for s in steps if isinstance(steps, list) else []:
        if isinstance(s, str):
            parsed = _parse_jsonish(s)
            if parsed is None:
                continue
            s = parsed
        if isinstance(s, dict):
            coerced_steps.append(s)
    steps = coerced_steps

    # Truncate silently rather than error - some small models over-produce.
    if len(steps) > 8:
        steps = steps[:8]

    # ---- Validate ----
    if not goal or not isinstance(goal, str):
        return {"error": "goal must be a non-empty string"}
    if not why or not isinstance(why, str):
        return {"error": "why must be a non-empty string"}
    if not user_message or not isinstance(user_message, str):
        return {"error": "user_message must be a non-empty string"}
    if not isinstance(steps, list) or not steps:
        return {"error": "steps must be a non-empty list of step objects"}

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return {"error": f"step {i} must be a dict"}
        if not step.get("title") or not step.get("detail"):
            return {"error": f"step {i} missing title or detail"}

    # Canonical, fully-coerced plan dict. The agent runner pulls this out
    # of the trace so persistence + UI rendering use the validated shape
    # (not the LLM's raw args, which may have stringified arrays).
    canonical_plan = {
        "goal": goal,
        "why": why,
        "steps": steps,
        "metrics_to_track": metrics_to_track or [],
        "check_in_days": int(check_in_days),
        "user_message": user_message,
    }

    return {
        "success": True,
        "user_id": user_id,
        "n_steps": len(steps),
        "metrics_to_track": metrics_to_track or [],
        "check_in_days": int(check_in_days),
        "plan": canonical_plan,
        "summary": (
            f"Plan accepted: '{goal}' "
            f"({len(steps)} steps, check-in in {check_in_days} days)."
        ),
    }
