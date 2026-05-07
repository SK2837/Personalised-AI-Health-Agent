"""
PHAI - Streamlit app.

Run from the project root:
    streamlit run ui/app.py

The app opens at http://localhost:8501. Saving this file auto-reloads
the browser.

This file is the entire UI - kept as one file for the demo. Tabs are
filled in incrementally across Steps 17b-17e.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Allow importing from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DB_PATH = PROJECT_ROOT / "phai.db"

# Auto-initialize DB on first run (needed on Streamlit Cloud where phai.db is not committed).
if not DB_PATH.exists():
    _schema = (PROJECT_ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    with sqlite3.connect(DB_PATH) as _conn:
        _conn.executescript(_schema)


# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PHAI - Personalised AI Health Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Reusable inline SVG / HTML snippets
# ---------------------------------------------------------------------------

HERO_HTML = """
<div style="
    padding: 1.6rem 1.9rem;
    background: linear-gradient(135deg, #dc2626 0%, #7f1d1d 55%, #171717 100%);
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 30px rgba(220, 38, 38, 0.35);
    border: 1px solid rgba(220, 38, 38, 0.4);
">
  <div style="display: flex; align-items: center; gap: 0.85rem;">
    <span style="font-size: 2.6rem;">🧬</span>
    <div>
      <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; line-height: 1;
                 letter-spacing: 0.02em; text-transform: uppercase;">PHAI</h1>
      <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem; opacity: 0.95;
                letter-spacing: 0.02em;">
        Personalised AI Health Agent &nbsp;·&nbsp; multi-agent &nbsp;·&nbsp;
        gene + wearable + NL fusion
      </p>
    </div>
  </div>
  <div style="display: flex; gap: 0.5rem; margin-top: 1.1rem; flex-wrap: wrap;">
    <span style="background: rgba(0,0,0,0.35); padding: 0.3rem 0.8rem;
                 border-radius: 999px; font-size: 0.78rem; font-weight: 600;
                 border: 1px solid rgba(255,255,255,0.2);">📊 Data Scientist</span>
    <span style="background: rgba(0,0,0,0.35); padding: 0.3rem 0.8rem;
                 border-radius: 999px; font-size: 0.78rem; font-weight: 600;
                 border: 1px solid rgba(255,255,255,0.2);">🧬 Domain Expert</span>
    <span style="background: rgba(0,0,0,0.35); padding: 0.3rem 0.8rem;
                 border-radius: 999px; font-size: 0.78rem; font-weight: 600;
                 border: 1px solid rgba(255,255,255,0.2);">🎯 Health Coach</span>
    <span style="background: rgba(0,0,0,0.35); padding: 0.3rem 0.8rem;
                 border-radius: 999px; font-size: 0.78rem; font-weight: 600;
                 border: 1px solid rgba(255,255,255,0.2);">🧭 Orchestrator</span>
  </div>
</div>
"""

ARCHITECTURE_SVG = """
<svg viewBox="0 0 820 280" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3"
            orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#6b7280"/>
    </marker>
    <linearGradient id="orchGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#dc2626"/>
      <stop offset="100%" stop-color="#7f1d1d"/>
    </linearGradient>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#fef2f2"/>
      <stop offset="100%" stop-color="#fafafa"/>
    </linearGradient>
    <!-- Reusable paths for animateMotion -->
    <path id="userToOrch" d="M 92,140 L 175,140" fill="none"/>
    <path id="orchToDS"   d="M 320,120 L 395,50"  fill="none"/>
    <path id="orchToDE"   d="M 320,140 L 395,140" fill="none"/>
    <path id="orchToHC"   d="M 320,160 L 395,230" fill="none"/>
  </defs>

  <rect width="820" height="280" fill="url(#bgGrad)" rx="12"/>

  <!-- User -->
  <circle cx="55" cy="140" r="32" fill="#dc2626" stroke="#fafafa" stroke-width="2"/>
  <text x="55" y="135" text-anchor="middle" fill="white" font-size="22">👤</text>
  <text x="55" y="155" text-anchor="middle" fill="white" font-size="11" font-weight="bold">User</text>

  <!-- Connection lines -->
  <line x1="92" y1="140" x2="175" y2="140" stroke="#a3a3a3" stroke-width="2"
        marker-end="url(#arrow)"/>

  <!-- Orchestrator -->
  <rect x="180" y="100" width="140" height="80" fill="url(#orchGrad)" rx="10"/>
  <text x="250" y="128" text-anchor="middle" fill="white" font-size="14"
        font-weight="bold">🧭 Orchestrator</text>
  <text x="250" y="148" text-anchor="middle" fill="white" font-size="11"
        opacity="0.9">Intent classifier</text>
  <text x="250" y="163" text-anchor="middle" fill="white" font-size="11"
        opacity="0.9">+ routing</text>

  <!-- Lines from orchestrator to agents -->
  <line x1="320" y1="120" x2="395" y2="50" stroke="#a3a3a3" stroke-width="2"
        marker-end="url(#arrow)"/>
  <line x1="320" y1="140" x2="395" y2="140" stroke="#a3a3a3" stroke-width="2"
        marker-end="url(#arrow)"/>
  <line x1="320" y1="160" x2="395" y2="230" stroke="#a3a3a3" stroke-width="2"
        marker-end="url(#arrow)"/>

  <!-- ANIMATED FLOW DOTS - data moving through the pipeline -->
  <circle r="4" fill="#dc2626" opacity="1">
    <animateMotion dur="2.5s" repeatCount="indefinite">
      <mpath href="#userToOrch"/>
    </animateMotion>
  </circle>
  <circle r="4" fill="#1f2937" opacity="0.95">
    <animateMotion dur="3s" repeatCount="indefinite" begin="0.5s">
      <mpath href="#orchToDS"/>
    </animateMotion>
  </circle>
  <circle r="4" fill="#dc2626" opacity="0.95">
    <animateMotion dur="3s" repeatCount="indefinite" begin="1s">
      <mpath href="#orchToDE"/>
    </animateMotion>
  </circle>
  <circle r="4" fill="#1f2937" opacity="0.95">
    <animateMotion dur="3s" repeatCount="indefinite" begin="1.5s">
      <mpath href="#orchToHC"/>
    </animateMotion>
  </circle>

  <!-- Three agents -->
  <rect x="395" y="20" width="170" height="62" fill="#ffffff" rx="10"
        stroke="#dc2626" stroke-width="2"/>
  <text x="480" y="44" text-anchor="middle" fill="#1f2937" font-size="13"
        font-weight="bold">📊 Data Scientist</text>
  <text x="480" y="64" text-anchor="middle" fill="#6b7280" font-size="10">wearables · stats · ML</text>

  <rect x="395" y="110" width="170" height="62" fill="#ffffff" rx="10"
        stroke="#dc2626" stroke-width="2"/>
  <text x="480" y="134" text-anchor="middle" fill="#1f2937" font-size="13"
        font-weight="bold">🧬 Domain Expert</text>
  <text x="480" y="154" text-anchor="middle" fill="#6b7280" font-size="10">genes · KB / RAG</text>

  <rect x="395" y="200" width="170" height="62" fill="#ffffff" rx="10"
        stroke="#dc2626" stroke-width="2"/>
  <text x="480" y="224" text-anchor="middle" fill="#1f2937" font-size="13"
        font-weight="bold">🎯 Health Coach</text>
  <text x="480" y="244" text-anchor="middle" fill="#6b7280" font-size="10">plans · motivation</text>

  <!-- Dashed arrows from agents to data layer with pulse -->
  <line x1="565" y1="50" x2="635" y2="120" stroke="#9ca3af" stroke-width="1.5"
        stroke-dasharray="5,3">
    <animate attributeName="stroke-dashoffset" from="0" to="-16" dur="2s"
             repeatCount="indefinite"/>
  </line>
  <line x1="565" y1="141" x2="635" y2="141" stroke="#9ca3af" stroke-width="1.5"
        stroke-dasharray="5,3">
    <animate attributeName="stroke-dashoffset" from="0" to="-16" dur="2s"
             repeatCount="indefinite"/>
  </line>
  <line x1="565" y1="232" x2="635" y2="160" stroke="#9ca3af" stroke-width="1.5"
        stroke-dasharray="5,3">
    <animate attributeName="stroke-dashoffset" from="0" to="-16" dur="2s"
             repeatCount="indefinite"/>
  </line>

  <!-- Data layer -->
  <rect x="635" y="100" width="170" height="80" fill="#1f2937" rx="10"/>
  <text x="720" y="124" text-anchor="middle" fill="#fafafa" font-size="13"
        font-weight="bold">Data layer</text>
  <text x="720" y="144" text-anchor="middle" fill="#fca5a5" font-size="10">Gene panel · 10 SNPs</text>
  <text x="720" y="158" text-anchor="middle" fill="#fafafa" font-size="10">Wearables · 1000 users</text>
  <text x="720" y="172" text-anchor="middle" fill="#cbd5e1" font-size="10">Knowledge base · RAG</text>
</svg>
"""


# DNA helix banner used at the top of the Genes tab.
DNA_HELIX_SVG = """
<svg viewBox="0 0 800 110" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;background:#fafafa;">
  <defs>
    <linearGradient id="dnaA" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#dc2626"/>
      <stop offset="100%" stop-color="#7f1d1d"/>
    </linearGradient>
    <linearGradient id="dnaB" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1f2937"/>
      <stop offset="100%" stop-color="#374151"/>
    </linearGradient>
  </defs>

  <!-- Light gradient background -->
  <rect width="800" height="110" fill="#fafafa"/>

  <!-- Two DNA strands as repeating sine-like paths -->
  <path d="M 0 30 Q 50 5  100 30 T 200 30 T 300 30 T 400 30 T 500 30 T 600 30 T 700 30 T 800 30"
        fill="none" stroke="url(#dnaA)" stroke-width="3"/>
  <path d="M 0 80 Q 50 105 100 80 T 200 80 T 300 80 T 400 80 T 500 80 T 600 80 T 700 80 T 800 80"
        fill="none" stroke="url(#dnaB)" stroke-width="3"/>

  <!-- Base-pair rungs with a pulse animation -->""" + "".join([
    f'<line x1="{x}" y1="30" x2="{x}" y2="80" stroke="#94a3b8" stroke-width="1.5" opacity="0.7">'
    f'<animate attributeName="opacity" values="0.3;1;0.3" dur="2s" begin="{(x % 200) / 100}s" repeatCount="indefinite"/>'
    f'</line>'
    for x in range(25, 800, 25)
]) + """
</svg>
"""


# ---------------------------------------------------------------------------
# Cached DB helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def list_users() -> pd.DataFrame:
    """All users with day counts and gene-variant counts. Sorted by data richness."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            """
            SELECT u.user_id, u.source, u.synthetic, u.age, u.gender, u.bmi,
                   COUNT(DISTINCT d.date)  AS n_days,
                   COUNT(DISTINCT v.rsid)  AS n_variants
            FROM users u
            LEFT JOIN daily_summary d  USING (user_id)
            LEFT JOIN user_variants v  USING (user_id)
            GROUP BY u.user_id
            ORDER BY
                CASE u.source
                    WHEN 'onboarded' THEN 1
                    WHEN 'lifesnaps' THEN 2
                    WHEN 'synthetic' THEN 3
                    ELSE 4
                END,
                u.created_at DESC,
                n_days DESC
            """,
            conn,
        )
    return df


@st.cache_data(show_spinner=False)
def get_user_recent(user_id: str, days: int = 30) -> pd.DataFrame:
    """The user's most recent N days of daily_summary rows."""
    with sqlite3.connect(DB_PATH) as conn:
        max_date = conn.execute(
            "SELECT MAX(date) FROM daily_summary WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        if not max_date:
            return pd.DataFrame()
        df = pd.read_sql(
            "SELECT date, steps, sleep_min, hrv_rmssd, stress_score, "
            "       resting_hr, very_active_min, sleep_efficiency, "
            "       mood_tired, mood_happy "
            "FROM daily_summary "
            "WHERE user_id = ? AND date <= ? "
            "ORDER BY date DESC LIMIT ?",
            conn, params=(user_id, max_date, days),
        )
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_latest_narrative(user_id: str) -> tuple[str, str] | None:
    """Most recent (date, text) narrative for the user, or None."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT date, text FROM nl_narratives "
            "WHERE user_id = ? ORDER BY date DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return (str(row[0]), str(row[1])) if row else None


@st.cache_data(show_spinner=False)
def get_user_plans(user_id: str) -> pd.DataFrame:
    """All saved plans for the user, most recent first."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT plan_id, query, plan_json, status, created_at "
            "FROM plans WHERE user_id = ? ORDER BY created_at DESC",
            conn, params=(user_id,),
        )
    return df


def delete_plan(plan_id: int) -> None:
    """Delete one plan by id. NOT cached (mutating)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM plans WHERE plan_id = ?", (int(plan_id),))
        conn.commit()


@st.cache_data(show_spinner=False)
def get_user_genes_df(user_id: str) -> pd.DataFrame:
    """User's gene panel with full details + per-genotype interpretation.
    Wraps agents.tools.get_user_genes for consistency with the agents."""
    from agents.tools import get_user_genes
    result = get_user_genes(user_id)
    if "error" in result:
        return pd.DataFrame()
    return pd.DataFrame(result["variants"])


@st.cache_data(show_spinner=False)
def get_cohort_baselines() -> dict[str, dict[str, float]]:
    """Per-metric distribution of per-user means across the whole cohort."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            """
            SELECT user_id,
                   AVG(steps)         AS steps,
                   AVG(sleep_min)     AS sleep_min,
                   AVG(hrv_rmssd)     AS hrv_rmssd,
                   AVG(stress_score)  AS stress_score,
                   AVG(resting_hr)    AS resting_hr
            FROM daily_summary GROUP BY user_id
            """,
            conn,
        )
    out: dict[str, dict[str, float]] = {}
    for col in ("steps", "sleep_min", "hrv_rmssd", "stress_score", "resting_hr"):
        s = df[col].dropna()
        if s.empty:
            continue
        out[col] = {
            "median": float(s.median()),
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
        }
    return out


def _user_label(row: pd.Series) -> str:
    """Compact display label for the picker dropdown."""
    src = row["source"]
    badge = {"lifesnaps": "REAL", "synthetic": "SYNTH", "onboarded": "YOU"}.get(src, src.upper())
    short_id = (str(row["user_id"])[:14] + "...") if len(str(row["user_id"])) > 14 else str(row["user_id"])
    return f"[{badge}] {short_id}  ({row['n_days']}d, {row['n_variants']} variants)"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

@st.dialog("Onboard yourself", width="large")
def show_onboarding_dialog() -> None:
    """Questionnaire -> synthesised user -> auto-switch the sidebar to them."""
    st.markdown(
        "Answer a short questionnaire and PHAI will synthesise a 30-day "
        "wearable trajectory plus a plausible 10-SNP gene panel based on "
        "your answers. The chat then works on this profile immediately. "
        "**The generated data is clearly flagged as synthetic.**"
    )

    with st.form("onboarding_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
            sex = st.radio("Sex", ["female", "male", "other"], horizontal=True)
            bmi = st.number_input(
                "BMI", min_value=15.0, max_value=40.0, value=23.0, step=0.5
            )
        with c2:
            chronotype = st.radio(
                "Chronotype",
                options=["early", "intermediate", "late"],
                index=1,
                format_func=lambda x: {
                    "early": "Morning person",
                    "intermediate": "Intermediate",
                    "late": "Night owl",
                }[x],
            )
            caffeine = st.radio(
                "Caffeine sensitivity",
                options=["sensitive", "moderate", "tolerant"],
                index=1,
                format_func=lambda x: {
                    "sensitive": "Very sensitive (one cup keeps me up)",
                    "moderate": "Moderate",
                    "tolerant": "Tolerates a lot",
                }[x],
            )

        exercise = st.radio(
            "Typical exercise level",
            options=["sedentary", "moderate", "active", "very_active"],
            index=1,
            horizontal=True,
            format_func=lambda x: {
                "sedentary": "Sedentary",
                "moderate": "Moderate",
                "active": "Active",
                "very_active": "Very active",
            }[x],
        )
        sleep_quality = st.radio(
            "Typical sleep quality",
            options=["poor", "fair", "good"],
            index=1,
            horizontal=True,
            format_func=lambda x: {"poor": "Often poor", "fair": "Mixed", "good": "Generally good"}[x],
        )
        stress_level = st.radio(
            "Typical stress level",
            options=["low", "moderate", "high"],
            index=1,
            horizontal=True,
            format_func=lambda x: {"low": "Low", "moderate": "Moderate", "high": "High"}[x],
        )

        submitted = st.form_submit_button(
            "Create my profile and start chatting",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        profile = {
            "age": int(age),
            "sex": sex,
            "bmi": float(bmi),
            "chronotype": chronotype,
            "caffeine_sensitivity": caffeine,
            "exercise_level": exercise,
            "sleep_quality": sleep_quality,
            "stress_level": stress_level,
        }
        with st.spinner("Generating your synthetic profile..."):
            from onboarding.synth_user import create_user_from_questionnaire
            new_uid = create_user_from_questionnaire(profile, days=30)

        # Force everything to refresh: caches + selector.
        st.cache_data.clear()
        st.session_state["new_onboarded_uid"] = new_uid
        # Drop chat history so we don't carry the old user's conversation.
        st.session_state["chat_history"] = []
        st.session_state["chat_user_id"] = new_uid

        st.success(f"Profile created — `{new_uid}`. Switching now...")
        st.rerun()


def render_sidebar() -> str:
    """Render the sidebar and return the selected user_id."""
    st.sidebar.markdown("# PHAI")
    st.sidebar.caption(
        "Personalised AI Health Agent — gene + wearable + NL fusion."
    )
    st.sidebar.divider()

    # If we just onboarded a user, force the sidebar selector to that user.
    if "new_onboarded_uid" in st.session_state:
        new_uid = st.session_state.pop("new_onboarded_uid")
        users_now = list_users()
        idx_match = users_now.index[users_now["user_id"] == new_uid].tolist()
        if idx_match:
            st.session_state["user_selector"] = int(idx_match[0])

    users = list_users()

    # Build options once.
    labels = [_user_label(r) for _, r in users.iterrows()]
    user_ids = users["user_id"].tolist()

    # Default to the user with the richest data (already first because of ORDER BY).
    default_index = 0
    selected_index = st.sidebar.selectbox(
        "Choose user",
        options=range(len(labels)),
        format_func=lambda i: labels[i],
        index=default_index,
        key="user_selector",
    )
    user_id = user_ids[selected_index]
    user_row = users.iloc[selected_index]

    # Profile card.
    st.sidebar.markdown("### Profile")
    st.sidebar.markdown(f"**Source:** `{user_row['source']}`")
    st.sidebar.markdown(f"**Days of wearable data:** {int(user_row['n_days'])}")
    st.sidebar.markdown(f"**Gene variants:** {int(user_row['n_variants'])}")

    if pd.notna(user_row.get("age")):
        st.sidebar.markdown(f"**Age:** {int(user_row['age'])}")
    if user_row.get("gender"):
        st.sidebar.markdown(f"**Gender:** {user_row['gender']}")
    if pd.notna(user_row.get("bmi")):
        st.sidebar.markdown(f"**BMI:** {float(user_row['bmi']):.1f}")

    if user_row["source"] == "synthetic":
        st.sidebar.warning(
            "Synthetic user (resampled from a real LifeSnaps participant "
            "with controlled noise).",
            icon=None,
        )

    st.sidebar.divider()

    # Open the questionnaire dialog.
    if st.sidebar.button(
        "Onboard yourself",
        use_container_width=True,
        type="primary",
    ):
        show_onboarding_dialog()

    st.sidebar.divider()
    st.sidebar.caption(
        "Inspired by Google Research's *The Anatomy of a Personal Health "
        "Agent* (arXiv 2508.20148)."
    )

    return str(user_id)


# ---------------------------------------------------------------------------
# Tab placeholders (filled in by Steps 17b-17e)
# ---------------------------------------------------------------------------

def _line_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    cohort: dict,
    *,
    color: str = "#3b82f6",
) -> go.Figure:
    """One Plotly line chart for a metric, with cohort p25-p75 band + median."""
    fig = go.Figure()

    if metric in cohort and not df.empty:
        b = cohort[metric]
        date_min, date_max = df["date"].min(), df["date"].max()
        # Cohort p25-p75 band as a filled rectangle.
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=date_min, x1=date_max,
            y0=b["p25"], y1=b["p75"],
            fillcolor="rgba(120,120,120,0.10)",
            line=dict(width=0),
            layer="below",
        )
        # Cohort median as a dashed reference line.
        fig.add_hline(
            y=b["median"],
            line_dash="dash",
            line_color="rgba(120,120,120,0.6)",
            annotation_text=f"cohort median: {b['median']:.0f}",
            annotation_position="bottom right",
            annotation_font_size=10,
        )

    # The user's daily values.
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[metric],
            mode="lines+markers",
            name="You",
            line=dict(color=color, width=2),
            marker=dict(size=5),
            connectgaps=False,  # leave a gap on missing days, don't interpolate
        )
    )
    fig.update_layout(
        title=dict(text=title, x=0.0, font=dict(size=14)),
        height=270,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        xaxis=dict(title=None),
        yaxis=dict(title=None),
        hovermode="x unified",
    )
    return fig


def _metric_card(
    col,
    label: str,
    value: float | None,
    cohort_median: float | None,
    *,
    suffix: str = "",
    fmt: str = "{:.0f}",
    inverse: bool = False,
) -> None:
    """One Streamlit metric card with a delta vs cohort median."""
    if value is None or pd.isna(value):
        col.metric(label, "no data")
        return
    val_str = fmt.format(value) + suffix
    if cohort_median is None or pd.isna(cohort_median):
        col.metric(label, val_str)
        return
    delta = value - cohort_median
    delta_str = ("+" if delta >= 0 else "") + fmt.format(delta).lstrip("+") + suffix
    col.metric(
        label,
        val_str,
        delta=delta_str,
        delta_color=("inverse" if inverse else "normal"),
    )


def _gradient_card(
    col,
    icon: str,
    label: str,
    value: float | None,
    cohort_median: float | None,
    *,
    grad_from: str,
    grad_to: str,
    suffix: str = "",
    fmt: str = "{:.0f}",
    inverse: bool = False,
) -> None:
    """A coloured gradient card with an icon, value, and delta vs cohort."""
    if value is None or pd.isna(value):
        val_str, delta_html = "no data", ""
    else:
        val_str = fmt.format(value) + suffix
        if cohort_median is None or pd.isna(cohort_median):
            delta_html = "<span style='opacity:0.85;font-size:0.8rem;'>no cohort baseline</span>"
        else:
            delta = value - cohort_median
            sign = "+" if delta >= 0 else ""
            arrow = "▲" if delta >= 0 else "▼"
            # For inverse metrics (e.g. stress), positive delta = bad.
            good = (delta < 0) if inverse else (delta >= 0)
            color = "#dcfce7" if good else "#fee2e2"
            delta_html = (
                f"<span style='background:{color};color:#1f2937;"
                f"padding:0.1rem 0.5rem;border-radius:999px;"
                f"font-size:0.78rem;font-weight:600;'>"
                f"{arrow} {sign}{fmt.format(delta).lstrip('+')}{suffix} vs cohort</span>"
            )

    col.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {grad_from} 0%, {grad_to} 100%);
            color: white;
            padding: 1rem 1.1rem;
            border-radius: 14px;
            box-shadow: 0 3px 14px {grad_from}33;
            min-height: 130px;
        ">
          <div style="display:flex; align-items:center; gap:0.5rem; opacity:0.95;">
            <span style="font-size:1.3rem;">{icon}</span>
            <span style="font-size:0.85rem; font-weight:600; letter-spacing:0.02em;">
              {label}
            </span>
          </div>
          <div style="font-size:1.9rem; font-weight:700; margin-top:0.4rem; line-height:1;">
            {val_str}
          </div>
          <div style="margin-top:0.55rem;">{delta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_tab(user_id: str) -> None:
    df = get_user_recent(user_id, days=30)

    if df.empty:
        st.warning(f"No daily data for user `{user_id}`.")
        return

    # ---- Latest narrative ----
    narr = get_latest_narrative(user_id)
    if narr:
        date, text = narr
        st.info(f"**{date}**  ·  {text}")
    else:
        st.info("No narratives generated for this user yet.")

    # ---- Headline metric cards ----
    cohort = get_cohort_baselines()

    avg_steps = df["steps"].mean() if df["steps"].notna().any() else None
    avg_sleep_min = df["sleep_min"].mean() if df["sleep_min"].notna().any() else None
    avg_sleep_h = (avg_sleep_min / 60) if avg_sleep_min is not None else None
    avg_hrv = df["hrv_rmssd"].mean() if df["hrv_rmssd"].notna().any() else None
    avg_stress = df["stress_score"].mean() if df["stress_score"].notna().any() else None

    cohort_steps = cohort.get("steps", {}).get("median")
    cohort_sleep_h = (cohort.get("sleep_min", {}).get("median") or 0) / 60 if cohort.get("sleep_min") else None
    cohort_hrv = cohort.get("hrv_rmssd", {}).get("median")
    cohort_stress = cohort.get("stress_score", {}).get("median")

    cols = st.columns(4)
    _gradient_card(cols[0], "👟", "Steps (30d avg)", avg_steps, cohort_steps,
                   grad_from="#dc2626", grad_to="#7f1d1d", fmt="{:,.0f}")
    _gradient_card(cols[1], "😴", "Sleep (30d avg)", avg_sleep_h, cohort_sleep_h,
                   grad_from="#6366f1", grad_to="#3730a3", fmt="{:.1f}", suffix=" h")
    _gradient_card(cols[2], "❤️", "HRV (30d avg)", avg_hrv, cohort_hrv,
                   grad_from="#ec4899", grad_to="#9d174d", fmt="{:.0f}", suffix=" ms")
    _gradient_card(cols[3], "🧘", "Stress (30d avg)", avg_stress, cohort_stress,
                   grad_from="#f59e0b", grad_to="#b45309", fmt="{:.0f}", inverse=True)
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    st.divider()

    # ---- Line charts ----
    chart_cols_top = st.columns(2)
    with chart_cols_top[0]:
        st.plotly_chart(
            _line_chart(df, "steps", "Daily steps", cohort, color="#3b82f6"),
            use_container_width=True,
        )
    with chart_cols_top[1]:
        st.plotly_chart(
            _line_chart(df, "sleep_min", "Sleep (minutes)", cohort, color="#8b5cf6"),
            use_container_width=True,
        )

    chart_cols_bot = st.columns(2)
    with chart_cols_bot[0]:
        st.plotly_chart(
            _line_chart(df, "hrv_rmssd", "HRV (rmssd, ms)", cohort, color="#10b981"),
            use_container_width=True,
        )
    with chart_cols_bot[1]:
        st.plotly_chart(
            _line_chart(df, "stress_score", "Stress score", cohort, color="#f59e0b"),
            use_container_width=True,
        )

    # ---- Tiny coverage footer ----
    coverage = {
        "steps": int(df["steps"].notna().sum()),
        "sleep_min": int(df["sleep_min"].notna().sum()),
        "hrv_rmssd": int(df["hrv_rmssd"].notna().sum()),
        "stress_score": int(df["stress_score"].notna().sum()),
    }
    cov_text = "  ·  ".join(f"{m}: {c}/{len(df)}d" for m, c in coverage.items())
    st.caption(f"Data coverage in window — {cov_text}.  Shaded band = cohort p25-p75 (1000 users).")


def render_genes_tab(user_id: str) -> None:
    genes = get_user_genes_df(user_id)

    if genes.empty:
        st.warning(f"No gene data for user `{user_id}`.")
        return

    # Decorative DNA helix banner. Use components.html so the iframe
    # bypasses the markdown sanitizer that strips <animate> SVG tags.
    # Height tuned to fit the SVG's 800:110 aspect ratio.
    components.html(DNA_HELIX_SVG, height=170, scrolling=False)

    st.caption(
        f"{len(genes)} curated lifestyle-relevant variants in the panel. "
        "Genotypes are sampled under Hardy-Weinberg equilibrium from "
        "published population allele frequencies (gnomAD / 1000 Genomes); "
        "see README for the honest disclosure."
    )

    # ---- Compact summary table ----
    summary = genes[["gene", "rsid", "genotype", "interpretation"]].rename(
        columns={
            "gene": "Gene",
            "rsid": "Variant",
            "genotype": "Your genotype",
            "interpretation": "What this means for you",
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Expand any gene below for the full card and citation.")

    # ---- Per-gene detail cards ----
    for _, row in genes.iterrows():
        title = (
            f"**{row['gene']}**  ·  {row['rsid']}  ·  "
            f"your genotype: **{row['genotype']}**"
        )
        with st.expander(title):
            left, right = st.columns([3, 1])
            with left:
                st.markdown("**What this gene does**")
                st.markdown(row["trait_summary"])

                st.markdown(f"**What your `{row['genotype']}` genotype means**")
                st.markdown(row["interpretation"])

                st.markdown("**Lifestyle implications**")
                st.markdown(row["lifestyle_implications"])
            with right:
                st.markdown("**Classification**")
                sig = row.get("clinvar_significance") or "polymorphism"
                st.markdown(f"`{sig}`")
                if row.get("citation_url"):
                    st.markdown(f"[Reference →]({row['citation_url']})")


SUGGESTED_QUERIES: list[tuple[str, str]] = [
    ("DS-led",   "Why have I been sluggish this week?"),
    ("DE-led",   "What does my CYP1A2 result mean for caffeine?"),
    ("Full plan","Build me a 7-day plan to feel more energetic."),
]


def _render_plan(plan: dict) -> None:
    """Render a Health Coach structured plan as a bordered card."""
    with st.container(border=True):
        st.markdown(f"### {plan.get('goal', 'Plan')}")
        if plan.get("why"):
            st.markdown(f"**Why this for you:** {plan['why']}")

        st.markdown("**Steps:**")
        for i, step in enumerate(plan.get("steps", []), 1):
            st.markdown(f"**{i}. {step.get('title', '(untitled)')}**")
            if step.get("detail"):
                st.markdown(step["detail"])
            meta_bits = []
            if step.get("frequency"):
                meta_bits.append(f"Frequency: {step['frequency']}")
            if step.get("evidence"):
                meta_bits.append(f"Evidence: {step['evidence']}")
            if meta_bits:
                st.caption("  ·  ".join(meta_bits))

        cols = st.columns(2)
        if plan.get("metrics_to_track"):
            cols[0].markdown(
                "**Metrics to track:** " + ", ".join(plan["metrics_to_track"])
            )
        cols[1].markdown(
            f"**Check-in:** in {plan.get('check_in_days', 7)} days"
        )


def _render_trace(intent: str, agents_run: list[str], trace_by_agent: dict) -> None:
    """Render the 'How I built this answer' panel."""
    with st.expander(
        "**How I built this answer** — agents, tool calls, evidence",
        expanded=False,
    ):
        st.markdown(f"**Intent classification:** `{intent}`")
        st.markdown(
            "**Pipeline:** "
            + (" → ".join(agents_run) if agents_run else "_no agents fired_")
        )

        for agent_name, trace in trace_by_agent.items():
            st.markdown("---")
            st.markdown(f"#### {agent_name}")
            tool_calls = [e for e in trace if e.get("type") == "tool_call"]
            if not tool_calls:
                st.caption("(no tool calls)")
                continue
            st.caption(f"{len(tool_calls)} tool call(s)")
            for i, event in enumerate(tool_calls, 1):
                args = event.get("args", {})
                summary = f"Step {i}: `{event['name']}({', '.join(args.keys())})`"
                with st.expander(summary, expanded=False):
                    st.markdown("**Arguments:**")
                    st.json(args)
                    st.markdown("**Result:**")
                    result = event.get("result", {})
                    if isinstance(result, dict) and "error" in result:
                        st.error(result["error"])
                    else:
                        st.json(result)


def _render_assistant_message(msg: dict) -> None:
    """Render the assistant side of a saved chat message."""
    st.markdown(msg["content"])
    if msg.get("plan"):
        _render_plan(msg["plan"])
    if msg.get("trace_by_agent"):
        _render_trace(
            intent=msg.get("intent", "?"),
            agents_run=msg.get("agents_run", []),
            trace_by_agent=msg["trace_by_agent"],
        )


def _show_suggestions() -> None:
    """Architecture diagram + suggested-query buttons when chat is empty."""
    with st.container(border=True):
        st.markdown("#### How a question travels through PHAI")
        # Use components.html so the iframe bypasses the markdown sanitizer,
        # which strips <animate> / <animateMotion> SVG tags. Height tuned
        # to fit the SVG's 820:280 aspect ratio on wide screens.
        components.html(ARCHITECTURE_SVG, height=480, scrolling=False)
        st.caption(
            "The Orchestrator classifies your question, fires the right "
            "specialist sub-agents (often in sequence for plan requests), "
            "and synthesises one grounded answer."
        )

    st.markdown("##### Try one of these to see different agent routes:")
    cols = st.columns(len(SUGGESTED_QUERIES))
    for col, (label, q) in zip(cols, SUGGESTED_QUERIES):
        if col.button(
            f"**{label}**\n\n{q}",
            use_container_width=True,
            key=f"suggest_{label}",
        ):
            st.session_state["pending_query"] = q
            st.rerun()


def render_chat_tab(user_id: str) -> None:
    # Reset history when the user changes (chat is per-person).
    if st.session_state.get("chat_user_id") != user_id:
        st.session_state["chat_history"] = []
        st.session_state["chat_user_id"] = user_id
    st.session_state.setdefault("chat_history", [])

    # Header with clear-chat control.
    head_cols = st.columns([4, 1])
    head_cols[0].caption(
        f"Chatting about user `{user_id[:14]}...`. "
        "Switching users in the sidebar clears the chat."
    )
    if head_cols[1].button(
        "Clear chat",
        use_container_width=True,
        disabled=not st.session_state["chat_history"],
    ):
        st.session_state["chat_history"] = []
        st.rerun()

    # Suggested queries: prominent when empty, folded into an expander after.
    if not st.session_state["chat_history"]:
        _show_suggestions()
    else:
        with st.expander("Try another agent route", expanded=False):
            _show_suggestions()

    # Replay history.
    for msg in st.session_state["chat_history"]:
        avatar = "🧬" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant_message(msg)

    # ALWAYS render the chat input so it's available after every turn.
    # (Gating it behind `pending` short-circuited and never drew the widget.)
    typed_input = st.chat_input(
        "Ask PHAI about your data, your genes, or for a personalised plan..."
    )
    # A pending query (from a clicked suggestion) takes precedence over typed.
    pending = st.session_state.pop("pending_query", None)
    user_input = pending or typed_input

    if not user_input:
        return

    # Append + render the user turn immediately.
    st.session_state["chat_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Run the orchestrator with live status updates.
    from agents import orchestrator  # local import keeps cold-start cheap

    with st.chat_message("assistant", avatar="🧬"):
        status = st.status("Thinking...", expanded=True)

        def on_step(event_type: str, data: dict) -> None:
            if event_type == "intent_classified":
                status.update(label=f"Intent: {data['intent']}")
                status.write(f"**Intent:** `{data['intent']}`")
            elif event_type == "agent_start":
                status.write(f"→ Running **{data['agent']}** agent...")
            elif event_type == "tool_call":
                status.write(f"   · `{data['name']}` called by {data['agent']}")
            elif event_type == "agent_end":
                status.write(
                    f"   · **{data['agent']}** done in "
                    f"{data.get('iterations', '?')} iterations"
                )
            elif event_type == "synthesise_start":
                status.write("→ Synthesising final answer...")

        try:
            result = orchestrator.run(user_id, user_input, on_step=on_step)
            status.update(
                label=(
                    f"Done · {result['intent']} · "
                    + (" → ".join(result["agents_run"]) or "no agents")
                ),
                state="complete",
                expanded=False,
            )
        except Exception as e:
            status.update(label="Pipeline failed", state="error")
            st.error(f"The agent pipeline failed: {e}")
            # Don't save to history when failed.
            return

        # Render answer / plan / trace inline so the user sees them now.
        st.markdown(result["answer"])
        if result.get("plan"):
            _render_plan(result["plan"])
        _render_trace(
            intent=result["intent"],
            agents_run=result["agents_run"],
            trace_by_agent=result["trace_by_agent"],
        )

    # Save the assistant turn to history.
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": result["answer"],
        "intent": result["intent"],
        "agents_run": result["agents_run"],
        "trace_by_agent": result["trace_by_agent"],
        "plan": result.get("plan"),
        "plan_id": result.get("plan_id"),
    })

    # If the Coach saved a plan, invalidate the Plans tab cache so it
    # picks up the new row on next render.
    if result.get("plan_id"):
        get_user_plans.clear()

    # Rerun so the header (Clear chat enabled), suggestion area (folded
    # into expander), and chat input all refresh with the updated state.
    # The just-rendered messages will reappear from the history loop.
    st.rerun()


def render_plans_tab(user_id: str) -> None:
    plans_df = get_user_plans(user_id)

    if plans_df.empty:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 2.5rem 1rem;
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%);
            border-radius: 16px;
            border: 1px dashed #a7f3d0;
        ">
          <div style="font-size: 3rem; margin-bottom: 0.5rem;">📋</div>
          <h3 style="margin: 0; color: #047857;">No plans saved yet</h3>
          <p style="margin: 0.5rem 0 0 0; color: #4b5563;">
            Head to the <b>💬 Chat</b> tab and ask
            <i>"Build me a 7-day plan to feel more energetic"</i> —
            the Health Coach's plans land here automatically.
          </p>
        </div>
        """, unsafe_allow_html=True)
        return

    st.caption(f"{len(plans_df)} saved plan(s) for this user, most recent first.")

    for _, row in plans_df.iterrows():
        try:
            plan = json.loads(row["plan_json"])
        except json.JSONDecodeError:
            continue

        plan_id = int(row["plan_id"])
        # Colour-coded status indicator.
        status_dot = {
            "active": "🟢",
            "completed": "✅",
            "archived": "⚪",
        }.get(row["status"], "🟡")
        title = (
            f"{status_dot}  **{plan.get('goal', 'Plan')}**  ·  "
            f"saved {row['created_at']}  ·  "
            f"status: `{row['status']}`"
        )
        with st.expander(title, expanded=False):
            if row["query"]:
                st.caption(f"Original question: \"{row['query']}\"")
            _render_plan(plan)

            # Delete control - small button at the bottom-right.
            st.divider()
            spacer, btn = st.columns([4, 1])
            spacer.caption(f"Plan ID: `{plan_id}`")
            if btn.button(
                "🗑️ Delete plan",
                key=f"delete_plan_{plan_id}",
                use_container_width=True,
            ):
                delete_plan(plan_id)
                get_user_plans.clear()
                st.toast(f"Plan #{plan_id} deleted.", icon="🗑️")
                st.rerun()


# ---------------------------------------------------------------------------
# Evaluation tab - all the metrics we measured during the build
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_energy_clf_metrics() -> dict:
    """Read the energy classifier metrics JSON saved by training."""
    path = PROJECT_ROOT / "models" / "energy_clf_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def get_cohort_validation() -> pd.DataFrame:
    """Per-source averages of headline metrics. Real and synthetic should be close."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            """
            SELECT u.source                            AS Source,
                   COUNT(DISTINCT u.user_id)           AS Users,
                   ROUND(AVG(d.steps), 0)              AS "Avg steps",
                   ROUND(AVG(d.sleep_min), 0)          AS "Avg sleep (min)",
                   ROUND(AVG(d.hrv_rmssd), 1)          AS "Avg HRV",
                   ROUND(AVG(d.resting_hr), 1)         AS "Avg resting HR",
                   ROUND(AVG(d.stress_score), 1)       AS "Avg stress"
            FROM users u
            JOIN daily_summary d USING (user_id)
            WHERE u.source IN ('lifesnaps', 'synthetic')
            GROUP BY u.source
            ORDER BY u.source
            """,
            conn,
        )
    return df


@st.cache_data(show_spinner=False)
def get_hwe_validation() -> pd.DataFrame:
    """Per-SNP observed vs HWE-expected genotype distribution."""
    from reference.snp_panel import PANEL

    with sqlite3.connect(DB_PATH) as conn:
        observed = pd.read_sql(
            "SELECT rsid, genotype, COUNT(*) AS n "
            "FROM user_variants GROUP BY rsid, genotype",
            conn,
        )

    rows = []
    for snp in PANEL:
        rsid = snp["rsid"]
        gene = snp["gene"]
        maf = snp["minor_allele_freq"]
        major, minor = snp["alleles"]

        snp_obs = observed[observed["rsid"] == rsid]
        total = int(snp_obs["n"].sum())
        if total == 0:
            continue

        # Hardy-Weinberg expected
        p, q = 1 - maf, maf
        exp_majmaj = (p * p) * 100
        exp_het = (2 * p * q) * 100
        exp_minmin = (q * q) * 100

        # Observed - try both orderings of het.
        majmaj_n = int(snp_obs[snp_obs["genotype"] == major + major]["n"].sum())
        minmin_n = int(snp_obs[snp_obs["genotype"] == minor + minor]["n"].sum())
        het_n = total - majmaj_n - minmin_n

        rows.append({
            "Gene": gene,
            "rsid": rsid,
            "MAF": maf,
            "Major hom %": f"{majmaj_n / total * 100:.1f}",
            "(expected)": f"{exp_majmaj:.1f}",
            "Het %": f"{het_n / total * 100:.1f}",
            " (expected)": f"{exp_het:.1f}",
            "Minor hom %": f"{minmin_n / total * 100:.1f}",
            "  (expected)": f"{exp_minmin:.1f}",
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_narrative_coverage() -> dict:
    """NL narrative generation stats - the calibration metric."""
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM nl_narratives").fetchone()[0]
        if total == 0:
            return {"total": 0}
        typical = conn.execute(
            "SELECT COUNT(*) FROM nl_narratives WHERE text LIKE 'A typical day%'"
        ).fetchone()[0]
        with_hrv = conn.execute(
            "SELECT COUNT(*) FROM nl_narratives WHERE LOWER(text) LIKE '%hrv%'"
        ).fetchone()[0]
        with_sleep = conn.execute(
            "SELECT COUNT(*) FROM nl_narratives "
            "WHERE LOWER(text) LIKE '%sleep%' OR LOWER(text) LIKE '%night%'"
        ).fetchone()[0]
        with_mood = conn.execute(
            "SELECT COUNT(*) FROM nl_narratives WHERE text LIKE '%logged feeling%'"
        ).fetchone()[0]
        with_steps = conn.execute(
            "SELECT COUNT(*) FROM nl_narratives "
            "WHERE LOWER(text) LIKE '%step%' OR LOWER(text) LIKE '%active%'"
        ).fetchone()[0]
    return {
        "total": int(total),
        "typical_pct": typical / total * 100,
        "with_hrv": int(with_hrv),
        "with_sleep": int(with_sleep),
        "with_mood": int(with_mood),
        "with_steps": int(with_steps),
    }


def run_kb_smoke_tests() -> pd.DataFrame:
    """A fixed set of queries against the KB. Returns top-1 hit + distance."""
    from agents.tools import kb_search

    queries = [
        "I want to feel more energetic",
        "my caffeine intake is keeping me up at night",
        "what should I eat to lose weight",
        "how do I improve sleep quality",
        "tips for managing stress",
        "best exercise for endurance",
        "morning daylight benefits",
    ]
    rows = []
    for q in queries:
        result = kb_search(q, k=1)
        if "error" in result or not result.get("results"):
            rows.append({"Query": q, "Top match": "(none)", "Topic": "-", "Distance": None})
            continue
        top = result["results"][0]
        rows.append({
            "Query": q,
            "Top match": top["snippet_id"],
            "Topic": top.get("topic", "-"),
            "Distance": round(top["distance"], 3),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_url_grounding() -> dict:
    """For every saved plan, % of cited URLs that exist in our KB or SNP panel."""
    import re

    # Build the set of legitimate URLs the system knows about.
    known: set[str] = set()

    kb_path = PROJECT_ROOT / "kb" / "snippets.json"
    if kb_path.exists():
        for s in json.loads(kb_path.read_text(encoding="utf-8")):
            url = (s.get("url") or "").rstrip("/")
            if url:
                known.add(url)

    with sqlite3.connect(DB_PATH) as conn:
        for (u,) in conn.execute(
            "SELECT citation_url FROM snp_reference WHERE citation_url IS NOT NULL"
        ).fetchall():
            if u:
                known.add(u.rstrip("/"))
        plans_df = pd.read_sql("SELECT plan_id, plan_json FROM plans", conn)

    if plans_df.empty:
        return {"n_plans": 0, "total_urls": 0, "urls_grounded": 0, "pct": None}

    url_pat = re.compile(r"https?://[^\s\)\]\}\"',]+")
    n_plans, total_urls, grounded = 0, 0, 0
    for _, row in plans_df.iterrows():
        try:
            plan = json.loads(row["plan_json"])
        except json.JSONDecodeError:
            continue
        n_plans += 1
        for u in url_pat.findall(json.dumps(plan)):
            u = u.rstrip(",.;)\"'/").rstrip("/")
            total_urls += 1
            if u in known:
                grounded += 1

    return {
        "n_plans": n_plans,
        "total_urls": total_urls,
        "urls_grounded": grounded,
        "pct": (grounded / total_urls * 100) if total_urls else None,
    }


def render_eval_tab() -> None:
    """The Evaluation tab - real numbers from the build."""
    st.markdown(
        "Real numbers, computed live from the database, models, and saved plans. "
        "What we measured rigorously, what we measured by sanity check, and what "
        "we honestly didn't measure."
    )

    # ---- Headline cards ----
    energy = get_energy_clf_metrics()
    with sqlite3.connect(DB_PATH) as conn:
        n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        n_plans_total = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
    kb_path = PROJECT_ROOT / "kb" / "snippets.json"
    n_snippets = len(json.loads(kb_path.read_text(encoding="utf-8"))) if kb_path.exists() else 0

    h1, h2, h3, h4 = st.columns(4)
    h1.metric(
        "Energy clf · Test AUC",
        f"{energy.get('test_auc', '-')}" if energy else "-",
    )
    h2.metric("Cohort size", f"{n_users:,}")
    h3.metric("KB snippets", f"{n_snippets}")
    h4.metric("Plans saved", f"{n_plans_total}")

    st.divider()

    # ---- 1. Energy classifier ----
    st.markdown("### 1. Energy classifier (XGBoost)")
    st.caption(
        "Predicts whether tomorrow's step count will exceed the user's personal "
        "median. Per-user time-based 70/30 split prevents future leakage."
    )

    if energy:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Train AUC", energy.get("train_auc", "-"))
        e2.metric("Test AUC", energy.get("test_auc", "-"))
        e3.metric("Test accuracy", energy.get("test_accuracy", "-"))
        e4.metric("Test base rate", energy.get("test_base_rate", "-"))

        e5, e6, e7 = st.columns(3)
        e5.metric("N train examples", f"{energy.get('n_train', 0):,}")
        e6.metric("N test examples", f"{energy.get('n_test', 0):,}")
        e7.metric("Test users", f"{energy.get('n_test_users', 0):,}")

        if "top_features" in energy:
            top = energy["top_features"]
            feat_df = (
                pd.DataFrame({"feature": list(top.keys()), "importance": list(top.values())})
                .sort_values("importance", ascending=True)
            )
            fig = go.Figure(go.Bar(
                x=feat_df["importance"],
                y=feat_df["feature"],
                orientation="h",
                marker=dict(color="#dc2626"),
            ))
            fig.update_layout(
                title="Top 10 features by XGBoost importance",
                height=320, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Importance", yaxis_title=None,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python -m models.energy_clf train` to populate metrics.")

    st.divider()

    # ---- 2. Cohort validation ----
    st.markdown("### 2. Cohort validation (real vs synthetic)")
    st.caption(
        "Synthetic users are generated by resampling LifeSnaps templates with "
        "controlled noise. Averages should be statistically close between real "
        "and synthetic - confirming noise was applied without systematic bias."
    )
    cohort_df = get_cohort_validation()
    if not cohort_df.empty:
        st.dataframe(cohort_df, use_container_width=True, hide_index=True)
        st.caption(
            "Pass condition: synthetic averages within roughly 5% of real averages."
        )

    st.divider()

    # ---- 3. Hardy-Weinberg validation ----
    st.markdown("### 3. Hardy-Weinberg genotype validation")
    st.caption(
        "Genotypes are sampled from published population allele frequencies "
        "under HWE. Observed frequencies should match expected (within sampling "
        "noise from a 1000-user cohort)."
    )
    hwe_df = get_hwe_validation()
    if not hwe_df.empty:
        st.dataframe(hwe_df, use_container_width=True, hide_index=True)

    st.divider()

    # ---- 4. KB retrieval smoke tests ----
    st.markdown("### 4. Knowledge base retrieval (smoke tests)")
    st.caption(
        "Sample queries against ChromaDB. Lower distance = more semantically "
        "relevant. Healthy retrieval shows on-topic snippets within ~0.4-1.0 "
        "for queries within the curated KB domain."
    )
    if st.button("▶ Run KB smoke tests", key="run_kb_smoke"):
        with st.spinner("Running queries..."):
            kb_df = run_kb_smoke_tests()
        st.dataframe(kb_df, use_container_width=True, hide_index=True)
    else:
        st.caption("Click to run the smoke-test suite (~5 seconds).")

    st.divider()

    # ---- 5. NL narrative coverage ----
    st.markdown("### 5. Natural-language narrative coverage")
    st.caption(
        "The rule-based NL generator runs on every user-day. The 'typical day' "
        "fallback rate is a calibration metric - 30-50% is healthy. Higher means "
        "rules are too strict; lower means too sensitive."
    )
    nl = get_narrative_coverage()
    if nl.get("total", 0) > 0:
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Total narratives", f"{nl['total']:,}")
        n2.metric("'Typical day' fallback", f"{nl['typical_pct']:.1f}%")
        n3.metric("Mention sleep/night", f"{nl['with_sleep']:,}")
        n4.metric("Mention HRV", f"{nl['with_hrv']:,}")

        n5, n6 = st.columns(2)
        n5.metric("Mention activity/steps", f"{nl['with_steps']:,}")
        n6.metric("Mention mood self-report", f"{nl['with_mood']:,}")

    st.divider()

    # ---- 6. URL grounding ----
    st.markdown("### 6. URL grounding (citation hallucination check)")
    st.caption(
        "For every saved plan, every URL cited is checked against URLs in our "
        "KB and SNP catalogue. 100% means no hallucinated citations; lower means "
        "the LLM invented sources (a known small-model failure mode)."
    )
    url = get_url_grounding()
    if url.get("n_plans", 0) > 0:
        u1, u2, u3, u4 = st.columns(4)
        u1.metric("Plans inspected", url["n_plans"])
        u2.metric("URLs cited (total)", url["total_urls"])
        u3.metric("URLs in our KB", url["urls_grounded"])
        pct = url["pct"]
        u4.metric("Grounding %", f"{pct:.1f}%" if pct is not None else "-")
    else:
        st.info(
            "No plans saved yet. Generate a plan from the **💬 Chat** tab "
            "to populate this metric."
        )

    st.divider()

    # ---- 7. Honest gaps ----
    st.markdown("### 7. Honest gaps (v2 evaluation roadmap)")
    st.caption(
        "Production health AI needs more than this. The Google PHA paper used "
        "7,000+ human annotations across 10 benchmark tasks. We're far below "
        "that - here's the v2 roadmap:"
    )
    gaps = [
        ("Plan-quality rubric",
         "LLM-as-judge on specificity, evidence-grounding, gene-aware reasoning. "
         "Score every plan, track over model versions."),
        ("Claim-level hallucination check",
         "Beyond URL grounding: extract every factual claim from a plan, verify "
         "against KB and the user's actual data."),
        ("RAG quality (precision@k, MRR, NDCG)",
         "Tune the retrieval layer with relevance-labeled ground truth - which "
         "snippets *should* surface for which queries."),
        ("Tool-call efficiency",
         "Avg tool calls per intent, % redundant calls, per-agent token cost. "
         "Useful for both quality and latency."),
        ("End-to-end latency p50/p95",
         "Per-intent response time distributions across providers."),
        ("Intent classifier accuracy",
         "Held-out labeled set, confusion matrix. Surface where the classifier "
         "drifts (today, eyeball-only)."),
        ("A/B testing across models",
         "Same query through llama-8b vs gpt-oss-120b vs qwen-235b, "
         "expert-graded blind."),
        ("Human evaluation cohort",
         "Following the Google PHA paper's 7,000+ annotation methodology - "
         "real users, real clinicians, real rubrics."),
    ]
    for title, desc in gaps:
        st.markdown(f"- **{title}** &nbsp;·&nbsp; {desc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    user_id = render_sidebar()

    st.markdown(HERO_HTML, unsafe_allow_html=True)

    tab_dashboard, tab_genes, tab_chat, tab_plans, tab_eval = st.tabs([
        "📊 Dashboard",
        "🧬 Genetic profile",
        "💬 Chat",
        "📋 Plans",
        "📈 Evaluation",
    ])
    with tab_dashboard:
        render_dashboard_tab(user_id)
    with tab_genes:
        render_genes_tab(user_id)
    with tab_chat:
        render_chat_tab(user_id)
    with tab_plans:
        render_plans_tab(user_id)
    with tab_eval:
        render_eval_tab()


if __name__ == "__main__":
    main()
