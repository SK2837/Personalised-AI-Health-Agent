"""
PHAI - Data Science Agent.

Job: take a user's question + their user_id, fetch the relevant wearable
data via tools, and produce a concise data-grounded analysis.

Run from the project root for a quick CLI test:
    python -m agents.data_science
    python -m agents.data_science --query "How has my sleep been?"
    python -m agents.data_science --user-id <id> --query "..."
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.base import run_agent_loop  # noqa: E402
from agents.tools import (  # noqa: E402
    ALLOWED_METRICS,
    compare_to_population,
    compute_correlation,
    get_recent_summary,
    get_user_profile,
    predict_energy_tomorrow,
)

DB_PATH = PROJECT_ROOT / "phai.db"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the Data Science Agent for PHAI, a personalised health AI.\n\n"
    "Your job: answer the user's question with a concise, data-grounded "
    "analysis using the tools available. You analyse wearable data (sleep, "
    "heart rate, HRV, stress, steps, mood self-reports). You do NOT diagnose, "
    "prescribe, or give medical advice.\n\n"
    "Tool budget: AT MOST 4 tool calls total. Be selective:\n"
    "- One get_recent_summary call gives you all metric averages at once.\n"
    "- Use compare_to_population for AT MOST 2 metrics directly relevant "
    "  to the question.\n"
    "- Use compute_correlation only if the question is about why one metric "
    "  affects another.\n"
    "- Use predict_energy_tomorrow ONLY if the question is about future "
    "  energy, tomorrow's plan, or readiness to push hard tomorrow.\n"
    "- After 4 calls, STOP and answer with what you have.\n\n"
    "Other rules:\n"
    "- ALWAYS call a tool before quoting any number. Never guess.\n"
    "- Final answer: 3-5 sentences. Plain English. No bullet lists.\n"
    "- If a metric has < 5 days of data, say so and avoid over-interpreting.\n\n"
    f"Available metrics: {', '.join(sorted(ALLOWED_METRICS))}"
)


# ---------------------------------------------------------------------------
# Tool specs (OpenAI / Groq function-calling format)
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": (
                "Get the user's basic profile (age, gender, BMI, source, "
                "days of data, gene panel size). Call once early."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The user's ID."},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_summary",
            "description": (
                "Get per-metric averages over the user's most recent N days "
                "plus the most recent natural-language narratives. Use this "
                "to understand current state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "days": {
                        "type": "integer",
                        "description": "Number of recent days (1-90). Default 30.",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_to_population",
            "description": (
                "Compare the user's average for one metric to the 1000-user "
                "cohort. Returns user_percentile_in_cohort and quartile cuts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "metric": {
                        "type": "string",
                        "description": "Metric name from the allowed list.",
                    },
                },
                "required": ["user_id", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Pearson correlation between two metrics for one user across "
                "their daily data. Use to find what predicts what (e.g. does "
                "low HRV predict tired-mood-reports?)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "metric_a": {"type": "string"},
                    "metric_b": {"type": "string"},
                },
                "required": ["user_id", "metric_a", "metric_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_energy_tomorrow",
            "description": (
                "Use the trained XGBoost classifier to predict tomorrow's "
                "energy level for the user. Returns the probability that "
                "tomorrow's step count will be above the user's personal "
                "median (an 'above-baseline activity day'). Use ONLY when "
                "the user asks about tomorrow, future energy, readiness, "
                "or whether to push hard tomorrow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                },
                "required": ["user_id"],
            },
        },
    },
]

TOOL_EXECUTORS = {
    "get_user_profile": get_user_profile,
    "get_recent_summary": get_recent_summary,
    "compare_to_population": compare_to_population,
    "compute_correlation": compute_correlation,
    "predict_energy_tomorrow": predict_energy_tomorrow,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(user_id: str, query: str, *, on_step=None, max_iterations: int = 6) -> dict:
    """
    Run the Data Science Agent. Returns {answer, trace, iterations}.
    """
    user_message = f"User ID: {user_id}\n\nUser question: {query}"
    return run_agent_loop(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        tool_specs=TOOL_SPECS,
        tool_executors=TOOL_EXECUTORS,
        max_iterations=max_iterations,
        temperature=0.3,
        on_step=on_step,
    )


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------

def _pick_demo_user() -> str:
    """Pick the LifeSnaps user with the most days of data - good demo subject."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT u.user_id "
            "FROM users u JOIN daily_summary d USING (user_id) "
            "WHERE u.source = 'lifesnaps' "
            "GROUP BY u.user_id "
            "ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
    if not row:
        raise RuntimeError("No LifeSnaps users in DB. Run ETL first.")
    return row[0]


def _format_trace_event(event_type: str, data: dict) -> str:
    if event_type == "tool_call":
        result_preview = str(data["result"])[:160].replace("\n", " ")
        return (
            f"  -> {data['name']}({data['args']})\n"
            f"     => {result_preview}{'...' if len(str(data['result'])) > 160 else ''}"
        )
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Data Science Agent.")
    parser.add_argument(
        "--user-id",
        default=None,
        help="A user_id from the database. If omitted, picks a demo user.",
    )
    parser.add_argument(
        "--query",
        default="How has my sleep and recovery been over the last 30 days?",
        help="Question to ask the agent.",
    )
    args = parser.parse_args()

    user_id = args.user_id or _pick_demo_user()
    print(f"Test user: {user_id}")
    print(f"Question:  {args.query}\n")
    print("--- Tool trace ---")

    def trace_print(event_type, data):
        line = _format_trace_event(event_type, data)
        if line:
            print(line)

    result = run(user_id, args.query, on_step=trace_print)

    print(f"\n--- Final answer ({result['iterations']} iterations) ---")
    print(result["answer"])
    print()


if __name__ == "__main__":
    main()
