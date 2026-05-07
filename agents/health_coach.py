"""
PHAI - Health Coach Agent.

Job: take the user's question + (optional) findings from the DS and DE
agents, and produce a personalised, evidence-grounded plan they can act on
this week. Style: motivational interviewing - warm, specific, autonomy-
respecting.

The Coach's headline tool is `submit_plan(...)`. After the agent loop
completes, run() extracts the submitted args from the trace and persists
the plan to the `plans` table.

CLI test:
    python -m agents.health_coach
    python -m agents.health_coach --query "I want to feel more energetic this week"
    python -m agents.health_coach --user-id <id> --query "..."
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.base import run_agent_loop  # noqa: E402
from agents.tools import (  # noqa: E402
    get_recent_summary,
    get_user_genes,
    kb_search,
    submit_plan,
)

DB_PATH = PROJECT_ROOT / "phai.db"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the Health Coach Agent for PHAI, a personalised health AI.\n\n"
    "Your job: turn the user's question into a personalised, "
    "evidence-grounded plan they can act on this week. Speak in "
    "motivational-interviewing style: warm, specific, autonomy-respecting. "
    "You do NOT diagnose, prescribe, or recommend supplements/doses.\n\n"
    "Tool budget: AT MOST 4 tool calls total (3 read + 1 submit_plan).\n"
    "If the user message includes 'Context from prior agents:', that "
    "context already contains the data analysis and the evidence - use "
    "it directly and do NOT re-fetch the same things. In that case, you "
    "can call submit_plan immediately, possibly with one optional "
    "kb_search if you need specific evidence for one step.\n\n"
    "Standalone workflow (when no prior context is provided):\n"
    "1. get_user_genes(user_id) - ONCE.\n"
    "2. get_recent_summary(user_id, 30) - ONCE.\n"
    "3. kb_search(query) - ONCE with a focused query.\n"
    "4. submit_plan(...) - call EXACTLY ONCE.\n"
    "5. End with one short friendly sentence.\n\n"
    "REQUIRED SPECIFICS (this is the most important rule):\n"
    "- The `why` field MUST cite at least ONE specific number from the "
    "  user's data (e.g. 'your sleep averages 6.5 h', 'your HRV is in "
    "  your bottom decile', 'your stress score of 72/100') AND at least "
    "  ONE specific gene+genotype the user carries (e.g. 'your CYP1A2 "
    "  AC genotype', 'your CLOCK CC night-owl tendency'). NO generic "
    "  rationales.\n"
    "- Each step's `evidence` field MUST reference EITHER a specific "
    "  user data point, OR the user's specific gene+genotype, OR a KB "
    "  source URL. 'Sleep is important' is NOT acceptable evidence.\n"
    "- Generic plans (e.g. 'establish a sleep schedule, avoid screens, "
    "  manage stress') are unacceptable. Every step must connect to "
    "  THIS user's data or biology.\n\n"
    "Plan principles:\n"
    "- 3-5 concrete steps. Quality over quantity.\n"
    "- Goal should reflect the user's own words from the question.\n"
    "- Tone: 'You might try X' not 'You should do X'.\n"
    "- user_message: 2-3 warm sentences that reference at least one "
    "  specific finding (number or gene) so the user feels seen.\n"
    "- metrics_to_track: pick 2-4 from sleep_min, sleep_efficiency, steps, "
    "  very_active_min, resting_hr, hrv_rmssd, stress_score, mood_tired, "
    "  mood_happy, mood_rested.\n"
    "- check_in_days: typically 7."
)


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_genes",
            "description": (
                "Get the user's complete gene panel with their specific "
                "genotype interpretations and lifestyle implications."
            ),
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_summary",
            "description": (
                "Get per-metric averages for the user's most recent N days "
                "and recent narratives. Tells you their current state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "days": {"type": "integer", "description": "Default 30."},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": (
                "Semantic KB search. Use to find evidence for specific "
                "recommendations (e.g. 'morning daylight circadian', "
                "'caffeine timing slow metaboliser')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "Default 3."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": (
                "Submit the FINAL structured plan. Call EXACTLY ONCE as "
                "your final tool action - after this, end with one short "
                "friendly sentence and no more tool calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "goal": {
                        "type": "string",
                        "description": (
                            "Concise goal in plain language, ideally "
                            "echoing the user's own framing."
                        ),
                    },
                    "why": {
                        "type": "string",
                        "description": (
                            "1-2 sentences explaining why this plan suits "
                            "THIS user, grounded in their data/biology."
                        ),
                    },
                    "steps": {
                        "type": "array",
                        "description": "3-5 concrete actions.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Short imperative.",
                                },
                                "detail": {
                                    "type": "string",
                                    "description": "1-2 sentences with specifics.",
                                },
                                "frequency": {
                                    "type": "string",
                                    "description": "e.g. 'daily', '3x/week'.",
                                },
                                "evidence": {
                                    "type": "string",
                                    "description": (
                                        "Link to user data, gene, or KB source."
                                    ),
                                },
                            },
                            "required": ["title", "detail"],
                        },
                    },
                    "metrics_to_track": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2-4 metric names to revisit at check-in.",
                    },
                    "check_in_days": {
                        "type": "integer",
                        "description": "When to revisit; default 7.",
                    },
                    "user_message": {
                        "type": "string",
                        "description": (
                            "Warm 2-3 sentence motivational-interviewing-"
                            "style message for the user."
                        ),
                    },
                },
                "required": ["user_id", "goal", "why", "steps", "user_message"],
            },
        },
    },
]

TOOL_EXECUTORS = {
    "get_user_genes": get_user_genes,
    "get_recent_summary": get_recent_summary,
    "kb_search": kb_search,
    "submit_plan": submit_plan,
}


# ---------------------------------------------------------------------------
# Persistence (called by run() after the loop)
# ---------------------------------------------------------------------------

def _extract_plan_from_trace(trace: list[dict]) -> dict | None:
    """Return the canonical plan from a successful submit_plan trace event.

    Prefers the coerced `result.plan` (validated by submit_plan) over the
    raw LLM args, since smaller models sometimes stringify nested arrays.
    """
    for event in trace:
        if event.get("type") == "tool_call" and event.get("name") == "submit_plan":
            result = event.get("result", {})
            if result.get("success"):
                return result.get("plan") or event.get("args")
    return None


def _save_plan(user_id: str, query: str, plan_args: dict) -> int:
    """Persist a finalized plan to the `plans` table. Returns plan_id."""
    plan_payload = {
        "goal": plan_args.get("goal"),
        "why": plan_args.get("why"),
        "steps": plan_args.get("steps", []),
        "metrics_to_track": plan_args.get("metrics_to_track", []),
        "check_in_days": plan_args.get("check_in_days", 7),
        "user_message": plan_args.get("user_message"),
    }
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO plans (user_id, query, plan_json, status) "
            "VALUES (?, ?, ?, 'active')",
            (user_id, query, json.dumps(plan_payload)),
        )
        conn.commit()
        return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    user_id: str,
    query: str,
    *,
    extra_context: str | None = None,
    on_step=None,
    max_iterations: int = 7,
) -> dict:
    """
    Run the Health Coach Agent. Returns:
        {answer, trace, iterations, plan_id, plan}

    `extra_context` is optional pre-computed evidence (e.g. from DS and DE
    agents when running through the Orchestrator). When provided, the
    Coach is encouraged to use it instead of re-fetching everything.
    """
    user_message = f"User ID: {user_id}\n\nUser question: {query}"
    if extra_context:
        user_message += f"\n\nContext from prior agents:\n{extra_context}"

    result = run_agent_loop(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        tool_specs=TOOL_SPECS,
        tool_executors=TOOL_EXECUTORS,
        max_iterations=max_iterations,
        temperature=0.4,  # slightly warmer for coaching tone
        on_step=on_step,
    )

    plan_args = _extract_plan_from_trace(result["trace"])
    if plan_args:
        plan_id = _save_plan(user_id, query, plan_args)
        result["plan_id"] = plan_id
        result["plan"] = plan_args
    else:
        result["plan_id"] = None
        result["plan"] = None

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pick_demo_user() -> str:
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
            f"  -> {data['name']}({list(data['args'].keys())})\n"
            f"     => {result_preview}{'...' if len(str(data['result'])) > 160 else ''}"
        )
    return ""


def _print_plan(plan: dict) -> None:
    print("\nGoal:")
    print(f"  {plan.get('goal')}")
    print("\nWhy:")
    print(f"  {plan.get('why')}")
    print("\nSteps:")
    for i, step in enumerate(plan.get("steps", []), 1):
        print(f"  {i}. {step.get('title')}")
        print(f"     {step.get('detail')}")
        if step.get("frequency"):
            print(f"     Frequency: {step['frequency']}")
        if step.get("evidence"):
            print(f"     Evidence: {step['evidence']}")
    if plan.get("metrics_to_track"):
        print("\nMetrics to track:")
        print(f"  {', '.join(plan['metrics_to_track'])}")
    print(f"\nCheck-in: {plan.get('check_in_days', 7)} days")
    print("\nMessage to user:")
    print(f"  {plan.get('user_message')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Health Coach Agent.")
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--query",
        default="I want to feel more energetic this week.",
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

    if result["plan"]:
        print(f"\n=== PLAN (saved as plan_id={result['plan_id']}) ===")
        _print_plan(result["plan"])
    else:
        print("\n[NO PLAN PERSISTED] - the Coach did not call submit_plan.")
    print()


if __name__ == "__main__":
    main()
