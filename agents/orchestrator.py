"""
PHAI - Orchestrator agent.

The manager that sits in front of the three sub-agents. It:
  1. Classifies the user's intent (one cheap LLM call).
  2. Routes to the right sub-agent(s):
        data_question      -> DS only
        knowledge_question -> DE only
        plan_request       -> DS -> DE -> HC (headline pipeline)
        mixed              -> DS + DE then synthesise
        out_of_scope       -> polite refusal
  3. Passes findings between agents as context.
  4. Aggregates everything into one result dict the UI can render.

CLI test:
    python -m agents.orchestrator
    python -m agents.orchestrator --query "I want to feel more energetic this week"
    python -m agents.orchestrator --user-id <id> --query "..."
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm import chat  # noqa: E402

from agents import data_science, domain_expert, health_coach  # noqa: E402

DB_PATH = PROJECT_ROOT / "phai.db"

VALID_INTENTS = {
    "data_question",
    "knowledge_question",
    "plan_request",
    "mixed",
    "medical_concern",
    "out_of_scope",
}


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

INTENT_PROMPT = """You are PHAI's intent classifier. PHAI is a personalised health AI
covering sleep, exercise, nutrition, stress, recovery, energy, mood, genetics,
and wearable data.

Classify the user's question into EXACTLY ONE intent label:

1. data_question - the user's own wearable metrics, trends, cohort comparisons
2. knowledge_question - genes, biology, or health science (with or without user context)
3. plan_request - asking for a personalised plan, recommendation, or what-to-do
4. mixed - needs BOTH user data AND biological grounding for one coherent answer
5. medical_concern - symptom, pain, injury, condition, diagnosis, medication/dose
6. out_of_scope - clearly NOT health-related (math, trivia, creative, weather, etc.)

THESE ARE HEALTH QUESTIONS (use 1-4, NEVER out_of_scope):
  "What does my CYP1A2 result mean for caffeine?"  -> knowledge_question
  "Why have I been sluggish this week?"            -> plan_request
  "How was my sleep last week?"                    -> data_question
  "Am I tired because of my genes?"                -> mixed
  "Help me sleep better"                           -> plan_request
  "Compared to others, am I active?"               -> data_question
  "Explain MTHFR"                                  -> knowledge_question
  "Build me a plan to feel energetic"              -> plan_request
  "What's my HRV trend?"                           -> data_question
  "What should I do about caffeine?"               -> plan_request
  "Tell me about my chronotype"                    -> knowledge_question

USE medical_concern ONLY for symptoms / diagnoses / medication:
  "I have blurred vision"        "my chest hurts"
  "do I have diabetes?"          "is this cancer?"
  "what dose of melatonin?"      "I think I'm depressed"
  "my heart is racing"           "I have a headache"

USE out_of_scope ONLY for clearly non-health queries:
  "factorial of 5"     "who is Galileo"     "write a poem"
  "what's the weather" "tell me a joke"     "suggest grocery stores"
  "explain quantum physics"   "how do I write Python"

If the question relates to ANYTHING about the user's wellness, biology, sleep,
exercise, nutrition, stress, recovery, mood, energy, weight, fitness, or
genetics, it is health-related and goes to an agent (1-4), NOT out_of_scope.

Respond with ONLY one of these exact words:
data_question, knowledge_question, plan_request, mixed, medical_concern, out_of_scope"""


def classify_intent(query: str) -> str:
    """Return one of VALID_INTENTS. Falls back to plan_request on any
    parsing miss (the most useful default for legitimate health queries;
    out_of_scope and medical_concern only trigger when the classifier is
    explicitly confident in those labels)."""
    raw = chat(
        [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=20,
    )
    if not raw:
        return "plan_request"

    # Robust parsing: strip punctuation/quotes, lowercase, scan tokens.
    cleaned = raw.lower()
    for ch in "`.,'\"!?":
        cleaned = cleaned.replace(ch, " ")
    for token in cleaned.split():
        token = token.strip(":-_*()[]{}")
        if token in VALID_INTENTS:
            return token

    # No valid label found in the response - assume legitimate health query.
    return "plan_request"


# ---------------------------------------------------------------------------
# Synthesiser (used by `mixed`)
# ---------------------------------------------------------------------------

SYNTHESISE_PROMPT = (
    "You are PHAI's synthesiser. You receive (a) a data analysis from "
    "the Data Science Agent and (b) a knowledge grounding from the "
    "Domain Expert Agent. Produce ONE coherent 4-6 sentence answer to "
    "the user's question. Cite specific numbers from the data analysis "
    "and any URLs from the knowledge grounding. Plain English, no "
    "bullet lists, no medical advice."
)


def synthesise_answer(query: str, ds_answer: str, de_answer: str) -> str:
    user_msg = (
        f"User question: {query}\n\n"
        f"Data analysis findings:\n{ds_answer}\n\n"
        f"Knowledge grounding:\n{de_answer}\n\n"
        f"Write the final synthesised answer."
    )
    return chat(
        [
            {"role": "system", "content": SYNTHESISE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=600,
    )


# ---------------------------------------------------------------------------
# Helper: wrap an on_step callback to tag its events with the agent name
# ---------------------------------------------------------------------------

def _wrap_on_step(parent_on_step, agent_name: str):
    if parent_on_step is None:
        return None

    def wrapped(event_type, data):
        parent_on_step(event_type, {**data, "agent": agent_name})

    return wrapped


def _emit(on_step, event_type: str, data: dict) -> None:
    if on_step:
        on_step(event_type, data)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(user_id: str, query: str, *, on_step=None) -> dict:
    """
    Run the full orchestrated pipeline.

    Returns:
        {
            "answer": str,                    # user-facing reply
            "intent": str,                    # classified intent
            "agents_run": list[str],          # which sub-agents fired
            "trace_by_agent": dict[str, list],# per-agent traces
            "plan_id": int | None,            # populated for plan_request
            "plan": dict | None,              # structured plan dict
        }
    """
    intent = classify_intent(query)
    _emit(on_step, "intent_classified", {"intent": intent, "query": query})

    base_result = {
        "intent": intent,
        "agents_run": [],
        "trace_by_agent": {},
        "plan_id": None,
        "plan": None,
    }

    # --- out_of_scope ------------------------------------------------------
    if intent == "out_of_scope":
        return {
            **base_result,
            "answer": (
                "I'm focused on personalised health insights from your "
                "wearable and gene data. Try asking me about your sleep, "
                "recovery, energy, stress, or a specific change you'd "
                "like to make."
            ),
        }

    # --- medical_concern (safety guardrail) --------------------------------
    if intent == "medical_concern":
        return {
            **base_result,
            "answer": (
                "It sounds like you're describing a medical symptom or "
                "concern. **PHAI is a wellness assistant, not a medical "
                "device** — I'm not able to diagnose conditions, interpret "
                "symptoms, or recommend treatment. Please speak with a "
                "qualified healthcare professional about this; many of the "
                "things I'd worry about (vision changes, chest discomfort, "
                "new pains, persistent fatigue) deserve a clinician's eye.\n\n"
                "If you'd like, I can still help with general lifestyle "
                "questions — sleep habits, stress management, nutrition "
                "patterns, or how your genetics shape day-to-day choices."
            ),
        }

    # --- data_question -> DS only ------------------------------------------
    if intent == "data_question":
        _emit(on_step, "agent_start", {"agent": "data_science"})
        ds = data_science.run(
            user_id, query, on_step=_wrap_on_step(on_step, "data_science")
        )
        _emit(on_step, "agent_end", {"agent": "data_science", "iterations": ds["iterations"]})
        return {
            **base_result,
            "answer": ds["answer"],
            "agents_run": ["data_science"],
            "trace_by_agent": {"data_science": ds["trace"]},
        }

    # --- knowledge_question -> DE only -------------------------------------
    if intent == "knowledge_question":
        _emit(on_step, "agent_start", {"agent": "domain_expert"})
        de = domain_expert.run(
            user_id, query, on_step=_wrap_on_step(on_step, "domain_expert")
        )
        _emit(on_step, "agent_end", {"agent": "domain_expert", "iterations": de["iterations"]})
        return {
            **base_result,
            "answer": de["answer"],
            "agents_run": ["domain_expert"],
            "trace_by_agent": {"domain_expert": de["trace"]},
        }

    # --- mixed -> DS + DE then synthesise ----------------------------------
    if intent == "mixed":
        _emit(on_step, "agent_start", {"agent": "data_science"})
        ds = data_science.run(
            user_id, query, on_step=_wrap_on_step(on_step, "data_science")
        )
        _emit(on_step, "agent_end", {"agent": "data_science", "iterations": ds["iterations"]})

        _emit(on_step, "agent_start", {"agent": "domain_expert"})
        de = domain_expert.run(
            user_id, query, on_step=_wrap_on_step(on_step, "domain_expert")
        )
        _emit(on_step, "agent_end", {"agent": "domain_expert", "iterations": de["iterations"]})

        _emit(on_step, "synthesise_start", {})
        final = synthesise_answer(query, ds["answer"], de["answer"])
        _emit(on_step, "synthesise_end", {})

        return {
            **base_result,
            "answer": final,
            "agents_run": ["data_science", "domain_expert"],
            "trace_by_agent": {
                "data_science": ds["trace"],
                "domain_expert": de["trace"],
            },
        }

    # --- plan_request -> DS -> DE -> HC (headline pipeline) ----------------
    if intent == "plan_request":
        _emit(on_step, "agent_start", {"agent": "data_science"})
        ds = data_science.run(
            user_id, query, on_step=_wrap_on_step(on_step, "data_science")
        )
        _emit(on_step, "agent_end", {"agent": "data_science", "iterations": ds["iterations"]})

        _emit(on_step, "agent_start", {"agent": "domain_expert"})
        de = domain_expert.run(
            user_id, query, on_step=_wrap_on_step(on_step, "domain_expert")
        )
        _emit(on_step, "agent_end", {"agent": "domain_expert", "iterations": de["iterations"]})

        # Hand DS+DE findings to the Coach as extra_context.
        context = (
            f"Data Science Agent findings:\n{ds['answer']}\n\n"
            f"Domain Expert Agent findings:\n{de['answer']}"
        )
        _emit(on_step, "agent_start", {"agent": "health_coach"})
        hc = health_coach.run(
            user_id,
            query,
            extra_context=context,
            on_step=_wrap_on_step(on_step, "health_coach"),
        )
        _emit(on_step, "agent_end", {"agent": "health_coach", "iterations": hc["iterations"]})

        return {
            **base_result,
            "answer": hc["answer"],
            "agents_run": ["data_science", "domain_expert", "health_coach"],
            "trace_by_agent": {
                "data_science": ds["trace"],
                "domain_expert": de["trace"],
                "health_coach": hc["trace"],
            },
            "plan_id": hc.get("plan_id"),
            "plan": hc.get("plan"),
        }

    # Should be unreachable due to fallback in classify_intent.
    raise RuntimeError(f"Unhandled intent: {intent}")


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


def _print_event(event_type: str, data: dict) -> None:
    agent = data.get("agent")
    if event_type == "intent_classified":
        print(f"\n[orchestrator] intent = {data['intent']}")
    elif event_type == "agent_start":
        print(f"\n--- {agent} agent ---")
    elif event_type == "agent_end":
        print(f"--- {agent} done in {data.get('iterations')} iterations ---")
    elif event_type == "tool_call":
        result_preview = str(data["result"])[:120].replace("\n", " ")
        print(
            f"  [{agent}] -> {data['name']}({list(data['args'].keys())})\n"
            f"     => {result_preview}{'...' if len(str(data['result'])) > 120 else ''}"
        )
    elif event_type == "synthesise_start":
        print("\n--- orchestrator synthesising final answer ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the PHAI Orchestrator.")
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--query",
        default="I want to feel more energetic this week.",
    )
    args = parser.parse_args()

    user_id = args.user_id or _pick_demo_user()
    print(f"Test user: {user_id}")
    print(f"Question:  {args.query}")

    result = run(user_id, args.query, on_step=_print_event)

    print("\n" + "=" * 70)
    print(f"INTENT:       {result['intent']}")
    print(f"AGENTS RUN:   {' -> '.join(result['agents_run']) or '(none)'}")
    print("=" * 70)
    print("\nFINAL ANSWER:\n")
    print(result["answer"])

    if result.get("plan"):
        print("\n" + "=" * 70)
        print(f"STRUCTURED PLAN (plan_id={result['plan_id']})")
        print("=" * 70)
        plan = result["plan"]
        print(f"\nGoal: {plan.get('goal')}")
        print(f"Why:  {plan.get('why')}")
        print("\nSteps:")
        for i, step in enumerate(plan.get("steps", []), 1):
            print(f"  {i}. {step.get('title')}")
            print(f"     {step.get('detail')}")
            if step.get("evidence"):
                print(f"     Evidence: {step['evidence']}")
        if plan.get("metrics_to_track"):
            print(f"\nMetrics to track: {', '.join(plan['metrics_to_track'])}")
        print(f"Check-in: {plan.get('check_in_days', 7)} days")
    print()


if __name__ == "__main__":
    main()
