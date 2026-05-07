"""
PHAI - Domain Expert Agent.

Job: ground answers in (a) the user's actual gene panel and (b) the curated
knowledge base. Every claim must trace to either a returned KB snippet (with
its URL) or the user's actual genotype (with its trait_summary). No
free-floating medical advice.

CLI test:
    python -m agents.domain_expert
    python -m agents.domain_expert --query "What does my CYP1A2 genotype mean?"
    python -m agents.domain_expert --user-id <id> --query "..."
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
    get_user_genes,
    kb_search,
    lookup_snp,
)

DB_PATH = PROJECT_ROOT / "phai.db"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the Domain Expert Agent for PHAI, a personalised health AI.\n\n"
    "Your job: answer the user's question by combining (a) their actual "
    "gene profile and (b) evidence retrieved from the curated knowledge "
    "base. You do NOT diagnose, prescribe, or give medical advice.\n\n"
    "Tool budget: AT MOST 3 tool calls total. Be selective:\n"
    "- get_user_genes(user_id): call ONCE up front when biology is relevant.\n"
    "- kb_search(query): call AT MOST 2 times, each with a focused query "
    "  and k=3 (default). Do NOT search the same topic with paraphrased "
    "  queries - one good search is enough.\n"
    "- lookup_snp(rsid): only if the user asks about a specific SNP.\n"
    "- After 3 calls, STOP and answer with what you have.\n\n"
    "Discipline:\n"
    "- ALWAYS gather evidence with tools before claiming anything.\n"
    "- Cite sources by URL or by gene+rsid. Format: '(source: <url>)' "
    "  or '(your CYP1A2 rs762551 result)'.\n"
    "- Do not invent citations. Only cite URLs returned by kb_search.\n"
    "- Be specific: 'as a slow CYP1A2 metaboliser (CC genotype), evidence "
    "  suggests...' rather than vague generalities.\n"
    "- Final answer: 3-6 sentences. Plain English. No bullet lists.\n"
)


# ---------------------------------------------------------------------------
# Tool specs (OpenAI / Groq function-calling format)
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_genes",
            "description": (
                "Get the user's complete 10-SNP profile - rsid, gene, "
                "their genotype, what that genotype means, trait summary, "
                "and lifestyle implications for each. Call this first when "
                "the user's specific biology is relevant."
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
    {
        "type": "function",
        "function": {
            "name": "lookup_snp",
            "description": (
                "Look up one SNP by rsid (e.g. 'rs762551'). Returns the "
                "trait summary, lifestyle implications, all genotype "
                "meanings, and the citation URL. Use for deeper dives on "
                "a specific gene the user asks about."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rsid": {
                        "type": "string",
                        "description": "rsid like 'rs762551'.",
                    },
                },
                "required": ["rsid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kb_search",
            "description": (
                "Semantic search of the curated knowledge base. Returns "
                "top-k snippets with text, topic, source, and URL. Use "
                "natural-language queries describing what the user wants "
                "to know about (e.g. 'morning daylight circadian rhythm', "
                "'high-protein meals satiety'). Use the URLs returned "
                "for citation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {
                        "type": "integer",
                        "description": "Number of snippets to retrieve (1-8). Default 3.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_EXECUTORS = {
    "get_user_genes": get_user_genes,
    "lookup_snp": lookup_snp,
    "kb_search": kb_search,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(user_id: str, query: str, *, on_step=None, max_iterations: int = 6) -> dict:
    """Run the Domain Expert Agent. Returns {answer, trace, iterations}."""
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
        result_preview = str(data["result"])[:180].replace("\n", " ")
        return (
            f"  -> {data['name']}({data['args']})\n"
            f"     => {result_preview}{'...' if len(str(data['result'])) > 180 else ''}"
        )
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Domain Expert Agent.")
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--query",
        default="Based on my genes, what should I know about caffeine and sleep?",
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
