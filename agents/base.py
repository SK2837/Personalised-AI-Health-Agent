"""
PHAI - shared agent infrastructure.

`run_agent_loop()` is the standard tool-calling loop reused by every PHAI
agent (Data Science, Domain Expert, Health Coach, Orchestrator). It:

  1. Sends [system, user] messages plus tool specs to the LLM.
  2. If the LLM responds with tool_calls, executes each one and appends
     the results back into the conversation.
  3. Loops until the LLM returns a final answer (no more tool calls)
     or hits max_iterations.
  4. Returns the final answer + a full trace of every step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm import complete  # noqa: E402


def run_agent_loop(
    system_prompt: str,
    user_message: str,
    tool_specs: list[dict],
    tool_executors: dict[str, Callable[..., Any]],
    *,
    max_iterations: int = 5,
    temperature: float = 0.3,
    on_step: Callable[[str, dict], None] | None = None,
) -> dict:
    """
    Run a tool-calling agent loop and return:

        {
            "answer":     str,        # final text reply from the LLM
            "trace":      list[dict], # ordered list of tool calls + final
            "iterations": int,        # how many LLM round-trips happened
        }

    `on_step(event_type, data)` is an optional callback for streaming
    visibility into the loop (used by the Streamlit UI later).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    trace: list[dict] = []

    for i in range(max_iterations):
        response = complete(messages, tools=tool_specs, temperature=temperature)
        # Always re-append the assistant message (with or without tool_calls).
        messages.append(response["message"])

        # Final answer? Done.
        if not response["tool_calls"]:
            trace.append({
                "step": i + 1,
                "type": "final_answer",
                "content": response["content"],
            })
            if on_step:
                on_step("final_answer", {"content": response["content"]})
            return {
                "answer": response["content"] or "",
                "trace": trace,
                "iterations": i + 1,
            }

        # Otherwise execute each tool call and append the result.
        for tc in response["tool_calls"]:
            name = tc["name"]
            args = tc["arguments"]
            executor = tool_executors.get(name)
            if executor is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = executor(**args)
                except Exception as e:  # surface to LLM for self-correction
                    result = {"error": f"{type(e).__name__}: {e}"}

            trace.append({
                "step": i + 1,
                "type": "tool_call",
                "name": name,
                "args": args,
                "result": result,
            })
            if on_step:
                on_step("tool_call", {"name": name, "args": args, "result": result})

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, default=str),
            })

    # Hit the safety cap.
    return {
        "answer": (
            "I reached the iteration limit while gathering data. "
            "Here's what I learned so far - please ask a more specific question."
        ),
        "trace": trace,
        "iterations": max_iterations,
    }
