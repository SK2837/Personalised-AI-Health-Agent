"""
PHAI - LLM provider abstraction.

Two public entry points:

    chat(messages, ...) -> str
        Simple text-in / text-out. Used by code that doesn't need tools.

    complete(messages, ..., tools=...) -> dict
        Full chat completion. Returns the assistant message dict (ready to
        re-append to the conversation), plus parsed tool_calls. Used by
        agents that need OpenAI-compatible function calling.

Provider chosen via the LLM_PROVIDER env var (groq | gemini).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Recovery from Llama-on-Groq malformed tool calls.
#
# Llama 3.x sometimes emits tool calls in a custom XML-like syntax instead
# of the structured tool_calls JSON Groq expects, e.g.:
#     <function=get_user_genes {"user_id": "abc"}></function>
# When that happens Groq returns a 400 BadRequestError with the
# malformed generation in `failed_generation`. We parse it back into a
# proper tool call and resume the agent loop.
# ---------------------------------------------------------------------------

_MALFORMED_TOOL_CALL_PATTERNS = [
    # <function=name {args}></function>   (most common)
    re.compile(r"<function=(\w+)\s+(\{.*?\})\s*>(?:</function>)?", re.DOTALL),
    # <function=name>{args}</function>
    re.compile(r"<function=(\w+)>\s*(\{.*?\})\s*</function>", re.DOTALL),
    # function_call: name({args})
    re.compile(r"(\w+)\s*\(\s*(\{.*?\})\s*\)", re.DOTALL),
]


def _try_recover_tool_call(failed_gen: str) -> dict | None:
    """Parse a malformed tool call into our standard response dict, or None."""
    if not failed_gen:
        return None
    for pattern in _MALFORMED_TOOL_CALL_PATTERNS:
        match = pattern.search(failed_gen)
        if not match:
            continue
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        tc_id = f"recovered_{int(time.time() * 1000)}"
        return {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            },
            "content": None,
            "tool_calls": [{"id": tc_id, "name": name, "arguments": args}],
            "finish_reason": "tool_calls",
        }
    return None


# ---------------------------------------------------------------------------
# Public: simple chat (no tools)
# ---------------------------------------------------------------------------

def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Send a chat completion and return the assistant's text reply."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    model = model or os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    if provider == "groq":
        return _chat_groq(messages, model, temperature, max_tokens, json_mode)
    if provider == "gemini":
        return _chat_gemini(messages, model, temperature, max_tokens, json_mode)
    if provider == "cerebras":
        return _chat_cerebras(messages, model, temperature, max_tokens, json_mode)
    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. "
        f"Use 'groq', 'gemini', or 'cerebras'."
    )


# ---------------------------------------------------------------------------
# Public: full chat completion (tools, finish_reason)
# ---------------------------------------------------------------------------

def complete(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict] | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> dict:
    """
    Chat completion with optional tool use. Returns:

        {
            "message":      dict,        # assistant message ready to re-append
            "content":      str | None,  # text reply (may be None if only tools)
            "tool_calls":   list[dict],  # parsed: [{id, name, arguments}]
            "finish_reason": str,
        }

    Currently only Groq is supported for tool calling.
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    model = model or os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

    if provider == "gemini":
        return _complete_gemini(messages, model, temperature, max_tokens, tools)

    if provider == "cerebras":
        return _complete_cerebras(messages, model, temperature, max_tokens, tools)

    if provider != "groq":
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            f"Use 'groq', 'gemini', or 'cerebras'."
        )

    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")

    client = Groq(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Recover from Llama-on-Groq malformed tool-call output.
        if tools and "tool_use_failed" in str(e):
            failed_gen = ""
            try:
                if hasattr(e, "body") and isinstance(e.body, dict):
                    failed_gen = e.body.get("error", {}).get("failed_generation", "")
            except Exception:
                pass
            recovered = _try_recover_tool_call(failed_gen)
            if recovered is not None:
                return recovered
        raise

    choice = response.choices[0]
    msg = choice.message

    # Parse tool calls into a friendly shape.
    parsed_tool_calls: list[dict] = []
    raw_tool_calls: list[dict] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            parsed_tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "arguments": args}
            )
            raw_tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

    # Build the assistant message dict in the format the API expects on re-send.
    message_dict: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if raw_tool_calls:
        message_dict["tool_calls"] = raw_tool_calls

    return {
        "message": message_dict,
        "content": msg.content,
        "tool_calls": parsed_tool_calls,
        "finish_reason": choice.finish_reason,
    }


# ---------------------------------------------------------------------------
# Internal: provider implementations for plain chat()
# ---------------------------------------------------------------------------

def _chat_groq(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    client = Groq(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _chat_cerebras(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    """Cerebras simple chat. SDK is OpenAI-compatible."""
    from cerebras.cloud.sdk import Cerebras

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set in .env")

    client = Cerebras(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def _complete_cerebras(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    tools: list[dict] | None,
) -> dict:
    """Cerebras tool-calling. Returns the same shape as Groq."""
    from cerebras.cloud.sdk import Cerebras

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is not set in .env")

    client = Cerebras(api_key=api_key)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    msg = choice.message

    parsed_tool_calls: list[dict] = []
    raw_tool_calls: list[dict] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            parsed_tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "arguments": args}
            )
            raw_tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
            )

    message_dict: dict[str, Any] = {"role": "assistant", "content": msg.content}
    if raw_tool_calls:
        message_dict["tool_calls"] = raw_tool_calls

    return {
        "message": message_dict,
        "content": msg.content,
        "tool_calls": parsed_tool_calls,
        "finish_reason": choice.finish_reason,
    }


def _chat_gemini(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    genai.configure(api_key=api_key)
    flat = "\n".join(f"[{m['role'].upper()}] {m['content']}" for m in messages)
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    gem_model = genai.GenerativeModel(
        model_name=model, generation_config=generation_config
    )
    response = _gemini_generate_with_retry(gem_model, flat)
    return response.text or ""


# ---------------------------------------------------------------------------
# Gemini tool-calling implementation + converters
# ---------------------------------------------------------------------------
#
# Gemini's tool format differs from OpenAI/Groq in a few ways:
#   - JSON Schema "type" values are UPPERCASE (STRING, OBJECT, ...).
#   - Roles are "user" and "model" (no "assistant", no "tool").
#   - Tool calls are returned as `function_call` parts; tool results are
#     fed back as `function_response` parts.
#   - Tool responses are looked up by FUNCTION NAME, not by tool_call_id.
#
# We translate at the boundary so the rest of PHAI keeps working with the
# OpenAI-style convention.
# ---------------------------------------------------------------------------

def _gemini_generate_with_retry(gem_model, contents, max_retries: int = 3):
    """Call Gemini's generate_content with 429-aware retry. Sleeps the
    `retry_delay` Gemini suggests (capped at 60s), retries up to N times."""
    from google.api_core.exceptions import ResourceExhausted

    for attempt in range(max_retries):
        try:
            return gem_model.generate_content(contents)
        except ResourceExhausted as e:
            wait = 15
            m = re.search(r"seconds:\s*(\d+)", str(e))
            if m:
                wait = min(int(m.group(1)) + 1, 60)
            if attempt < max_retries - 1:
                print(
                    f"  [gemini] 429 rate limit, sleeping {wait}s "
                    f"(retry {attempt + 1}/{max_retries - 1})..."
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini returned no response after all retries.")


def _proto_to_native(obj: Any) -> Any:
    """Recursively convert protobuf MapComposite / RepeatedComposite values
    into plain Python dict / list / scalar so json.dumps works."""
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    # dict-like (MapComposite, dict): test for items() first.
    if hasattr(obj, "items") and callable(getattr(obj, "items", None)):
        return {str(k): _proto_to_native(v) for k, v in obj.items()}
    # list-like (RepeatedComposite, list, tuple).
    if hasattr(obj, "__iter__"):
        return [_proto_to_native(item) for item in obj]
    return obj


def _convert_schema_to_gemini(schema: Any) -> Any:
    """Recursively uppercase 'type' values for Gemini's JSON Schema variant."""
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for k, v in schema.items():
        if k == "type" and isinstance(v, str):
            out[k] = v.upper()
        elif isinstance(v, dict):
            out[k] = _convert_schema_to_gemini(v)
        elif isinstance(v, list):
            out[k] = [
                _convert_schema_to_gemini(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            out[k] = v
    return out


def _convert_tools_to_gemini(openai_tools: list[dict]) -> list[dict]:
    """OpenAI/Groq tool specs -> Gemini's function_declarations format."""
    declarations: list[dict] = []
    for t in openai_tools:
        if t.get("type") != "function":
            continue
        fn = t["function"]
        declarations.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": _convert_schema_to_gemini(fn.get("parameters", {})),
            }
        )
    return [{"function_declarations": declarations}]


def _convert_messages_to_gemini(
    messages: list[dict],
) -> tuple[str | None, list[dict]]:
    """
    Translate the OpenAI-style message list into Gemini's (system_instruction,
    contents) pair.

    Tool messages must be matched to their original function name; we build
    a tool_call_id -> name map in a forward pass first.
    """
    tc_id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id_to_name[tc["id"]] = tc["function"]["name"]

    system_instruction: str | None = None
    contents: list[dict] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""

        if role == "system":
            # Concatenate multiple system messages if any.
            system_instruction = (
                content if not system_instruction else f"{system_instruction}\n{content}"
            )
            continue

        if role == "tool":
            # Tool result coming back from our agent loop.
            fn_name = tc_id_to_name.get(msg.get("tool_call_id", ""), "unknown_function")
            try:
                result_obj = json.loads(content) if isinstance(content, str) else content
            except (json.JSONDecodeError, TypeError):
                result_obj = {"raw": content}
            if not isinstance(result_obj, dict):
                result_obj = {"value": result_obj}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": fn_name,
                                "response": result_obj,
                            }
                        }
                    ],
                }
            )
            continue

        if role == "assistant":
            parts: list[dict] = []
            if content:
                parts.append({"text": content})
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                parts.append({"function_call": {"name": fn["name"], "args": args}})
            if not parts:
                parts = [{"text": ""}]
            contents.append({"role": "model", "parts": parts})
            continue

        # Default: treat as user.
        contents.append({"role": "user", "parts": [{"text": content}]})

    return system_instruction, contents


def _complete_gemini(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    tools: list[dict] | None,
) -> dict:
    """Gemini implementation of complete(), supporting tool calls."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    genai.configure(api_key=api_key)

    system_instruction, contents = _convert_messages_to_gemini(messages)

    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    gem_model = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=system_instruction,
        tools=_convert_tools_to_gemini(tools) if tools else None,
    )

    response = _gemini_generate_with_retry(gem_model, contents)
    candidate = response.candidates[0]

    text_parts: list[str] = []
    parsed_tool_calls: list[dict] = []
    raw_tool_calls: list[dict] = []

    for part in candidate.content.parts:
        # Text part?
        text = getattr(part, "text", None)
        if text:
            text_parts.append(text)
        # Function-call part?
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None):
            args = _proto_to_native(fc.args) if fc.args else {}
            if not isinstance(args, dict):
                args = {}
            tc_id = f"gem_{fc.name}_{int(time.time() * 1000)}"
            parsed_tool_calls.append(
                {"id": tc_id, "name": fc.name, "arguments": args}
            )
            raw_tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": fc.name, "arguments": json.dumps(args)},
                }
            )

    text_content = "".join(text_parts) if text_parts else None
    message_dict: dict[str, Any] = {"role": "assistant", "content": text_content}
    if raw_tool_calls:
        message_dict["tool_calls"] = raw_tool_calls

    finish_reason = "stop"
    fr = getattr(candidate, "finish_reason", None)
    if fr is not None and hasattr(fr, "name"):
        finish_reason = fr.name.lower()

    return {
        "message": message_dict,
        "content": text_content,
        "tool_calls": parsed_tool_calls,
        "finish_reason": finish_reason,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Provider: {os.environ.get('LLM_PROVIDER', 'groq')}")
    print(f"Model:    {os.environ.get('LLM_MODEL', 'llama-3.3-70b-versatile')}")
    print()
    reply = chat(
        [
            {"role": "system", "content": "You are PHAI. Reply in one short sentence."},
            {"role": "user", "content": "Confirm you can analyse wearable data."},
        ],
        max_tokens=80,
    )
    print("--- llm.chat() reply ---")
    print(reply)
    print("------------------------")
    print("\nllm.py self-test passed.")
