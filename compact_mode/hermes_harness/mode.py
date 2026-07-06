from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from compact_mode.hermes_harness.prompt import (
    CONTEXT_SUMMARY_PREFIX,
    MAX_OUTPUT_TOKENS,
    MAX_RENDERED_HISTORY_CHARS,
    MAX_SINGLE_FIELD_CHARS,
    choose_prompt,
)


class HermesHarnessModeError(ValueError):
    pass


@dataclass
class HermesHarnessMode:
    name: str = "hermes_harness"

    def build_chat_request(
        self,
        body: Any,
        client_path: str = "/chat/completions",
        model_override: str | None = None,
        language: str = "auto",
        disable_thinking: bool = True,
    ) -> dict[str, Any]:
        del client_path
        messages, request_model = _messages_from_body(body)
        model = model_override or request_model
        if not model:
            raise HermesHarnessModeError("compact request is missing model; pass --model to override")
        return self.build_chat_request_from_messages(
            messages,
            model=model,
            language=language,
            disable_thinking=disable_thinking,
        )

    def build_chat_request_from_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
        language: str = "auto",
        disable_thinking: bool = True,
    ) -> dict[str, Any]:
        if not messages:
            raise HermesHarnessModeError("cannot compact an empty message history")
        rendered_history = _truncate_middle(_render_hermes_turns(messages), MAX_RENDERED_HISTORY_CHARS)
        chat_request: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Agent conversation turns to summarize:\n\n" + rendered_history,
                },
                {"role": "user", "content": choose_prompt(language, rendered_history)},
            ],
            "stream": False,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        if disable_thinking and _looks_like_qwen_thinking_model(str(model)):
            chat_request["enable_thinking"] = False
        return chat_request

    def extract_summary_text(self, upstream_response: Any) -> str:
        text = extract_summary_text(upstream_response)
        if not text:
            preview = json.dumps(upstream_response, ensure_ascii=False)[:500]
            raise HermesHarnessModeError(f"upstream response does not contain summary text: {preview}")
        return text

    def wrap_compact_json(
        self,
        raw_summary: str,
        validate: bool = True,
        min_summary_chars: int = 300,
    ) -> dict[str, Any]:
        encrypted = compacted_content_from_raw(
            raw_summary,
            validate=validate,
            min_summary_chars=min_summary_chars,
        )
        return {"output": [{"type": "compaction", "encrypted_content": encrypted}]}

    def wrap_summary_as_sse(self, *args, **kwargs) -> str:
        raise NotImplementedError("hermes_harness mode is intended for main.py direct compaction, not Codex SSE")

    def wrap_upstream_response_as_sse(self, *args, **kwargs) -> str:
        raise NotImplementedError("hermes_harness mode is intended for main.py direct compaction, not Codex SSE")


def compacted_content_from_raw(
    raw: str,
    validate: bool = True,
    min_summary_chars: int = 300,
) -> str:
    summary = _ensure_summary_prefix(extract_summary_section(raw))
    if validate:
        validate_summary_quality(summary, min_summary_chars=min_summary_chars)
    return summary


def extract_summary_section(raw: str) -> str:
    matches = re.findall(r"<summary>(.*?)</summary>", raw, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()
    without_analysis = re.sub(
        r"<analysis>.*?</analysis>",
        "",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    return without_analysis or raw


def validate_summary_quality(summary: str, min_summary_chars: int = 300) -> None:
    body = _strip_summary_prefix(summary).strip()
    char_count = len(body)
    if char_count < min_summary_chars:
        raise HermesHarnessModeError(
            f"compact summary quality check failed: summary too short ({char_count} chars, minimum {min_summary_chars})"
        )
    if CONTEXT_SUMMARY_PREFIX not in summary:
        raise HermesHarnessModeError(
            f"compact summary quality check failed: summary must start with {CONTEXT_SUMMARY_PREFIX!r}"
        )


def extract_summary_text(parsed: Any) -> str | None:
    if isinstance(parsed, str):
        return parsed
    if not isinstance(parsed, dict):
        return None

    choices = parsed.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

    root = parsed.get("response") if isinstance(parsed.get("response"), dict) else parsed
    candidates = root.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if isinstance(parts, list):
            text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and part.get("thought") is not True
            )
            if text:
                return text

    content = root.get("content")
    if isinstance(content, list):
        text = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if text:
            return text

    return None


def _messages_from_body(body: Any) -> tuple[list[dict[str, Any]], str | None]:
    if isinstance(body, list):
        return body, None
    if not isinstance(body, dict):
        raise HermesHarnessModeError("compact input must be a messages list or an object with messages/input")

    model = body.get("model") if isinstance(body.get("model"), str) else None
    messages = body.get("messages")
    if isinstance(messages, list):
        return messages, model

    input_items = _normalize_input(body.get("input"))
    if input_items:
        return [_input_item_to_message(item) for item in input_items if not _is_skipped_input_item(item)], model

    raise HermesHarnessModeError("compact input has no messages or input history")


def _normalize_input(raw_input: Any) -> list[Any]:
    if raw_input is None:
        return []
    if isinstance(raw_input, list):
        return raw_input
    if isinstance(raw_input, str):
        return [{"type": "message", "role": "user", "content": raw_input}] if raw_input.strip() else []
    if isinstance(raw_input, dict):
        return [raw_input]
    return [{"type": "message", "role": "user", "content": str(raw_input)}]


def _is_skipped_input_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") in {"compaction_trigger", "reasoning"}


def _input_item_to_message(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"role": "user", "content": json.dumps(item, ensure_ascii=False)}
    item_type = item.get("type")
    if item_type == "function_call":
        name = item.get("name") or item.get("function", {}).get("name") or "function"
        call_id = item.get("call_id") or item.get("id") or ""
        args = item.get("arguments") or item.get("function", {}).get("arguments") or ""
        return {"role": "assistant", "content": f"function_call {name} {call_id}\n{args}"}
    if item_type == "function_call_output":
        call_id = item.get("call_id") or ""
        output = item.get("output") or item.get("content") or ""
        return {"role": "tool", "content": f"tool output {call_id}\n{output}"}
    role = str(item.get("role") or "user")
    return {"role": role, "content": item.get("content", "")}


def _render_hermes_turns(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for idx, message in enumerate(messages):
        role = _hermes_role(str(message.get("role") or "message"))
        value = _extract_text(message.get("content"))
        parts.append(f"[Turn {idx} - {role.upper()}]:\n{value}")
    return "\n\n".join(parts).strip()


def _hermes_role(role: str) -> str:
    role_map = {
        "assistant": "gpt",
        "user": "human",
        "tool": "tool",
        "system": "system",
        "developer": "system",
    }
    return role_map.get(role, role)


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _truncate_middle(content, MAX_SINGLE_FIELD_CHARS)
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif isinstance(part.get("input_text"), str):
                    parts.append(part["input_text"])
                elif isinstance(part.get("output_text"), str):
                    parts.append(part["output_text"])
                elif part.get("type") in {"input_image", "image_url"}:
                    parts.append("[image omitted from compact text render]")
                elif part.get("type") in {"input_file", "file"}:
                    parts.append("[file omitted from compact text render]")
                else:
                    parts.append(json.dumps(part, ensure_ascii=False))
            else:
                parts.append(str(part))
        return _truncate_middle("\n".join(parts), MAX_SINGLE_FIELD_CHARS)
    return _truncate_middle(json.dumps(content, ensure_ascii=False), MAX_SINGLE_FIELD_CHARS)


def _ensure_summary_prefix(summary: str) -> str:
    text = (summary or "").strip()
    if text.startswith(CONTEXT_SUMMARY_PREFIX):
        return text
    body = _strip_summary_prefix(text).strip()
    return CONTEXT_SUMMARY_PREFIX if not body else f"{CONTEXT_SUMMARY_PREFIX} {body}"


def _strip_summary_prefix(summary: str) -> str:
    return re.sub(r"^\s*\[CONTEXT SUMMARY\]:\s*", "", summary, flags=re.IGNORECASE)


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return (
        text[:head]
        + f"\n[... omitted {len(text) - max_chars} chars for compact request budget ...]\n"
        + text[-tail:]
    )


def _looks_like_qwen_thinking_model(model: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", model.lower())
    return "qwen3" in normalized or normalized.startswith("qwen")
