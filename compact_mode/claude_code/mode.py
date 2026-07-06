from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from compact_mode.claude_code.prompt import (
    COMPACTED_CONTEXT_HEADER,
    CONVERSATION_SUMMARIZATION_PROMPT,
    MAX_OUTPUT_TOKENS,
    MAX_RENDERED_HISTORY_CHARS,
    MAX_SINGLE_FIELD_CHARS,
    NO_TOOLS_GUARD,
    PARTIAL_COMPACTION_PROMPT,
)


class ClaudeCodeModeError(ValueError):
    pass


@dataclass
class ClaudeCodeMode:
    name: str = "claude_code"

    def build_chat_request(
        self,
        body: Any,
        client_path: str = "/chat/completions",
        model_override: str | None = None,
        language: str = "auto",
        disable_thinking: bool = True,
    ) -> dict[str, Any]:
        del client_path, language
        messages, request_model = _messages_from_body(body)
        model = model_override or request_model
        if not model:
            raise ClaudeCodeModeError("compact request is missing model; pass --model to override")
        return self._build_chat_request(
            messages,
            model=model,
            prompt=CONVERSATION_SUMMARIZATION_PROMPT,
            disable_thinking=disable_thinking,
        )

    def build_chat_request_from_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
        language: str = "auto",
        disable_thinking: bool = True,
    ) -> dict[str, Any]:
        del language
        return self._build_chat_request(
            messages,
            model=model,
            prompt=PARTIAL_COMPACTION_PROMPT,
            disable_thinking=disable_thinking,
        )

    def _build_chat_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        prompt: str,
        disable_thinking: bool,
    ) -> dict[str, Any]:
        if not messages:
            raise ClaudeCodeModeError("cannot compact an empty message history")
        rendered_history = _truncate_middle(_render_chat_messages(messages), MAX_RENDERED_HISTORY_CHARS)
        chat_request: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": NO_TOOLS_GUARD},
                {
                    "role": "user",
                    "content": "Conversation history to compact:\n\n" + rendered_history,
                },
                {"role": "user", "content": prompt},
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
            raise ClaudeCodeModeError(f"upstream response does not contain summary text: {preview}")
        return text

    def wrap_compact_json(
        self,
        raw_summary: str,
        validate: bool = True,
        min_summary_chars: int = 400,
    ) -> dict[str, Any]:
        encrypted = compacted_content_from_raw(
            raw_summary,
            validate=validate,
            min_summary_chars=min_summary_chars,
        )
        return {"output": [{"type": "compaction", "encrypted_content": encrypted}]}

    def wrap_summary_as_sse(self, *args, **kwargs) -> str:
        raise NotImplementedError("claude_code mode is intended for main.py direct compaction, not Codex SSE")

    def wrap_upstream_response_as_sse(self, *args, **kwargs) -> str:
        raise NotImplementedError("claude_code mode is intended for main.py direct compaction, not Codex SSE")


def compacted_content_from_raw(
    raw: str,
    validate: bool = True,
    min_summary_chars: int = 400,
) -> str:
    summary = extract_summary_section(raw).strip()
    if validate:
        validate_summary_quality(summary, min_summary_chars=min_summary_chars)
    return f"{COMPACTED_CONTEXT_HEADER}\n{summary}"


def extract_summary_section(raw: str) -> str:
    formatted = _remove_tagged_blocks(raw, "analysis").strip()
    matches = _find_tagged_blocks(formatted, "summary")
    if matches:
        return matches[-1]
    return formatted or raw


def _find_tagged_blocks(text: str, tag: str) -> list[str]:
    pattern = _tag_pattern(tag, capture=True)
    return re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)


def _remove_tagged_blocks(text: str, tag: str) -> str:
    pattern = _tag_pattern(tag, capture=False)
    return re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)


def _tag_pattern(tag: str, capture: bool) -> str:
    # Some Markdown renderers space out tag letters in code examples. Accept both
    # normal tags such as <summary> and spaced tags such as <s u m m a r y>.
    spaced = r"\s*".join(re.escape(ch) for ch in tag)
    body = r"([\s\S]*?)" if capture else r"[\s\S]*?"
    return rf"<\s*{spaced}\s*>{body}<\s*/\s*{spaced}\s*>"


def validate_summary_quality(summary: str, min_summary_chars: int = 400) -> None:
    char_count = len(summary)
    if char_count < min_summary_chars:
        raise ClaudeCodeModeError(
            f"compact summary quality check failed: summary too short ({char_count} chars, minimum {min_summary_chars})"
        )
    has_numbered_sections = bool(re.search(r"(?m)^\s*1\.\s+", summary))
    has_current_or_context = "Current Work" in summary or "Context for Continuing Work" in summary
    if not has_numbered_sections and not has_current_or_context and char_count < max(900, min_summary_chars):
        raise ClaudeCodeModeError(
            "compact summary quality check failed: summary does not look like a Claude Code structured summary"
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
        raise ClaudeCodeModeError("compact input must be a messages list or an object with messages/input")

    model = body.get("model") if isinstance(body.get("model"), str) else None
    messages = body.get("messages")
    if isinstance(messages, list):
        return messages, model

    input_items = _normalize_input(body.get("input"))
    if input_items:
        return [_input_item_to_message(item) for item in input_items if not _is_skipped_input_item(item)], model

    raise ClaudeCodeModeError("compact input has no messages or input history")


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


def _render_chat_messages(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "message")
        blocks.append(f"[{idx}] {role}\n{_extract_text(message.get('content'))}")
    return "\n\n".join(blocks).strip()


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
