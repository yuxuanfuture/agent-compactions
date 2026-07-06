from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from compact_mode.codex_remote_v2.prompt import (
    COMPACT_SUMMARY_PREFIX,
    choose_prompt,
)
from compact_mode.codex_remote_v2.sse import failed_sse, success_sse, zero_usage


RESPONSES_PATHS = {"/responses", "/v1/responses", "/openai/v1/responses"}
MAX_OUTPUT_TOKENS = 20_000
MAX_RENDERED_HISTORY_CHARS = 120_000
MAX_SINGLE_FIELD_CHARS = 12_000


class CompactModeError(ValueError):
    pass


@dataclass
class CodexRemoteV2Mode:
    name: str = "codex_remote_v2"

    def detect(self, body: dict[str, Any], client_path: str = "/responses") -> bool:
        path = _strip_query(client_path)
        if path not in RESPONSES_PATHS:
            return False
        return _has_compaction_trigger(body)

    def strip_compaction_trigger(self, body: dict[str, Any]) -> dict[str, Any]:
        stripped = json.loads(json.dumps(body, ensure_ascii=False))
        items = _normalize_input(stripped.get("input"))
        kept = [item for item in items if not _is_type(item, "compaction_trigger")]
        if len(kept) == len(items):
            raise CompactModeError("compact v2 body has no input item with type=compaction_trigger")
        stripped["input"] = kept
        return stripped

    def build_chat_request(
        self,
        body: dict[str, Any],
        client_path: str = "/responses",
        model_override: str | None = None,
        language: str = "auto",
        disable_thinking: bool = True,
    ) -> dict[str, Any]:
        if not self.detect(body, client_path):
            raise CompactModeError(
                "request is not Codex remote compaction v2: expected /responses with an input compaction_trigger item"
            )
        stripped = self.strip_compaction_trigger(body)
        items = [
            item
            for item in _normalize_input(stripped.get("input"))
            if not _is_type(item, "reasoning")
        ]
        if not items:
            prev_id = str(stripped.get("previous_response_id") or "").strip()
            detail = (
                f" previous_response_id={prev_id!r} was present, but this standalone mode has no Codex session cache."
                if prev_id
                else ""
            )
            raise CompactModeError(
                "compact v2 request has no inline history after removing compaction_trigger/reasoning;"
                + detail
                + " provide a request body with inline input history before calling the compact model"
            )
        rendered_history = _truncate_middle(_render_input_items(items), MAX_RENDERED_HISTORY_CHARS)
        prompt = choose_prompt(language, rendered_history)
        model = model_override or stripped.get("model")
        if not model:
            raise CompactModeError("compact request is missing model; pass --model to override")

        messages = [
            {
                "role": "user",
                "content": "Conversation history to compact:\n\n" + rendered_history,
            },
            {"role": "user", "content": prompt},
        ]
        chat_request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }
        _copy_if_present(stripped, chat_request, "temperature")
        _copy_if_present(stripped, chat_request, "top_p")
        _copy_if_present(stripped, chat_request, "seed")
        if disable_thinking and _looks_like_qwen_thinking_model(str(model)):
            chat_request["enable_thinking"] = False
        return chat_request

    def build_chat_request_from_messages(
        self,
        messages: list[dict[str, Any]],
        model: str,
        language: str = "auto",
        disable_thinking: bool = True,
    ) -> dict[str, Any]:
        if not messages:
            raise CompactModeError("cannot compact an empty message history")
        rendered_history = _truncate_middle(_render_chat_messages(messages), MAX_RENDERED_HISTORY_CHARS)
        prompt = choose_prompt(language, rendered_history)
        chat_request: dict[str, Any] = {
            "model": model,
            "messages": [
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
            raise CompactModeError(f"upstream response does not contain summary text: {preview}")
        return text

    def wrap_compact_json(
        self,
        raw_summary: str,
        validate: bool = True,
        min_summary_chars: int = 800,
    ) -> dict[str, Any]:
        encrypted = encrypted_content_from_raw(
            raw_summary,
            validate=validate,
            min_summary_chars=min_summary_chars,
        )
        return {"output": [{"type": "compaction", "encrypted_content": encrypted}]}

    def wrap_summary_as_sse(
        self,
        raw_summary: str,
        validate: bool = True,
        min_summary_chars: int = 800,
    ) -> str:
        try:
            encrypted = encrypted_content_from_raw(
                raw_summary,
                validate=validate,
                min_summary_chars=min_summary_chars,
            )
        except CompactModeError as exc:
            return failed_sse("invalid_prompt", "quality_check_failed", str(exc))
        return success_sse(encrypted, zero_usage())

    def wrap_upstream_response_as_sse(
        self,
        upstream_response: Any,
        validate: bool = True,
        min_summary_chars: int = 800,
    ) -> str:
        try:
            raw_summary = self.extract_summary_text(upstream_response)
            encrypted = encrypted_content_from_raw(
                raw_summary,
                validate=validate,
                min_summary_chars=min_summary_chars,
            )
        except CompactModeError as exc:
            return failed_sse("invalid_prompt", "quality_check_failed", str(exc))
        usage = extract_usage(upstream_response)
        return success_sse(encrypted, usage)


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


def extract_usage(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return zero_usage()
    root = parsed.get("response") if isinstance(parsed.get("response"), dict) else parsed
    usage = root.get("usage") if isinstance(root.get("usage"), dict) else {}
    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    if input_tokens == 0 and output_tokens == 0:
        meta = root.get("usageMetadata") if isinstance(root.get("usageMetadata"), dict) else {}
        input_tokens = int(meta.get("promptTokenCount") or 0)
        output_tokens = int(meta.get("candidatesTokenCount") or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def encrypted_content_from_raw(
    raw: str,
    validate: bool = True,
    min_summary_chars: int = 800,
) -> str:
    summary = extract_summary_section(raw).strip()
    if validate:
        validate_summary_quality(summary, min_summary_chars=min_summary_chars)
    return f"{COMPACT_SUMMARY_PREFIX}\n{summary}"


def validate_summary_quality(summary: str, min_summary_chars: int = 800) -> None:
    char_count = len(summary)
    if char_count < min_summary_chars:
        raise CompactModeError(
            f"compact summary quality check failed: summary too short ({char_count} chars, minimum {min_summary_chars})"
        )
    has_markdown_header = any(line.lstrip().startswith("#") for line in summary.splitlines())
    if not has_markdown_header and char_count < max(1500, min_summary_chars):
        raise CompactModeError(
            f"compact summary quality check failed: summary lacks markdown headers and is short ({char_count} chars)"
        )


def extract_summary_section(raw: str) -> str:
    start = raw.rfind("<summary>")
    if start == -1:
        return raw
    after = raw[start + len("<summary>") :]
    end = after.find("</summary>")
    if end == -1:
        return after
    return after[:end]


def _strip_query(path: str) -> str:
    return path.split("?", 1)[0].rstrip("/") or "/"


def _has_compaction_trigger(body: dict[str, Any]) -> bool:
    return any(_is_type(item, "compaction_trigger") for item in _normalize_input(body.get("input")))


def _is_type(value: Any, type_name: str) -> bool:
    return isinstance(value, dict) and value.get("type") == type_name


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


def _render_input_items(items: list[Any]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(items, start=1):
        blocks.append(_render_input_item(idx, item))
    return "\n\n".join(blocks).strip() or "[No inline history was present after compaction_trigger was removed.]"


def _render_chat_messages(messages: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for idx, message in enumerate(messages, start=1):
        role = str(message.get("role") or "message")
        blocks.append(f"[{idx}] {role}\n{_extract_text(message.get('content'))}")
    return "\n\n".join(blocks).strip()


def _render_input_item(idx: int, item: Any) -> str:
    if not isinstance(item, dict):
        return f"[{idx}] raw\n{_truncate_middle(json.dumps(item, ensure_ascii=False), MAX_SINGLE_FIELD_CHARS)}"

    item_type = item.get("type", "message")
    role = item.get("role", item_type)
    if item_type == "message" or "content" in item:
        return f"[{idx}] {role}\n{_extract_text(item.get('content'))}"
    if item_type == "function_call":
        name = item.get("name") or item.get("function", {}).get("name") or "function"
        call_id = item.get("call_id") or item.get("id") or ""
        args = item.get("arguments") or item.get("function", {}).get("arguments") or ""
        return f"[{idx}] assistant function_call {name} {call_id}\n{_truncate_middle(str(args), MAX_SINGLE_FIELD_CHARS)}"
    if item_type == "function_call_output":
        call_id = item.get("call_id") or ""
        output = item.get("output") or item.get("content") or ""
        return f"[{idx}] tool output {call_id}\n{_truncate_middle(str(output), MAX_SINGLE_FIELD_CHARS)}"
    if item_type == "compaction":
        return f"[{idx}] previous compaction\n{_truncate_middle(str(item.get('encrypted_content', '')), MAX_SINGLE_FIELD_CHARS)}"
    return f"[{idx}] {item_type}\n{_truncate_middle(json.dumps(item, ensure_ascii=False), MAX_SINGLE_FIELD_CHARS)}"


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


def _copy_if_present(src: dict[str, Any], dst: dict[str, Any], key: str) -> None:
    if key in src:
        dst[key] = src[key]


def _looks_like_qwen_thinking_model(model: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", model.lower())
    return "qwen3" in normalized or normalized.startswith("qwen")
