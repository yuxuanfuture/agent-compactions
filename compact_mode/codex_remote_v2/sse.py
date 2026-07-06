from __future__ import annotations

import json
import time
from typing import Any


def response_id() -> str:
    return f"resp_compact_v2_{int(time.time() * 1000)}"


def emit_sse_event(event: str, payload: dict[str, Any], sequence_number: int) -> str:
    framed = dict(payload)
    framed["sequence_number"] = sequence_number
    return f"event: {event}\ndata: {json.dumps(framed, ensure_ascii=False, separators=(',', ':'))}\n\n"


def success_sse(encrypted_content: str, usage: dict[str, Any] | None = None, rid: str | None = None) -> str:
    rid = rid or response_id()
    usage = usage or zero_usage()
    item = {"type": "compaction", "encrypted_content": encrypted_content}
    chunks = [
        emit_sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": rid,
                    "object": "response",
                    "status": "in_progress",
                    "output": [],
                },
            },
            0,
        ),
        emit_sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": item,
            },
            1,
        ),
        emit_sse_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": rid,
                    "object": "response",
                    "status": "completed",
                    "output": [item],
                    "usage": usage,
                },
            },
            2,
        ),
    ]
    return "".join(chunks)


def failed_sse(code: str, kind: str, message: str, rid: str | None = None) -> str:
    rid = rid or response_id()
    return "".join(
        [
            emit_sse_event(
                "response.created",
                {
                    "type": "response.created",
                    "response": {
                        "id": rid,
                        "object": "response",
                        "status": "in_progress",
                        "output": [],
                    },
                },
                0,
            ),
            emit_sse_event(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        "id": rid,
                        "object": "response",
                        "status": "failed",
                        "output": [],
                        "error": {
                            "code": code,
                            "type": kind,
                            "message": message,
                            "upstream_error_kind": kind,
                        },
                    },
                },
                1,
            ),
        ]
    )


def zero_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 0,
    }
