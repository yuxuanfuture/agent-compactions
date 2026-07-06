from __future__ import annotations

import json
import unittest

from compact_mode.codex_remote_v2.mode import (
    CodexRemoteV2Mode,
    CompactModeError,
    extract_usage,
)
from compact_mode.codex_remote_v2.prompt import COMPACT_SUMMARY_PREFIX


def request_body():
    return {
        "model": "Qwen3-30B-A3B",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": "实现 compact mode"},
            {"type": "reasoning", "summary": "internal thinking"},
            {"type": "compaction_trigger"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        ],
    }


class CodexRemoteV2Tests(unittest.TestCase):
    def test_detects_trigger_but_ignores_word_in_message(self):
        mode = CodexRemoteV2Mode()
        self.assertTrue(mode.detect(request_body(), "/v1/responses"))
        plain = {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": 'what is "compaction_trigger"?',
                }
            ],
        }
        self.assertFalse(mode.detect(plain, "/responses"))
        self.assertFalse(mode.detect(request_body(), "/chat/completions"))

    def test_strip_trigger_keeps_other_items(self):
        mode = CodexRemoteV2Mode()
        stripped = mode.strip_compaction_trigger(request_body())
        types = [item.get("type") for item in stripped["input"]]
        self.assertEqual(types, ["message", "reasoning", "function_call_output"])

    def test_build_chat_request_appends_prompt_and_disables_qwen_thinking(self):
        mode = CodexRemoteV2Mode()
        chat = mode.build_chat_request(request_body(), client_path="/responses")
        self.assertEqual(chat["model"], "Qwen3-30B-A3B")
        self.assertFalse(chat["enable_thinking"])
        self.assertFalse(chat["stream"])
        self.assertIn("CONTEXT CHECKPOINT COMPACTION", chat["messages"][-1]["content"])
        rendered = chat["messages"][0]["content"]
        self.assertIn("实现 compact mode", rendered)
        self.assertIn("tool output c1", rendered)
        self.assertNotIn("internal thinking", rendered)
        self.assertNotIn("compaction_trigger", rendered)

    def test_build_chat_request_rejects_trigger_only_without_session_cache(self):
        mode = CodexRemoteV2Mode()
        body = {
            "model": "Qwen3-30B-A3B",
            "previous_response_id": "resp_1",
            "input": [{"type": "compaction_trigger"}],
        }
        with self.assertRaises(CompactModeError) as ctx:
            mode.build_chat_request(body, client_path="/responses")
        self.assertIn("no inline history", str(ctx.exception))

    def test_wrap_summary_as_sse_success_shape(self):
        mode = CodexRemoteV2Mode()
        summary = "## Summary\n" + ("context detail line with substance. " * 60)
        sse = mode.wrap_summary_as_sse(summary)
        self.assertIn("event: response.created", sse)
        self.assertIn("event: response.output_item.done", sse)
        self.assertIn("event: response.completed", sse)
        self.assertEqual(sse.count('"type":"compaction"'), 2)
        self.assertIn('"sequence_number":0', sse)
        self.assertIn('"sequence_number":1', sse)
        self.assertIn('"sequence_number":2', sse)
        self.assertIn(COMPACT_SUMMARY_PREFIX, sse)

    def test_wrap_short_summary_emits_failed_sse(self):
        mode = CodexRemoteV2Mode()
        sse = mode.wrap_summary_as_sse("too short")
        self.assertIn("event: response.failed", sse)
        self.assertIn('"code":"invalid_prompt"', sse)
        self.assertNotIn("event: response.completed", sse)

    def test_extracts_usage_from_chat_and_gemini(self):
        chat = {"usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        self.assertEqual(extract_usage(chat)["total_tokens"], 10)
        gemini = {"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5}}
        self.assertEqual(extract_usage(gemini)["input_tokens"], 20)

    def test_wrap_upstream_response(self):
        mode = CodexRemoteV2Mode()
        summary = "## Summary\n" + ("context detail line with substance. " * 60)
        upstream = {
            "choices": [{"message": {"role": "assistant", "content": summary}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 340},
        }
        sse = mode.wrap_upstream_response_as_sse(upstream)
        self.assertIn('"input_tokens":1200', sse)
        self.assertIn('"output_tokens":340', sse)


if __name__ == "__main__":
    unittest.main()
