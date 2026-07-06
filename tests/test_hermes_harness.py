from __future__ import annotations

import unittest

import main
from compact_mode import get_mode
from compact_mode.hermes_harness.mode import (
    HermesHarnessMode,
    HermesHarnessModeError,
    extract_summary_section,
)
from compact_mode.hermes_harness.prompt import CONTEXT_SUMMARY_PREFIX


def long_summary() -> str:
    detail = "Preserved implementation context with concrete file names, tool results, and decisions. " * 8
    return f"""<analysis>
Chronological notes that should not survive.
</analysis>

<summary>
{CONTEXT_SUMMARY_PREFIX} The assistant inspected README.md, main.py, and compact mode implementations. It found that hermes_harness was registered as a placeholder and planned to replace it with a direct mode. {detail}
</summary>"""


class HermesHarnessModeTests(unittest.TestCase):
    def test_get_mode_returns_real_implementation(self):
        mode = get_mode("hermes_harness")
        self.assertIsInstance(mode, HermesHarnessMode)

    def test_build_chat_request_from_messages_uses_hermes_turn_format(self):
        mode = HermesHarnessMode()
        chat = mode.build_chat_request_from_messages(
            [
                {"role": "user", "content": "实现 Hermes compact mode"},
                {"role": "assistant", "content": "I inspected the repo."},
                {"role": "tool", "content": "tests passed"},
            ],
            model="Qwen3-30B-A3B",
        )
        self.assertEqual(chat["model"], "Qwen3-30B-A3B")
        self.assertFalse(chat["stream"])
        self.assertEqual(chat["max_tokens"], 1500)
        self.assertFalse(chat["enable_thinking"])
        rendered = chat["messages"][0]["content"]
        self.assertIn("[Turn 0 - HUMAN]", rendered)
        self.assertIn("[Turn 1 - GPT]", rendered)
        self.assertIn("[Turn 2 - TOOL]", rendered)
        self.assertIn("实现 Hermes compact mode", rendered)
        self.assertIn(CONTEXT_SUMMARY_PREFIX, chat["messages"][1]["content"])

    def test_build_chat_request_accepts_messages_body(self):
        mode = HermesHarnessMode()
        chat = mode.build_chat_request(
            {
                "model": "compact-model",
                "messages": [{"role": "user", "content": "full history"}],
            }
        )
        self.assertEqual(chat["model"], "compact-model")
        self.assertIn("[Turn 0 - HUMAN]", chat["messages"][0]["content"])

    def test_extract_summary_section_drops_analysis(self):
        section = extract_summary_section(long_summary())
        self.assertIn(CONTEXT_SUMMARY_PREFIX, section)
        self.assertNotIn("Chronological notes", section)

    def test_wrap_compact_json_normalizes_prefix_once(self):
        mode = HermesHarnessMode()
        wrapped = mode.wrap_compact_json(long_summary(), min_summary_chars=100)
        content = wrapped["output"][0]["encrypted_content"]
        self.assertTrue(content.startswith(CONTEXT_SUMMARY_PREFIX))
        self.assertEqual(content.count(CONTEXT_SUMMARY_PREFIX), 1)
        self.assertIn("README.md", content)
        self.assertNotIn("Chronological notes", content)

    def test_wrap_compact_json_adds_missing_prefix(self):
        mode = HermesHarnessMode()
        raw = "The assistant inspected files and retained concrete continuation details. " * 8
        content = mode.wrap_compact_json(raw, min_summary_chars=100)["output"][0]["encrypted_content"]
        self.assertTrue(content.startswith(CONTEXT_SUMMARY_PREFIX))

    def test_rejects_short_summary(self):
        mode = HermesHarnessMode()
        with self.assertRaises(HermesHarnessModeError):
            mode.wrap_compact_json(f"{CONTEXT_SUMMARY_PREFIX} too short")

    def test_main_compaction_path_uses_hermes_harness_mode(self):
        calls = []

        def fake_post_json(base_url, api_key, path, body, timeout):
            calls.append((base_url, api_key, path, body, timeout))
            return {
                "choices": [{"message": {"role": "assistant", "content": long_summary()}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 22},
            }

        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current request"},
        ]
        compacted, metadata = main.compact_messages_if_needed(
            messages,
            compact_mode=HermesHarnessMode(),
            compact_model="Qwen3-30B-A3B",
            compact_base_url="http://compact.local/v1",
            compact_api_key="key",
            timeout=30,
            language="en",
            threshold_chars=1,
            force=False,
            post_json=fake_post_json,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(metadata["mode"], "hermes_harness")
        self.assertIn(CONTEXT_SUMMARY_PREFIX, compacted[1]["content"])
        self.assertEqual(compacted[-1]["content"], "current request")


if __name__ == "__main__":
    unittest.main()
