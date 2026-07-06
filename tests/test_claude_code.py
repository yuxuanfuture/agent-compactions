from __future__ import annotations

import unittest

import main
from compact_mode.claude_code.mode import (
    ClaudeCodeMode,
    ClaudeCodeModeError,
    extract_summary_section,
)
from compact_mode.claude_code.prompt import COMPACTED_CONTEXT_HEADER


def long_summary() -> str:
    detail = "Important implementation context with enough detail to continue safely. " * 12
    return f"""<analysis>
Chronological check that should not be retained.
</analysis>

<summary>
1. Primary Request and Intent:
   Build a Claude Code-style compact mode.

2. Key Technical Concepts:
   - OpenAI-compatible chat completion compaction.
   - Claude Code-style summary tags.

3. Files and Code Sections:
   - compact_mode/claude_code/mode.py: implements the mode.

4. Errors and fixes:
   - None.

5. Problem Solving:
   {detail}

6. All user messages:
   - User asked for --compact_mode claude_code.

7. Pending Tasks:
   - Run tests.

8. Work Completed:
   - Added the mode implementation.

9. Context for Continuing Work:
   - Continue from the latest preserved user message.
</summary>"""


class ClaudeCodeModeTests(unittest.TestCase):
    def test_build_chat_request_from_messages_uses_partial_prompt(self):
        mode = ClaudeCodeMode()
        chat = mode.build_chat_request_from_messages(
            [
                {"role": "user", "content": "old request"},
                {"role": "assistant", "content": "old answer"},
            ],
            model="Qwen3-30B-A3B",
        )
        self.assertEqual(chat["model"], "Qwen3-30B-A3B")
        self.assertFalse(chat["stream"])
        self.assertFalse(chat["enable_thinking"])
        self.assertEqual(chat["messages"][0]["role"], "system")
        self.assertIn("Do not call tools", chat["messages"][0]["content"])
        self.assertIn("old request", chat["messages"][1]["content"])
        self.assertIn("newer messages", chat["messages"][2]["content"])
        self.assertIn("<summary>", chat["messages"][2]["content"])

    def test_build_chat_request_accepts_messages_body(self):
        mode = ClaudeCodeMode()
        chat = mode.build_chat_request(
            {
                "model": "compact-model",
                "messages": [{"role": "user", "content": "full history"}],
            }
        )
        self.assertEqual(chat["model"], "compact-model")
        self.assertIn("conversation so far", chat["messages"][2]["content"])

    def test_extract_summary_section_drops_analysis(self):
        section = extract_summary_section(long_summary())
        self.assertIn("Primary Request and Intent", section)
        self.assertNotIn("Chronological check", section)

    def test_wrap_compact_json_uses_summary_only(self):
        mode = ClaudeCodeMode()
        wrapped = mode.wrap_compact_json(long_summary(), min_summary_chars=100)
        content = wrapped["output"][0]["encrypted_content"]
        self.assertTrue(content.startswith(COMPACTED_CONTEXT_HEADER + "\n"))
        self.assertNotIn("<summary>", content)
        self.assertNotIn("</summary>", content)
        self.assertIn("Primary Request and Intent", content)
        self.assertNotIn("Chronological check", content)

    def test_extract_summary_section_accepts_spaced_tags(self):
        raw = "<a n a l y s i s>discard this</a n a l y s i s><s u m m a r y>keep this structured summary</s u m m a r y>"
        self.assertEqual(extract_summary_section(raw), "keep this structured summary")

    def test_rejects_short_summary(self):
        mode = ClaudeCodeMode()
        with self.assertRaises(ClaudeCodeModeError):
            mode.wrap_compact_json("<summary>too short</summary>")

    def test_main_compaction_path_uses_claude_code_mode(self):
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
            compact_mode=ClaudeCodeMode(),
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
        self.assertEqual(metadata["mode"], "claude_code")
        self.assertTrue(compacted[1]["content"].startswith(COMPACTED_CONTEXT_HEADER + "\n"))
        self.assertEqual(compacted[-1]["content"], "current request")


if __name__ == "__main__":
    unittest.main()
