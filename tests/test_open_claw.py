from __future__ import annotations

import unittest

import main
from compact_mode import get_mode
from compact_mode.open_claw.mode import (
    OpenClawMode,
    OpenClawModeError,
    extract_summary_section,
)
from compact_mode.open_claw.prompt import (
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
)


def checkpoint_summary() -> str:
    detail = "Preserve concrete implementation details, paths, decisions, outputs, and pending state. " * 6
    return f"""<analysis>
Chronological scratchpad that should be removed.
</analysis>

<summary>
## Goal
Build an OpenClaw-style compact mode that works through --compact_mode open_claw.

## Constraints & Preferences
- Preserve exact compact-mode spelling and OpenClaw context continuity.
- Keep the latest user turn outside the compacted summary.

## Progress
### Done
- [x] Read README.md and existing compact mode implementations.
- [x] Added compact_mode/open_claw/mode.py and prompt.py.

### In Progress
- [ ] Run tests.

### Blocked
- None.

## Key Decisions
- **OpenClaw wrapper**: Use the OpenClaw compaction summary prefix and <summary> tags.

## Next Steps
1. Run the local unittest suite.

## Critical Context
- File path: compact_mode/open_claw/mode.py.
- Identifier policy: preserve IDs and file names exactly.
- {detail}
</summary>"""


def safeguard_summary() -> str:
    detail = "Exact details include /tmp/project/app.py, request id 123456789, and URL https://example.test/a. " * 5
    return f"""## Decisions
- Use a direct OpenClaw compact mode.

## Open TODOs
- Run tests.

## Constraints/Rules
- Preserve exact identifiers and unresolved user asks.

## Pending user asks
- User asked for --compact_mode open_claw.

## Exact identifiers
- --compact_mode open_claw
- compact_mode/open_claw/mode.py
- {detail}"""


class OpenClawModeTests(unittest.TestCase):
    def test_get_mode_returns_real_implementation(self):
        mode = get_mode("open_claw")
        self.assertIsInstance(mode, OpenClawMode)

    def test_build_chat_request_from_messages_uses_openclaw_prompt(self):
        mode = OpenClawMode()
        chat = mode.build_chat_request_from_messages(
            [
                {"role": "user", "content": "实现 OpenClaw compact mode"},
                {"role": "assistant", "content": "I inspected the repo."},
                {"role": "tool", "content": "tests passed"},
            ],
            model="Qwen3-30B-A3B",
        )
        self.assertEqual(chat["model"], "Qwen3-30B-A3B")
        self.assertFalse(chat["stream"])
        self.assertEqual(chat["max_tokens"], 13000)
        self.assertFalse(chat["enable_thinking"])
        self.assertEqual(chat["messages"][0]["role"], "system")
        prompt = chat["messages"][1]["content"]
        self.assertIn("<conversation>", prompt)
        self.assertIn("[User]: 实现 OpenClaw compact mode", prompt)
        self.assertIn("[Assistant]: I inspected the repo.", prompt)
        self.assertIn("[Tool result]: tests passed", prompt)
        self.assertIn("## Critical Context", prompt)
        self.assertIn("## Exact identifiers", prompt)

    def test_build_chat_request_accepts_messages_body(self):
        mode = OpenClawMode()
        chat = mode.build_chat_request(
            {
                "model": "compact-model",
                "messages": [{"role": "user", "content": "full history"}],
            }
        )
        self.assertEqual(chat["model"], "compact-model")
        self.assertIn("[User]: full history", chat["messages"][1]["content"])

    def test_extract_summary_section_drops_analysis(self):
        section = extract_summary_section(checkpoint_summary())
        self.assertIn("## Goal", section)
        self.assertNotIn("Chronological scratchpad", section)

    def test_wrap_compact_json_uses_openclaw_summary_wrapper(self):
        mode = OpenClawMode()
        wrapped = mode.wrap_compact_json(checkpoint_summary(), min_summary_chars=100)
        content = wrapped["output"][0]["encrypted_content"]
        self.assertTrue(content.startswith(COMPACTION_SUMMARY_PREFIX))
        self.assertTrue(content.endswith(COMPACTION_SUMMARY_SUFFIX))
        self.assertEqual(content.count(COMPACTION_SUMMARY_PREFIX), 1)
        self.assertIn("## Critical Context", content)
        self.assertNotIn("Chronological scratchpad", content)

    def test_wrap_compact_json_accepts_safeguard_sections(self):
        mode = OpenClawMode()
        content = mode.wrap_compact_json(safeguard_summary(), min_summary_chars=100)["output"][0][
            "encrypted_content"
        ]
        self.assertIn("## Exact identifiers", content)
        self.assertTrue(content.startswith(COMPACTION_SUMMARY_PREFIX))

    def test_wrap_compact_json_does_not_double_wrap_existing_wrapper(self):
        mode = OpenClawMode()
        existing = f"{COMPACTION_SUMMARY_PREFIX}{safeguard_summary()}{COMPACTION_SUMMARY_SUFFIX}"
        content = mode.wrap_compact_json(existing, min_summary_chars=100)["output"][0]["encrypted_content"]
        self.assertEqual(content.count(COMPACTION_SUMMARY_PREFIX), 1)
        self.assertEqual(content.count(COMPACTION_SUMMARY_SUFFIX), 1)

    def test_rejects_short_summary(self):
        mode = OpenClawMode()
        with self.assertRaises(OpenClawModeError):
            mode.wrap_compact_json("## Goal\nToo short")

    def test_main_compaction_path_uses_openclaw_mode(self):
        calls = []

        def fake_post_json(base_url, api_key, path, body, timeout):
            calls.append((base_url, api_key, path, body, timeout))
            return {
                "choices": [{"message": {"role": "assistant", "content": checkpoint_summary()}}],
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
            compact_mode=OpenClawMode(),
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
        self.assertEqual(metadata["mode"], "open_claw")
        self.assertTrue(compacted[1]["content"].startswith(COMPACTION_SUMMARY_PREFIX))
        self.assertEqual(compacted[-1]["content"], "current request")


if __name__ == "__main__":
    unittest.main()
