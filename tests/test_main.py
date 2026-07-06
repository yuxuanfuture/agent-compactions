from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import main
from compact_mode.codex_remote_v2.mode import CodexRemoteV2Mode
from compact_mode.codex_remote_v2.prompt import COMPACT_SUMMARY_PREFIX


class MainRunnerTests(unittest.TestCase):
    def test_normalizes_spaced_compact_mode_typo(self):
        self.assertEqual(
            main._normalize_argv(["--", "compact_mode", "codex_remote_v2", "--model", "m"]),
            ["--compact_mode", "codex_remote_v2", "--model", "m"],
        )
        self.assertEqual(
            main._normalize_argv(["--", "compatct_mode", "claude_code", "--model", "m"]),
            ["--compact_mode", "claude_code", "--model", "m"],
        )
        self.assertEqual(
            main._normalize_argv(["--compatct_mode=claude_code", "--model", "m"]),
            ["--compact_mode=claude_code", "--model", "m"],
        )
        self.assertEqual(
            main._normalize_argv(["--", "compact_mode", "hermes_harness", "--model", "m"]),
            ["--compact_mode", "hermes_harness", "--model", "m"],
        )
        self.assertEqual(
            main._normalize_argv(["--", "compatct_mode", "hermes_harness", "--model", "m"]),
            ["--compact_mode", "hermes_harness", "--model", "m"],
        )
        self.assertEqual(
            main._normalize_argv(["--", "compact_mode", "open_claw", "--model", "m"]),
            ["--compact_mode", "open_claw", "--model", "m"],
        )
        self.assertEqual(
            main._normalize_argv(["--", "compatct_mode", "open_claw", "--model", "m"]),
            ["--compact_mode", "open_claw", "--model", "m"],
        )

    def test_compacts_history_and_keeps_latest_user_turn(self):
        calls = []

        def fake_post_json(base_url, api_key, path, body, timeout):
            calls.append((base_url, api_key, path, body, timeout))
            summary = "## Summary\n" + ("previous task context. " * 80)
            return {
                "choices": [{"message": {"role": "assistant", "content": summary}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }

        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current request"},
        ]
        compacted, metadata = main.compact_messages_if_needed(
            messages,
            compact_mode=CodexRemoteV2Mode(),
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
        self.assertEqual(compacted[0]["role"], "system")
        self.assertIn(COMPACT_SUMMARY_PREFIX, compacted[1]["content"])
        self.assertEqual(compacted[-1]["content"], "current request")
        self.assertEqual(metadata["mode"], "codex_remote_v2")
        self.assertIn("raw_summary", metadata)
        self.assertIn("encrypted_content", metadata)
        self.assertIn(COMPACT_SUMMARY_PREFIX, metadata["encrypted_content"])
        self.assertEqual(metadata["compacted_messages_output"], compacted)

    def test_does_not_compact_below_threshold(self):
        messages = [{"role": "user", "content": "short"}]
        compacted, metadata = main.compact_messages_if_needed(
            messages,
            compact_mode=CodexRemoteV2Mode(),
            compact_model="Qwen3-30B-A3B",
            compact_base_url=None,
            compact_api_key=None,
            timeout=30,
            language="en",
            threshold_chars=9999,
            force=False,
        )
        self.assertIs(compacted, messages)
        self.assertIsNone(metadata)

    def test_trajectory_recorder_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/trajectory.jsonl"
            recorder = main.TrajectoryRecorder(path)
            recorder.record("context_compacted", {"encrypted_content": "summary"})
            rows = [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(rows[0]["event"], "context_compacted")
        self.assertEqual(rows[0]["encrypted_content"], "summary")


if __name__ == "__main__":
    unittest.main()
