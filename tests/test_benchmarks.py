from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import main
from benchmarks import get_benchmark
from benchmarks.base import read_benchmark_records, write_prediction
from benchmarks.swe_bench import SweBenchAdapter, extract_patch
from benchmarks.tau_bench import TauBenchAdapter
from compact_mode import get_mode


class BenchmarkAdapterTests(unittest.TestCase):
    def test_reads_json_object_with_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "instances.json"
            path.write_text(
                json.dumps({"instances": [{"instance_id": "one"}, {"instance_id": "two"}]}),
                encoding="utf-8",
            )
            self.assertEqual([row["instance_id"] for row in read_benchmark_records(str(path))], ["one", "two"])

    def test_swe_bench_loads_task_and_extracts_patch_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swe.jsonl"
            record = {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "base_commit": "abc123",
                "problem_statement": "Fix the bug.",
                "FAIL_TO_PASS": ["tests/test_bug.py::test_fix"],
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            task = SweBenchAdapter().load_task(str(path), "django__django-1", extra_prompt="Keep it minimal.")
        self.assertEqual(task.task_id, "django__django-1")
        self.assertIn("Fix the bug.", task.messages[0]["content"])
        self.assertIn("Keep it minimal.", task.messages[0]["content"])
        patch = extract_patch(
            "Here is the fix:\n```diff\n"
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "```"
        )
        self.assertTrue(patch.startswith("diff --git a/app.py b/app.py"))
        prediction = SweBenchAdapter().build_prediction(task, patch, "model-x")
        self.assertEqual(prediction["instance_id"], "django__django-1")
        self.assertEqual(prediction["model_name_or_path"], "model-x")
        self.assertIn("+new", prediction["model_patch"])

    def test_tau_bench_builds_prediction_with_json_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tau.json"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "task_001",
                        "domain": "retail",
                        "instruction": "Cancel the order.",
                        "tools": [{"name": "cancel_order"}],
                        "initial_state": {"orders": [{"id": "o1"}]},
                    }
                ),
                encoding="utf-8",
            )
            task = TauBenchAdapter().load_task(input_path=str(path), benchmark_id=None)
        prediction = TauBenchAdapter().build_prediction(task, '```json\n{"status":"done"}\n```', "m")
        self.assertEqual(prediction["task_id"], "task_001")
        self.assertEqual(prediction["response_json"], {"status": "done"})

    def test_write_prediction_jsonl_appends_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            write_prediction(path, {"a": 1})
            write_prediction(path, {"b": 2})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, [{"a": 1}, {"b": 2}])

    def test_get_benchmark_accepts_dash_alias(self):
        self.assertIs(get_benchmark("swe-bench"), get_benchmark("swe_bench"))
        self.assertIs(get_benchmark("tau-bench"), get_benchmark("tau_bench"))


class BenchmarkMainIntegrationTests(unittest.TestCase):
    def test_run_benchmark_writes_swe_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "swe.jsonl"
            output_path = Path(tmp) / "predictions.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "instance_id": "repo__project-1",
                        "repo": "repo/project",
                        "base_commit": "abc123",
                        "problem_statement": "Fix a failing import.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = main.build_parser().parse_args(
                [
                    "--benchmark",
                    "swe_bench",
                    "--benchmark-input",
                    str(input_path),
                    "--instance-id",
                    "repo__project-1",
                    "--benchmark-output",
                    str(output_path),
                    "--model",
                    "task-model",
                    "--base-url",
                    "http://task.local/v1",
                    "--no-trajectory",
                ]
            )

            def fake_post_json(base_url, api_key, path, body, timeout):
                self.assertEqual(base_url, "http://task.local/v1")
                self.assertIn("Fix a failing import.", body["messages"][-1]["content"])
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a\n+b",
                            }
                        }
                    ]
                }

            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = main.run_benchmark(args, get_mode(args.compact_mode), post_json=fake_post_json)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(exit_code, 0)
        self.assertEqual(rows[0]["instance_id"], "repo__project-1")
        self.assertIn("+b", rows[0]["model_patch"])


if __name__ == "__main__":
    unittest.main()
