from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from benchmarks.base import (
    BenchmarkTask,
    compact_json,
    default_prediction_path,
    read_benchmark_records,
    select_record,
)


SYSTEM = (
    "You are an expert software engineer solving a SWE-bench task. "
    "Reason from the issue and repository context, then return only a git-compatible unified diff patch."
)

ID_FIELDS = ("instance_id", "id")


class SweBenchAdapter:
    name = "swe_bench"

    def load_task(
        self,
        input_path: str | None,
        benchmark_id: str | None,
        extra_prompt: str | None = None,
    ) -> BenchmarkTask:
        records = read_benchmark_records(input_path)
        record = select_record(records, benchmark_id, ID_FIELDS)
        task_id = _require_id(record)
        prompt = build_swe_prompt(record, extra_prompt=extra_prompt)
        return BenchmarkTask(
            benchmark=self.name,
            task_id=task_id,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            source=record,
            metadata={
                "instance_id": task_id,
                "repo": record.get("repo"),
                "base_commit": record.get("base_commit"),
                "version": record.get("version"),
            },
        )

    def build_prediction(self, task: BenchmarkTask, assistant_text: str, model_name: str | None) -> dict[str, Any]:
        return {
            "instance_id": task.task_id,
            "model_name_or_path": model_name or "unknown",
            "model_patch": extract_patch(assistant_text),
        }

    def default_output_path(self, task: BenchmarkTask) -> Path:
        return default_prediction_path(self.name, task.task_id)


def build_swe_prompt(record: dict[str, Any], extra_prompt: str | None = None) -> str:
    parts = [
        "# SWE-bench Task",
        "",
        f"Instance ID: {_require_id(record)}",
        f"Repository: {record.get('repo', 'unknown')}",
        f"Base commit: {record.get('base_commit', 'unknown')}",
    ]
    if record.get("version") is not None:
        parts.append(f"Version: {record.get('version')}")
    parts.extend(
        [
            "",
            "## Issue",
            _field(record, "problem_statement", default=compact_json(record)),
        ]
    )
    if record.get("hints_text"):
        parts.extend(["", "## Hints", str(record["hints_text"])])
    tests = _test_context(record)
    if tests:
        parts.extend(["", "## Test Context", tests])
    if extra_prompt:
        parts.extend(["", "## Additional Instruction", extra_prompt])
    parts.extend(
        [
            "",
            "## Output Contract",
            "Return only a unified diff patch that can be applied with git apply.",
            "Do not include Markdown fences, prose, or test logs.",
        ]
    )
    return "\n".join(parts)


def extract_patch(text: str) -> str:
    stripped = text.strip()
    for block in re.findall(r"```(?:diff|patch)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL):
        candidate = block.strip()
        if _looks_like_patch(candidate):
            return candidate
    lines = stripped.splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("diff --git ") or line.startswith("--- "):
            return "\n".join(lines[idx:]).strip()
    return stripped


def _looks_like_patch(value: str) -> bool:
    return bool("diff --git " in value or re.search(r"^--- .*\n\+\+\+ ", value, flags=re.MULTILINE))


def _require_id(record: dict[str, Any]) -> str:
    for field in ID_FIELDS:
        value = record.get(field)
        if value is not None:
            return str(value)
    raise SystemExit("SWE-bench record is missing instance_id")


def _field(record: dict[str, Any], key: str, default: str = "") -> str:
    value = record.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return compact_json(value)


def _test_context(record: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        if record.get(key) is not None:
            chunks.append(f"{key}:\n{compact_json(record[key])}")
    return "\n\n".join(chunks)
