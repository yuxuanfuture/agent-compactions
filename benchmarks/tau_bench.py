from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.base import (
    BenchmarkTask,
    compact_json,
    default_prediction_path,
    maybe_parse_json_object,
    read_benchmark_records,
    select_record,
)


SYSTEM = (
    "You are solving a tau-bench task. Follow the task instruction, respect tool schemas and environment state, "
    "and return the final answer or action trace requested by the task."
)

ID_FIELDS = ("task_id", "id", "taskId", "instance_id")


class TauBenchAdapter:
    name = "tau_bench"

    def load_task(
        self,
        input_path: str | None,
        benchmark_id: str | None,
        extra_prompt: str | None = None,
    ) -> BenchmarkTask:
        records = read_benchmark_records(input_path)
        record = select_record(records, benchmark_id, ID_FIELDS)
        task_id = _require_id(record)
        prompt = build_tau_prompt(record, extra_prompt=extra_prompt)
        return BenchmarkTask(
            benchmark=self.name,
            task_id=task_id,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            source=record,
            metadata={
                "task_id": task_id,
                "domain": record.get("domain"),
                "user_id": record.get("user_id") or record.get("userId"),
            },
        )

    def build_prediction(self, task: BenchmarkTask, assistant_text: str, model_name: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": task.task_id,
            "model": model_name or "unknown",
            "answer": assistant_text.strip(),
        }
        parsed = maybe_parse_json_object(assistant_text)
        if parsed is not None:
            payload["response_json"] = parsed
        return payload

    def default_output_path(self, task: BenchmarkTask) -> Path:
        return default_prediction_path(self.name, task.task_id)


def build_tau_prompt(record: dict[str, Any], extra_prompt: str | None = None) -> str:
    parts = [
        "# tau-bench Task",
        "",
        f"Task ID: {_require_id(record)}",
    ]
    if record.get("domain"):
        parts.append(f"Domain: {record['domain']}")
    instruction = _first_present(record, ("instruction", "user_instruction", "user_goal", "goal", "query"))
    if instruction:
        parts.extend(["", "## Instruction", instruction])
    if record.get("tools") is not None:
        parts.extend(["", "## Available Tools", compact_json(record["tools"])])
    state = _first_present(record, ("initial_state", "state", "environment", "database"))
    if state:
        parts.extend(["", "## Initial Environment State", state])
    if record.get("expected_result") is not None:
        parts.extend(["", "## Expected Result Schema", compact_json(record["expected_result"])])
    parts.extend(["", "## Full Task JSON", compact_json(record)])
    if extra_prompt:
        parts.extend(["", "## Additional Instruction", extra_prompt])
    parts.extend(
        [
            "",
            "## Output Contract",
            "Return the final answer. If the evaluator expects a structured trace, return a single JSON object.",
        ]
    )
    return "\n".join(parts)


def _require_id(record: dict[str, Any]) -> str:
    for field in ID_FIELDS:
        value = record.get(field)
        if value is not None:
            return str(value)
    raise SystemExit("tau-bench record is missing task_id or id")


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return value
        return compact_json(value)
    return None
