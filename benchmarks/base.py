from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


Message = dict[str, Any]


@dataclass
class BenchmarkTask:
    benchmark: str
    task_id: str
    system: str | None
    messages: list[Message]
    source: dict[str, Any]
    metadata: dict[str, Any]


class BenchmarkAdapter(Protocol):
    name: str

    def load_task(
        self,
        input_path: str | None,
        benchmark_id: str | None,
        extra_prompt: str | None = None,
    ) -> BenchmarkTask:
        ...

    def build_prediction(self, task: BenchmarkTask, assistant_text: str, model_name: str | None) -> dict[str, Any]:
        ...

    def default_output_path(self, task: BenchmarkTask) -> Path:
        ...


def read_benchmark_records(input_path: str | None) -> list[dict[str, Any]]:
    if not input_path:
        raise SystemExit("benchmark mode requires --benchmark-input")
    path = Path(input_path)
    if not path.exists():
        raise SystemExit(f"benchmark input not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            rows = loaded
        elif isinstance(loaded, dict):
            rows = _records_from_object(loaded)
        else:
            raise SystemExit("--benchmark-input must contain a JSON object, JSON list, or JSONL records")
    if not all(isinstance(row, dict) for row in rows):
        raise SystemExit("--benchmark-input records must be JSON objects")
    return rows


def select_record(records: list[dict[str, Any]], benchmark_id: str | None, id_fields: tuple[str, ...]) -> dict[str, Any]:
    if not records:
        raise SystemExit("--benchmark-input did not contain any records")
    if benchmark_id is None:
        if len(records) == 1:
            return records[0]
        available = ", ".join(_record_id(row, id_fields) for row in records[:10])
        raise SystemExit(f"--benchmark-id is required when input has multiple records; first ids: {available}")
    for row in records:
        if _record_id(row, id_fields) == benchmark_id:
            return row
    available = ", ".join(_record_id(row, id_fields) for row in records[:10])
    raise SystemExit(f"benchmark id {benchmark_id!r} not found; first ids: {available}")


def write_prediction(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return out


def default_prediction_path(benchmark: str, task_id: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_task_id = safe_filename(task_id)
    return Path("predictions") / f"{benchmark}-{safe_task_id}-{stamp}-{os.getpid()}.jsonl"


def run_evaluator_command(
    command: str,
    *,
    prediction_path: Path,
    input_path: str | None,
    task: BenchmarkTask,
    model_name: str | None,
    timeout: float,
) -> dict[str, Any]:
    context = _FormatContext(
        {
            "benchmark": task.benchmark,
            "benchmark_id": task.task_id,
            "task_id": task.task_id,
            "instance_id": task.task_id,
            "prediction_path": str(prediction_path),
            "benchmark_input": input_path or "",
            "input_path": input_path or "",
            "model": model_name or "",
        }
    )
    rendered = command.format_map(context)
    completed = subprocess.run(
        rendered,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "command": rendered,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def maybe_parse_json_object(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if not fence:
        return None
    try:
        return json.loads(fence.group(1).strip())
    except json.JSONDecodeError:
        return None


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned[:120] or "task"


def _records_from_object(value: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("instances", "tasks", "records", "data", "examples"):
        rows = value.get(key)
        if isinstance(rows, list):
            return rows
    return [value]


def _record_id(row: dict[str, Any], id_fields: tuple[str, ...]) -> str:
    for field in id_fields:
        value = row.get(field)
        if value is not None:
            return str(value)
    return ""


class _FormatContext(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"
