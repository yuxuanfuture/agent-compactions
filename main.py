from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks import benchmark_names, get_benchmark
from benchmarks.base import BenchmarkTask, run_evaluator_command, write_prediction
from compact_mode import get_mode, mode_names


Message = dict[str, Any]


@dataclass
class TaskRunResult:
    messages: list[Message]
    assistant_text: str
    compact_metadata: dict[str, Any] | None
    task_request: dict[str, Any]
    task_response: dict[str, Any]


class TrajectoryRecorder:
    def __init__(self, path: str | None) -> None:
        self.path = self._resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_path(path: str | None) -> Path:
        if not path or path == "auto":
            stamp = time.strftime("%Y%m%d-%H%M%S")
            return Path("trajectories") / f"trajectory-{stamp}-{os.getpid()}.jsonl"
        return Path(path)

    def record(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_secret(raw: str | None, fallback: str | None = None) -> str | None:
    value = raw if raw is not None else fallback
    if not value:
        return None
    if value.startswith("env:"):
        return os.environ.get(value[4:])
    return value


def _post_json(base_url: str, api_key: str | None, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_preview = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"upstream HTTP {exc.code}: {body_preview}") from exc


def _normalize_argv(argv: list[str]) -> list[str]:
    """Accept `--compact_mode x`, `-- compact_mode x`, and the compatct typo."""
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        if argv[idx] == "--" and idx + 1 < len(argv) and argv[idx + 1] in {"compact_mode", "compatct_mode"}:
            out.append("--compact_mode")
            idx += 2
            continue
        if argv[idx] == "--compatct_mode":
            out.append("--compact_mode")
            idx += 1
            continue
        if argv[idx].startswith("--compatct_mode="):
            out.append("--compact_mode=" + argv[idx].split("=", 1)[1])
            idx += 1
            continue
        out.append(argv[idx])
        idx += 1
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main task-model runner with pluggable context compaction.",
    )
    parser.add_argument("--compact_mode", default="codex_remote_v2", choices=mode_names())
    parser.add_argument("--model", default=os.environ.get("MAIN_MODEL") or os.environ.get("MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default="env:OPENAI_API_KEY")
    parser.add_argument("--compact-model", default=os.environ.get("COMPACT_MODEL", "Qwen3-30B-A3B"))
    parser.add_argument("--compact-base-url", default=os.environ.get("COMPACT_BASE_URL"))
    parser.add_argument("--compact-api-key", default=os.environ.get("COMPACT_API_KEY"))
    parser.add_argument("--prompt", default=None, help="User task. Defaults to stdin when stdin is piped.")
    parser.add_argument("--system", default=None, help="Optional system message for the task model.")
    parser.add_argument("--messages", default=None, help="Existing chat messages JSON list, or object with a messages field.")
    parser.add_argument("--save-messages", default=None, help="Write the final message transcript here.")
    parser.add_argument(
        "--benchmark",
        choices=benchmark_names(),
        default=None,
        help="Run a benchmark adapter instead of a raw prompt task.",
    )
    parser.add_argument("--benchmark-input", default=None, help="Benchmark task JSON/JSONL input file.")
    parser.add_argument(
        "--benchmark-id",
        "--instance-id",
        "--task-id",
        dest="benchmark_id",
        default=None,
        help="Benchmark record id. For SWE-bench this is instance_id; for tau-bench this is task_id/id.",
    )
    parser.add_argument("--benchmark-output", default=None, help="Prediction output path. Defaults to predictions/*.jsonl.")
    parser.add_argument("--benchmark-results", default=None, help="Optional JSON path for evaluator result metadata.")
    parser.add_argument(
        "--evaluator-command",
        default=None,
        help=(
            "Optional shell command to evaluate the prediction. Supports placeholders: "
            "{prediction_path}, {benchmark_input}, {benchmark_id}, {instance_id}, {task_id}, {model}."
        ),
    )
    parser.add_argument("--evaluator-timeout", type=float, default=3600.0)
    parser.add_argument(
        "--trajectory",
        default="auto",
        help="JSONL trajectory path. Defaults to trajectories/trajectory-<timestamp>-<pid>.jsonl.",
    )
    parser.add_argument("--no-trajectory", action="store_true", help="Disable trajectory writing.")
    parser.add_argument("--compact-after-chars", type=int, default=120_000)
    parser.add_argument("--force-compact", action="store_true")
    parser.add_argument("--language", choices=["auto", "en", "zh"], default="auto")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--dry-run",
        choices=["task_request", "compact_request"],
        default=None,
        help="Print the request that would be sent, without calling a model.",
    )
    return parser


def load_messages(args: argparse.Namespace) -> list[Message]:
    messages: list[Message] = []
    if args.messages:
        loaded = _read_json(args.messages)
        if isinstance(loaded, dict):
            loaded = loaded.get("messages")
        if not isinstance(loaded, list):
            raise SystemExit("--messages must be a JSON list or an object with a messages field")
        messages.extend(loaded)
    if args.system:
        messages.insert(0, {"role": "system", "content": args.system})
    prompt = args.prompt
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if prompt:
        messages.append({"role": "user", "content": prompt})
    if not messages:
        raise SystemExit("no task supplied; pass --prompt, pipe stdin, or pass --messages")
    return messages


def messages_char_count(messages: list[Message]) -> int:
    return sum(len(json.dumps(message, ensure_ascii=False)) for message in messages)


def split_for_compaction(messages: list[Message]) -> tuple[list[Message], list[Message], list[Message]]:
    system_prefix: list[Message] = []
    body = list(messages)
    while body and body[0].get("role") in {"system", "developer"}:
        system_prefix.append(body.pop(0))
    keep_tail = body[-1:] if body else []
    compact_target = body[:-1] if keep_tail else body
    return system_prefix, compact_target, keep_tail


def compact_messages_if_needed(
    messages: list[Message],
    compact_mode: Any,
    compact_model: str,
    compact_base_url: str | None,
    compact_api_key: str | None,
    timeout: float,
    language: str,
    threshold_chars: int,
    force: bool,
    post_json: Callable[[str, str | None, str, dict[str, Any], float], dict[str, Any]] = _post_json,
) -> tuple[list[Message], dict[str, Any] | None]:
    should_compact = force or messages_char_count(messages) >= threshold_chars
    if not should_compact:
        return messages, None

    system_prefix, compact_target, keep_tail = split_for_compaction(messages)
    if not compact_target:
        return messages, None

    compact_request = compact_mode.build_chat_request_from_messages(
        compact_target,
        model=compact_model,
        language=language,
    )
    if not compact_base_url:
        raise SystemExit("compaction is required but no --compact-base-url or COMPACT_BASE_URL was provided")
    compact_response = post_json(compact_base_url, compact_api_key, "/chat/completions", compact_request, timeout)
    raw_summary = compact_mode.extract_summary_text(compact_response)
    compact_json = compact_mode.wrap_compact_json(raw_summary)
    encrypted = compact_json["output"][0]["encrypted_content"]
    compacted_messages = system_prefix + [{"role": "user", "content": encrypted}] + keep_tail
    metadata = {
        "mode": compact_mode.name,
        "before_chars": messages_char_count(messages),
        "after_chars": messages_char_count(compacted_messages),
        "compacted_messages": len(compact_target),
        "compact_model": compact_model,
        "compact_request": compact_request,
        "compact_response_usage": _extract_usage(compact_response),
        "raw_summary": raw_summary,
        "encrypted_content": encrypted,
        "compacted_messages_output": compacted_messages,
    }
    return compacted_messages, metadata


def build_task_request(args: argparse.Namespace, messages: list[Message]) -> dict[str, Any]:
    if not args.model:
        raise SystemExit("missing task model; pass --model or set MAIN_MODEL")
    body: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": False,
        "max_tokens": args.max_tokens,
    }
    if args.temperature is not None:
        body["temperature"] = args.temperature
    return body


def extract_assistant_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return json.dumps(response, ensure_ascii=False)
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return json.dumps(response, ensure_ascii=False)
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def run_task_messages(
    args: argparse.Namespace,
    compact_mode: Any,
    messages: list[Message],
    trajectory: TrajectoryRecorder | None = None,
    post_json: Callable[[str, str | None, str, dict[str, Any], float], dict[str, Any]] = _post_json,
) -> TaskRunResult | None:
    compact_base_url = args.compact_base_url or args.base_url
    compact_api_key = _resolve_secret(args.compact_api_key, _resolve_secret(args.api_key))
    if args.dry_run == "compact_request":
        _, compact_target, _ = split_for_compaction(messages)
        if not compact_target:
            raise SystemExit("nothing to compact; need at least one historical non-system message before the latest user turn")
        compact_request = compact_mode.build_chat_request_from_messages(
            compact_target,
            model=args.compact_model,
            language=args.language,
        )
        if trajectory:
            trajectory.record("dry_run_compact_request", {"compact_request": compact_request})
        print(json.dumps(compact_request, ensure_ascii=False, indent=2))
        return None

    original_messages = list(messages)
    messages, compact_metadata = compact_messages_if_needed(
        messages,
        compact_mode=compact_mode,
        compact_model=args.compact_model,
        compact_base_url=compact_base_url,
        compact_api_key=compact_api_key,
        timeout=args.timeout,
        language=args.language,
        threshold_chars=args.compact_after_chars,
        force=args.force_compact,
        post_json=post_json,
    )
    if trajectory:
        if compact_metadata:
            trajectory.record("context_compacted", compact_metadata)
        else:
            trajectory.record(
                "context_compaction_skipped",
                {
                    "before_chars": messages_char_count(original_messages),
                    "threshold_chars": args.compact_after_chars,
                    "force_compact": args.force_compact,
                },
            )
    task_request = build_task_request(args, messages)
    if args.dry_run == "task_request":
        if trajectory:
            trajectory.record("dry_run_task_request", {"compact": compact_metadata, "task_request": task_request})
        print(json.dumps({"compact": compact_metadata, "request": task_request}, ensure_ascii=False, indent=2))
        return None

    if not args.base_url:
        raise SystemExit("missing task base URL; pass --base-url or set OPENAI_BASE_URL")
    task_response = post_json(
        args.base_url,
        _resolve_secret(args.api_key),
        "/chat/completions",
        task_request,
        args.timeout,
    )
    if trajectory:
        trajectory.record("task_response", {"task_request": task_request, "task_response": task_response})
    assistant_text = extract_assistant_text(task_response)
    messages.append({"role": "assistant", "content": assistant_text})
    return TaskRunResult(
        messages=messages,
        assistant_text=assistant_text,
        compact_metadata=compact_metadata,
        task_request=task_request,
        task_response=task_response,
    )


def benchmark_messages(args: argparse.Namespace, task: BenchmarkTask) -> list[Message]:
    system = args.system if args.system is not None else task.system
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(task.messages)
    return messages


def run_benchmark(
    args: argparse.Namespace,
    compact_mode: Any,
    post_json: Callable[[str, str | None, str, dict[str, Any], float], dict[str, Any]] = _post_json,
) -> int:
    if args.messages:
        raise SystemExit("benchmark mode builds messages from --benchmark-input; --messages is not supported")
    adapter = get_benchmark(args.benchmark)
    task = adapter.load_task(args.benchmark_input, args.benchmark_id, extra_prompt=args.prompt)
    messages = benchmark_messages(args, task)
    trajectory = None if args.no_trajectory else TrajectoryRecorder(args.trajectory)
    if trajectory:
        trajectory.record(
            "run_start",
            {
                "model": args.model,
                "compact_mode": args.compact_mode,
                "compact_model": args.compact_model,
                "compact_after_chars": args.compact_after_chars,
                "force_compact": args.force_compact,
                "dry_run": args.dry_run,
                "benchmark": task.benchmark,
                "benchmark_id": task.task_id,
                "benchmark_metadata": task.metadata,
                "initial_messages": messages,
            },
        )

    result = run_task_messages(args, compact_mode, messages, trajectory=trajectory, post_json=post_json)
    if result is None:
        return 0

    prediction = adapter.build_prediction(task, result.assistant_text, args.model)
    prediction_path = write_prediction(args.benchmark_output or adapter.default_output_path(task), prediction)
    evaluation = None
    if args.evaluator_command:
        evaluation = run_evaluator_command(
            args.evaluator_command,
            prediction_path=prediction_path,
            input_path=args.benchmark_input,
            task=task,
            model_name=args.model,
            timeout=args.evaluator_timeout,
        )
        if args.benchmark_results:
            _write_json(args.benchmark_results, evaluation)
    if args.save_messages:
        _write_json(args.save_messages, result.messages)
    summary = {
        "benchmark": task.benchmark,
        "benchmark_id": task.task_id,
        "prediction_path": str(prediction_path),
        "prediction": prediction,
        "evaluation": evaluation,
    }
    if trajectory:
        trajectory.record("benchmark_prediction", summary)
        trajectory.record(
            "run_end",
            {
                "final_messages": result.messages,
                "assistant_text": result.assistant_text,
                "trajectory_path": str(trajectory.path),
                "benchmark": task.benchmark,
                "benchmark_id": task.task_id,
            },
        )
        print(f"trajectory: {trajectory.path}", file=sys.stderr)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _extract_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else None


def main(argv: list[str] | None = None) -> int:
    argv = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    args = build_parser().parse_args(argv)
    compact_mode = get_mode(args.compact_mode)
    if args.benchmark:
        return run_benchmark(args, compact_mode)
    messages = load_messages(args)
    trajectory = None if args.no_trajectory else TrajectoryRecorder(args.trajectory)
    if trajectory:
        trajectory.record(
            "run_start",
            {
                "model": args.model,
                "compact_mode": args.compact_mode,
                "compact_model": args.compact_model,
                "compact_after_chars": args.compact_after_chars,
                "force_compact": args.force_compact,
                "dry_run": args.dry_run,
                "initial_messages": messages,
            },
        )

    result = run_task_messages(args, compact_mode, messages, trajectory=trajectory)
    if result is None:
        return 0
    if trajectory:
        trajectory.record(
            "run_end",
            {
                "final_messages": result.messages,
                "assistant_text": result.assistant_text,
                "trajectory_path": str(trajectory.path),
            },
        )
        print(f"trajectory: {trajectory.path}", file=sys.stderr)
    if args.save_messages:
        _write_json(args.save_messages, result.messages)
    print(result.assistant_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
