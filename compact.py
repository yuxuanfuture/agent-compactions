from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from compact_mode import get_mode, mode_names


def _read_text(path: str | None) -> str:
    if not path or path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def _write_text(path: str | None, text: str) -> None:
    if not path or path == "-":
        sys.stdout.write(text)
        if text and not text.endswith("\n"):
            sys.stdout.write("\n")
        return
    Path(path).write_text(text, encoding="utf-8")


def _read_json(path: str | None) -> Any:
    text = _read_text(path)
    if not text.strip():
        raise SystemExit("empty JSON input")
    return json.loads(text)


def _resolve_api_key(raw: str | None) -> str | None:
    if not raw:
        return None
    if raw.startswith("env:"):
        return os.environ.get(raw[4:])
    return raw


def _post_json(base_url: str, api_key: str | None, path: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"upstream HTTP {exc.code}: {payload[:1000]}") from exc
    return json.loads(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compact-mode runner. Use --compact_mode codex_remote_v2 for Codex remote compaction v2.",
    )
    parser.add_argument(
        "--compact_mode",
        default="codex_remote_v2",
        choices=mode_names(),
        help="Compact mode implementation to use.",
    )
    parser.add_argument(
        "--input",
        default="-",
        help="Codex Responses request JSON path. Defaults to stdin. Use '-' for stdin.",
    )
    parser.add_argument(
        "--path",
        default="/responses",
        help="Client path used for mode detection, e.g. /responses or /v1/responses.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override request model in the compact chat request.",
    )
    parser.add_argument(
        "--language",
        choices=["auto", "en", "zh"],
        default="auto",
        help="Summarization prompt language. auto picks zh when CJK appears in user text.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output path. Defaults to stdout. Use '-' for stdout.",
    )
    parser.add_argument(
        "--emit",
        choices=["auto", "chat_request", "sse", "compact_json"],
        default="auto",
        help=(
            "What to emit. auto emits chat_request unless --summary/--summary-file/"
            "--upstream-response/--base-url is supplied."
        ),
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Raw model-produced summary text to wrap into Codex remote compaction v2 SSE.",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="File containing raw model-produced summary text.",
    )
    parser.add_argument(
        "--upstream-response",
        default=None,
        help="OpenAI-compatible chat/Gemini/Anthropic response JSON to extract summary from.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1.",
    )
    parser.add_argument(
        "--api-key",
        default="env:OPENAI_API_KEY",
        help="API key or env:NAME. Defaults to env:OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Upstream HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--min-summary-chars",
        type=int,
        default=800,
        help="Minimum summary length before wrapping as a successful compaction.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip summary quality checks when wrapping model output.",
    )
    parser.add_argument(
        "--no-disable-thinking",
        action="store_true",
        help="Do not inject enable_thinking=false for Qwen compact chat requests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = get_mode(args.compact_mode)

    try:
        request_body = _read_json(args.input)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input is not valid JSON: {exc}") from exc

    validate_summary = not args.no_validate
    disable_thinking = not args.no_disable_thinking
    emit = args.emit
    if emit == "auto":
        emit = (
            "sse"
            if args.summary or args.summary_file or args.upstream_response or args.base_url
            else "chat_request"
        )

    if emit == "chat_request":
        try:
            chat_request = mode.build_chat_request(
                request_body,
                client_path=args.path,
                model_override=args.model,
                language=args.language,
                disable_thinking=disable_thinking,
            )
        except (ValueError, NotImplementedError) as exc:
            raise SystemExit(str(exc)) from exc
        _write_text(args.output, json.dumps(chat_request, ensure_ascii=False, indent=2))
        return 0

    summary_text = args.summary
    upstream_json: Any | None = None
    if args.summary_file:
        summary_text = _read_text(args.summary_file)
    elif args.upstream_response:
        upstream_json = _read_json(args.upstream_response)
    elif args.base_url:
        try:
            chat_request = mode.build_chat_request(
                request_body,
                client_path=args.path,
                model_override=args.model,
                language=args.language,
                disable_thinking=disable_thinking,
            )
        except (ValueError, NotImplementedError) as exc:
            raise SystemExit(str(exc)) from exc
        upstream_json = _post_json(
            args.base_url,
            _resolve_api_key(args.api_key),
            "/chat/completions",
            chat_request,
            args.timeout,
        )

    if emit == "compact_json":
        try:
            if summary_text is None:
                if upstream_json is None:
                    raise SystemExit("compact_json requires --summary, --summary-file, --upstream-response, or --base-url")
                summary_text = mode.extract_summary_text(upstream_json)
            compact_json = mode.wrap_compact_json(
                summary_text,
                validate=validate_summary,
                min_summary_chars=args.min_summary_chars,
            )
        except (ValueError, NotImplementedError) as exc:
            raise SystemExit(str(exc)) from exc
        _write_text(args.output, json.dumps(compact_json, ensure_ascii=False, indent=2))
        return 0

    if emit == "sse":
        if summary_text is None:
            if upstream_json is None:
                raise SystemExit("sse requires --summary, --summary-file, --upstream-response, or --base-url")
            sse = mode.wrap_upstream_response_as_sse(
                upstream_json,
                validate=validate_summary,
                min_summary_chars=args.min_summary_chars,
            )
        else:
            sse = mode.wrap_summary_as_sse(
                summary_text,
                validate=validate_summary,
                min_summary_chars=args.min_summary_chars,
            )
        _write_text(args.output, sse)
        return 0

    raise SystemExit(f"unsupported emit mode: {emit}")


if __name__ == "__main__":
    raise SystemExit(main())
