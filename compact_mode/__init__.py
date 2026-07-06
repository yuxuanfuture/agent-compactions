from __future__ import annotations

from typing import Protocol

from compact_mode.claude_code import ClaudeCodeMode
from compact_mode.codex_remote_v2.mode import CodexRemoteV2Mode
from compact_mode.hermes_harness import HermesHarnessMode
from compact_mode.open_claw import OpenClawMode


class CompactMode(Protocol):
    name: str


_MODES = {
    "codex_remote_v2": CodexRemoteV2Mode(),
    "claude_code": ClaudeCodeMode(),
    "hermes_harness": HermesHarnessMode(),
    "open_claw": OpenClawMode(),
}


def mode_names() -> list[str]:
    return sorted(_MODES)


def get_mode(name: str):
    try:
        return _MODES[name]
    except KeyError as exc:
        known = ", ".join(mode_names())
        raise SystemExit(f"unknown compact mode {name!r}; known modes: {known}") from exc
