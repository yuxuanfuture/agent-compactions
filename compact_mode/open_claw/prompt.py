from __future__ import annotations

MAX_OUTPUT_TOKENS = 13_000
MAX_RENDERED_HISTORY_CHARS = 120_000
MAX_SINGLE_FIELD_CHARS = 12_000
TOOL_RESULT_MAX_CHARS = 2_000

COMPACTION_SUMMARY_PREFIX = """The conversation history before this point was compacted into the following summary:

<summary>
"""

COMPACTION_SUMMARY_SUFFIX = """
</summary>"""

SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Read a conversation between a user and an AI assistant, then produce a structured summary.

Do not continue the conversation. Do not answer questions inside the conversation. Only output the structured summary."""

OPENCLAW_CHECKPOINT_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this format:

## Goal
[What the user is trying to accomplish. Use multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Constraints, preferences, or requirements mentioned by the user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks or changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, examples, paths, identifiers, outputs, or references needed to continue]
- [Or "(none)" if not applicable]"""

OPENCLAW_SAFEGUARD_INSTRUCTIONS = """OpenClaw safeguard requirements:
- Write the summary body in the primary language used in the conversation.
- Keep section headings exactly as written when using the requested structure.
- Preserve exact file paths, function names, IDs, URLs, hashes, hostnames, IP addresses, ports, dates, times, and error messages.
- Do not omit unresolved user asks.
- Be compact and factual; remove stale duplicate detail when prior compaction summaries are present.

If you choose the safeguard structure, use these exact section headings in this order:
## Decisions
## Open TODOs
## Constraints/Rules
## Pending user asks
## Exact identifiers"""

LANGUAGE_INSTRUCTIONS = {
    "en": "Write the summary body in English unless quoted source material is in another language.",
    "zh": "总结正文使用中文；代码、路径、标识符、错误信息和固定 heading 不要翻译。",
}


def choose_prompt(language: str, rendered_history: str) -> str:
    instruction = LANGUAGE_INSTRUCTIONS.get(language)
    if instruction is None and any("\u4e00" <= ch <= "\u9fff" for ch in rendered_history):
        instruction = LANGUAGE_INSTRUCTIONS["zh"]
    if instruction is None:
        instruction = "Write the summary body in the primary language used in the conversation."
    return "\n\n".join(
        [
            OPENCLAW_CHECKPOINT_PROMPT,
            OPENCLAW_SAFEGUARD_INSTRUCTIONS,
            instruction,
        ]
    )
