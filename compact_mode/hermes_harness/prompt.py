from __future__ import annotations

MAX_OUTPUT_TOKENS = 1_500
MAX_RENDERED_HISTORY_CHARS = 120_000
MAX_SINGLE_FIELD_CHARS = 12_000
SUMMARY_TARGET_TOKENS = 750

CONTEXT_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"


HERMES_SUMMARIZATION_PROMPT_EN = f"""Summarize the following agent conversation turns concisely.

This summary will replace these turns in the conversation history. Write the summary from a neutral perspective describing what the assistant did and learned.

Include:
1. What actions the assistant took, including tool calls, searches, and file operations.
2. Key information or results obtained.
3. Any important decisions or findings.
4. Relevant data, file names, values, or outputs.

Keep the summary factual and informative. Target approximately {SUMMARY_TARGET_TOKENS} tokens.

Write only the summary, starting with the exact "{CONTEXT_SUMMARY_PREFIX}" prefix."""


HERMES_SUMMARIZATION_PROMPT_ZH = f"""请简明总结下面这些 agent 对话轮次。

这份总结会替换这些轮次，放回后续对话上下文。请用中性视角描述 assistant 做了什么、查到了什么、学到了什么。

必须包含：
1. assistant 采取的行动，包括工具调用、搜索、文件操作。
2. 获得的关键信息或结果。
3. 重要决策或发现。
4. 相关数据、文件名、具体值、关键输出。

保持事实性和信息密度，目标约 {SUMMARY_TARGET_TOKENS} tokens。

只输出 summary，并且必须以精确的 "{CONTEXT_SUMMARY_PREFIX}" 前缀开头。"""


def choose_prompt(language: str, rendered_history: str) -> str:
    if language == "zh":
        return HERMES_SUMMARIZATION_PROMPT_ZH
    if language == "en":
        return HERMES_SUMMARIZATION_PROMPT_EN
    if any("\u4e00" <= ch <= "\u9fff" for ch in rendered_history):
        return HERMES_SUMMARIZATION_PROMPT_ZH
    return HERMES_SUMMARIZATION_PROMPT_EN
