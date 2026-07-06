from __future__ import annotations

COMPACT_SUMMARIZATION_PROMPT_EN = """You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

Include:
- Current progress and key decisions made
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue
- **All user messages so far, verbatim or near-verbatim, in chronological order** - this preserves intent shifts that get lost otherwise
- **Next Step** - the immediate next action aligned with the user's most recent explicit request. Include a **verbatim direct quote** from the most recent user message showing exactly where you left off; this prevents task drift.

Be concise, structured, and focused on helping the next LLM seamlessly continue the work."""


COMPACT_SUMMARIZATION_PROMPT_ZH = """你正在执行 CONTEXT CHECKPOINT COMPACTION(上下文检查点压缩)。为下一个接手任务的 LLM 写一份交接总结。

包含:
- 当前进度和已做出的关键决策
- 重要 context、约束、或 user 偏好
- 还有什么待办(清晰的下一步)
- 继续任务所需的关键数据、示例、引用
- **截至目前的所有 user message,按时间顺序逐字或近似逐字保留** - 这能保留其它方式会丢失的 intent 演变
- **Next Step** - 跟 user 最近一次显式请求对齐的下一个动作。包含从 user 最近一条 message 中**逐字引用**的直接 quote,标明你停在了哪里;这能防止任务漂移。

精简、结构化,聚焦于帮助下一个 LLM 无缝接续工作。"""


# Codex recognizes compaction summaries by this exact English prefix.
COMPACT_SUMMARY_PREFIX = "Another language model started to solve this problem and produced a summary of its thinking process. You also have access to the state of the tools that were used by that language model. Use this to build on the work that has already been done and avoid duplicating work. Here is the summary produced by the other language model, use the information in this summary to assist with your own analysis:"


def choose_prompt(language: str, rendered_history: str) -> str:
    if language == "zh":
        return COMPACT_SUMMARIZATION_PROMPT_ZH
    if language == "en":
        return COMPACT_SUMMARIZATION_PROMPT_EN
    if any("\u4e00" <= ch <= "\u9fff" for ch in rendered_history):
        return COMPACT_SUMMARIZATION_PROMPT_ZH
    return COMPACT_SUMMARIZATION_PROMPT_EN
