from __future__ import annotations

MAX_OUTPUT_TOKENS = 20_000
MAX_RENDERED_HISTORY_CHARS = 120_000
MAX_SINGLE_FIELD_CHARS = 12_000


NO_TOOLS_GUARD = """Return plain text only. Do not call tools.

- Do not use file, shell, search, edit, or other tools.
- The transcript supplied in this request is all the context needed.
- The response must contain an <analysis> block followed by a <summary> block.
"""


CONVERSATION_SUMMARIZATION_PROMPT = """Create a detailed continuation summary for the conversation so far.

The goal is to preserve enough context for another coding agent to resume the work without re-reading the original transcript. In the <analysis> block, review the conversation chronologically and check user intent, agent actions, technical decisions, files, code edits, errors, fixes, and any security-sensitive constraints.

The <summary> block must use this structure:

1. Primary Request and Intent: describe the user's explicit requests and success criteria.
2. Key Technical Concepts: list important technologies, frameworks, APIs, and code patterns.
3. Files and Code Sections: name files examined, created, or modified; include important snippets or signatures when they are needed to continue accurately.
4. Errors and fixes: record errors, failed attempts, corrections, and relevant user feedback.
5. Problem Solving: summarize solved problems and active troubleshooting threads.
6. All user messages: preserve every non-tool user message in chronological order, with security-relevant constraints verbatim.
7. Pending Tasks: list tasks the user explicitly asked for that remain incomplete.
8. Current Work: explain exactly what was being worked on immediately before compaction.
9. Optional Next Step: include only the next action that follows directly from the most recent explicit request; quote the recent user text that anchors that action.

Write in the same language as the conversation unless the user requested otherwise. Be precise, structured, and complete enough to prevent repeated work.
"""


PARTIAL_COMPACTION_PROMPT = """Create a detailed summary of this earlier part of the conversation.

This summary will be placed before newer messages that are not shown in this compact request, so it must explain the prior context without pretending to know what happened later. In the <analysis> block, review this portion chronologically and check user intent, agent actions, technical decisions, files, code edits, errors, fixes, and any security-sensitive constraints.

The <summary> block must use this structure:

1. Primary Request and Intent: describe the user's explicit requests and intent in this portion.
2. Key Technical Concepts: list important technologies, frameworks, APIs, and code patterns.
3. Files and Code Sections: name files examined, created, or modified; include important snippets or signatures when needed.
4. Errors and fixes: record errors, failed attempts, corrections, and relevant user feedback.
5. Problem Solving: summarize solved problems and ongoing troubleshooting.
6. All user messages: preserve every non-tool user message in chronological order, with security-relevant constraints verbatim.
7. Pending Tasks: list tasks from this portion that remain incomplete.
8. Work Completed: describe what was accomplished by the end of this portion.
9. Context for Continuing Work: capture decisions, state, and details needed by the newer messages that will follow.

Write in the same language as the conversation unless the user requested otherwise. Be precise, structured, and complete enough to prevent repeated work.
"""


COMPACTED_CONTEXT_HEADER = "Summary:"
