# claude_code

Claude Code-style compact mode for `main.py`.

This mode builds an OpenAI-compatible chat request for the compact model using
Claude Code's compact-summary structure:

1. A no-tools guard.
2. The rendered conversation history.
3. A prompt that asks for `<analysis>` followed by `<summary>`.

When used through `main.py`, the latest user turn is preserved outside the
compacted history, so this mode uses partial-compaction semantics: summarize the
older history as context, then let the newest user message follow as the next
real request.

Example:

```bash
python main.py \
  --compact_mode claude_code \
  --compact-model Qwen3-30B-A3B \
  --compact-base-url http://compact-model.local/v1 \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --prompt "继续执行"
```
