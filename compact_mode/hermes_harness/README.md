# hermes_harness

Hermes Harness-style compact mode for `main.py`.

This mode adapts the summary semantics from NousResearch/hermes-agent's
`trajectory_compressor.py` into the direct compact-mode interface used by this
repo:

1. Render chat history as Hermes-style turns: `[Turn N - HUMAN/GPT/TOOL]`.
2. Ask the compact model for a neutral factual summary of actions, tool calls,
   file operations, results, decisions, and concrete values.
3. Normalize the model output so the compacted context starts with
   `[CONTEXT SUMMARY]:`.
4. Return the summary as a single compaction item so `main.py` can place it
   before the latest preserved user request.

Usage:

```bash
python main.py \
  --compact_mode hermes_harness \
  --compact-model Qwen3-30B-A3B \
  --compact-base-url http://compact-model.local/v1 \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --prompt "your task"
```
