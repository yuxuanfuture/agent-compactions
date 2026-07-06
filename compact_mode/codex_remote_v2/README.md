# codex_remote_v2

This mode reproduces the Codex remote compaction v2 wire shape:

1. Detect a regular `/responses` request whose `input` contains `{"type":"compaction_trigger"}`.
2. Strip that trigger item.
3. Render the remaining Responses input as a compactable transcript.
4. Append the Codex-style context checkpoint prompt as the final user message.
5. Wrap the summary model output as Responses SSE:
   `response.created` -> `response.output_item.done` with one `compaction` item -> `response.completed`.

Example:

```bash
python compact.py \
  --compact_mode codex_remote_v2 \
  --input codex-request.json \
  --base-url http://localhost:8000/v1 \
  --model Qwen3-30B-A3B \
  --output compact-response.sse
```

To inspect the request that will be sent to the compact model:

```bash
python compact.py --compact_mode codex_remote_v2 --input codex-request.json --emit chat_request
```
