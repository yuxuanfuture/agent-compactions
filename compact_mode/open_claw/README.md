# open_claw

OpenClaw-style direct compact mode for `main.py`.

This mode renders the compact target as an OpenClaw-style conversation block,
asks the compact model to produce a structured checkpoint summary, and wraps the
result as:

```text
The conversation history before this point was compacted into the following summary:

<summary>
...
</summary>
```

It accepts both the base OpenClaw checkpoint sections and the newer safeguard
sections during validation.
