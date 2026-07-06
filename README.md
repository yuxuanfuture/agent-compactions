# compact

这是一个“主模型执行任务 + 独立压缩模型做上下文压缩”的实验工程。

核心目标：`main.py` 负责调用真正执行任务的大模型，例如 GLM 5.2、GPT 5.5；`--compact_mode` 只决定上下文需要压缩时采用哪种压缩格式。当前已实现 `codex_remote_v2`、`claude_code`、`hermes_harness`、`open_claw` 四种压缩模式，可让 Qwen3-30B-A3B 这类专门压缩模型产出对应 agent harness 可接续的 summary。

## 目录结构

- `main.py`：主入口。正常执行任务时用这个文件。
- `compact.py`：协议调试工具。只在需要单独检查 Codex compact payload 或 SSE 包装时使用。
- `compact_mode/codex_remote_v2/`：已实现的 Codex remote compaction v2 压缩模式。
- `compact_mode/claude_code/`：Claude Code 风格 compact 模式。
- `compact_mode/hermes_harness/`：Hermes Harness 风格 compact 模式。
- `compact_mode/open_claw/`：OpenClaw 风格 compact 模式。
- `tests/`：单元测试。

## 基本用法

主模型执行任务，Qwen3-30B-A3B 作为压缩模型：

```bash
python main.py \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --compact-model Qwen3-30B-A3B \
  --compact-base-url http://compact-model.local/v1 \
  --compact_mode codex_remote_v2 \
  --prompt "你的任务"
```

这里的含义是：

- `--model GLM-5.2`：真正执行任务的大模型。
- `--base-url http://task-model.local/v1`：任务模型的 OpenAI-compatible API 地址。
- `--compact-model Qwen3-30B-A3B`：专门用于上下文压缩的模型。
- `--compact-base-url http://compact-model.local/v1`：压缩模型的 OpenAI-compatible API 地址。
- `--compact_mode codex_remote_v2`：压缩时使用 Codex remote compaction v2 风格。
- `--prompt "你的任务"`：本轮要交给主模型执行的任务。

也兼容这个写法：

```bash
python main.py -- compact_mode codex_remote_v2 ...
```

但推荐统一写成：

```bash
--compact_mode codex_remote_v2
```

## 轨迹文件 trajectory

`main.py` 每次运行默认都会保存 trajectory，路径格式：

```text
trajectories/trajectory-<timestamp>-<pid>.jsonl
```

你也可以手动指定：

```bash
python main.py ... --trajectory runs/exp1.jsonl
```

禁用 trajectory：

```bash
python main.py ... --no-trajectory
```

trajectory 是 JSONL，每行一个事件。常见事件：

- `run_start`：记录启动参数、初始 messages。
- `context_compaction_skipped`：本轮没有触发压缩。
- `context_compacted`：本轮触发了压缩。
- `task_response`：记录主模型请求和响应。
- `run_end`：记录最终 messages 和 assistant 输出。

压缩发生时，`context_compacted` 会保存关键内容：

- `raw_summary`：压缩模型原始输出。
- `encrypted_content`：按当前 compact mode 包装后的最终压缩内容。
- `compact_request`：发给 Qwen3-30B-A3B 的压缩请求。
- `compact_response_usage`：压缩模型返回的 token usage，如果上游提供。
- `compacted_messages_output`：真正送回任务模型的压缩后上下文。

## 什么时候会压缩

默认规则：当当前 messages 序列化后的字符数超过 `--compact-after-chars` 时触发压缩。

默认阈值：

```text
120000
```

修改阈值：

```bash
python main.py ... --compact-after-chars 80000
```

强制压缩：

```bash
python main.py ... --force-compact
```

压缩时，`main.py` 会保留 system/developer 前缀和最新用户请求，把中间历史发给压缩模型总结，再把压缩结果作为一条 user message 放回上下文，继续交给主模型执行任务。

## Dry Run

查看最终会发给主模型的请求，不真正调用模型：

```bash
python main.py \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --compact_mode codex_remote_v2 \
  --prompt "你的任务" \
  --dry-run task_request
```

查看会发给压缩模型的请求：

```bash
python main.py \
  --dry-run compact_request \
  --compact_mode codex_remote_v2 \
  --compact-model Qwen3-30B-A3B \
  --messages transcript.json \
  --prompt "继续执行"
```

## Benchmark 主任务

`main.py` 现在也可以把 benchmark instance 当成主任务运行。普通 `--prompt` 流程仍然保留；指定 `--benchmark` 后，入口会改为：

```text
benchmark adapter -> 生成主模型 messages -> 可选 compaction -> 调用主模型 -> 保存 prediction -> 可选 evaluator
```

### SWE-bench

输入可以是单条 JSON、JSON list，或 JSONL。字段按 SWE-bench 常见格式读取：`instance_id`、`repo`、`base_commit`、`problem_statement`、`hints_text`、`FAIL_TO_PASS`、`PASS_TO_PASS`。

```bash
python main.py \
  --benchmark swe_bench \
  --benchmark-input data/swebench-lite.jsonl \
  --instance-id django__django-12345 \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --compact-model Qwen3-30B-A3B \
  --compact-base-url http://compact-model.local/v1 \
  --compact_mode codex_remote_v2
```

输出 prediction 默认写到：

```text
predictions/swe_bench-<instance-id>-<timestamp>-<pid>.jsonl
```

每行格式兼容 SWE-bench prediction 的核心字段：

```json
{"instance_id":"...","model_name_or_path":"...","model_patch":"..."}
```

如果要接外部 evaluator，可传命令模板：

```bash
python main.py \
  --benchmark swe_bench \
  --benchmark-input data/swebench-lite.jsonl \
  --instance-id django__django-12345 \
  --benchmark-output runs/swe_predictions.jsonl \
  --evaluator-command 'python -m swebench.harness.run_evaluation --predictions_path {prediction_path}' \
  --benchmark-results runs/swe_eval_result.json \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1
```

### tau-bench

tau-bench 使用同一个主入口切换：

```bash
python main.py \
  --benchmark tau_bench \
  --benchmark-input data/taubench-tasks.jsonl \
  --task-id task_001 \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --compact_mode claude_code
```

tau-bench adapter 会读取 `task_id` / `id`、`domain`、`instruction` / `user_goal` / `goal`、`tools`、`initial_state` / `environment` 等字段，把完整 task JSON 放进 prompt，并保存：

```json
{"task_id":"...","model":"...","answer":"..."}
```

如果模型返回 JSON，prediction 里会额外保存 `response_json`。真实 tau-bench 环境执行通常需要外部 harness，可以通过 `--evaluator-command` 接入。

可用的 evaluator 命令占位符：

- `{prediction_path}`
- `{benchmark_input}` / `{input_path}`
- `{benchmark_id}` / `{instance_id}` / `{task_id}`
- `{model}`

## 环境变量

也可以用环境变量减少命令行参数：

```bash
export OPENAI_BASE_URL=http://task-model.local/v1
export OPENAI_API_KEY=...
export MAIN_MODEL=GLM-5.2
export COMPACT_BASE_URL=http://compact-model.local/v1
export COMPACT_API_KEY=...
export COMPACT_MODEL=Qwen3-30B-A3B
```

然后运行：

```bash
python main.py \
  --compact_mode codex_remote_v2 \
  --prompt "你的任务"
```

切换为 Claude Code 风格 compact：

```bash
python main.py \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --compact-model Qwen3-30B-A3B \
  --compact-base-url http://compact-model.local/v1 \
  --compact_mode claude_code \
  --prompt "你的任务"
```

也兼容常见拼写错误：

```bash
--compatct_mode claude_code
python main.py -- compatct_mode claude_code ...
```

但推荐统一写成：

```bash
--compact_mode claude_code
```

切换为 Hermes Harness 风格 compact：

```bash
python main.py \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --compact-model Qwen3-30B-A3B \
  --compact-base-url http://compact-model.local/v1 \
  --compact_mode hermes_harness \
  --prompt "你的任务"
```

也兼容分隔符写法：

```bash
python main.py -- compact_mode hermes_harness ...
```

同样兼容常见拼写错误：

```bash
--compatct_mode hermes_harness
python main.py -- compatct_mode hermes_harness ...
```

但推荐统一写成：

```bash
--compact_mode hermes_harness
```

切换为 OpenClaw 风格 compact：

```bash
python main.py \
  --model GLM-5.2 \
  --base-url http://task-model.local/v1 \
  --compact-model Qwen3-30B-A3B \
  --compact-base-url http://compact-model.local/v1 \
  --compact_mode open_claw \
  --prompt "你的任务"
```

也兼容分隔符写法：

```bash
python main.py -- compact_mode open_claw ...
```

同样兼容常见拼写错误：

```bash
--compatct_mode open_claw
python main.py -- compatct_mode open_claw ...
```

但推荐统一写成：

```bash
--compact_mode open_claw
```

## 关于 codex_remote_v2

`codex_remote_v2` 复刻的是 Codex remote compaction v2 的压缩语义：

- 使用 Codex 风格的 context checkpoint prompt。
- 压缩结果前面保留 Codex 识别用的英文 summary prefix。
- 对 Qwen/Qwen3 模型默认注入 `enable_thinking=false`，避免压缩模型把输出预算花在 reasoning trace 上。
- `compact.py` 调试工具还可以把 summary 包成 Codex V2 SSE：`response.created -> response.output_item.done -> response.completed`，其中只包含一个 `type=compaction` item。

普通任务执行不要直接用 `compact.py`，应该用 `main.py`。

## 关于 claude_code

`claude_code` 模式把历史压缩成 Claude Code `/compact` 风格的结构化 summary：

- 压缩模型请求包含 no-tools guard，要求只返回 `<analysis>` 和 `<summary>` 文本。
- summary 结构保留 Claude Code compact 的核心字段：用户意图、技术概念、文件/代码、错误修复、问题解决、所有用户消息、待办、当前/已完成工作、继续上下文。
- `main.py` 会保留最新用户请求，所以 `claude_code` 在主流程里使用 partial-compaction 语义：只总结较早历史，最新 user message 仍作为真实下一轮请求跟在 summary 后面。
- 压缩模型输出中的 `<analysis>` 会被丢弃，只把 `<summary>` 作为继续上下文放回任务模型。

Prompt 结构参考 Piebald-AI 的 Claude Code system prompt inventory：
`agent-prompt-conversation-summarization.md`、`agent-prompt-summarization-no-tools-guard.md`、`system-prompt-partial-compaction-instructions.md`。

## 关于 hermes_harness

`hermes_harness` 模式把 Hermes Agent 的 trajectory compressor 语义做成 `main.py` 可直接调用的 compact mode：

- 压缩模型请求使用 Hermes Harness 风格的 turn 渲染：`[Turn N - HUMAN/GPT/TOOL]`。
- summary prompt 要求中性事实视角，保留 assistant 行动、工具/搜索/文件操作、关键结果、决策、文件名、具体值和输出。
- 压缩结果会归一化为以 `[CONTEXT SUMMARY]:` 开头的一条 user message，放回任务模型上下文。
- `main.py` 仍保留最新 user message，所以这里也是 partial-compaction：只总结较早历史，最新请求继续作为真实下一轮输入。
- 对 Qwen/Qwen3 模型默认注入 `enable_thinking=false`。

参考来源是 NousResearch/hermes-agent 的 `trajectory_compressor.py`：它将被压缩区间替换成一条 human summary message，并要求 summary 使用 `[CONTEXT SUMMARY]:` 前缀。

## 关于 open_claw

`open_claw` 模式把 OpenClaw 的 compaction summary 语义做成 `main.py` 可直接调用的 compact mode：

- 压缩模型请求使用 OpenClaw context checkpoint prompt，并加入 safeguard 风格的结构化 summary、identifier preservation 要求。
- summary 输出会被包装成 OpenClaw 继续上下文格式：`The conversation history before this point was compacted into the following summary:` 加 `<summary>...</summary>`。
- validation 接受 OpenClaw 基础 checkpoint sections：`Goal / Constraints & Preferences / Progress / Key Decisions / Next Steps / Critical Context`，也接受当前 safeguard sections：`Decisions / Open TODOs / Constraints/Rules / Pending user asks / Exact identifiers`。
- `main.py` 仍保留最新 user message，所以这里也是 partial-compaction：只总结较早历史，最新请求继续作为真实下一轮输入。
- 对 Qwen/Qwen3 模型默认注入 `enable_thinking=false`。

## 调试 compact.py

只有在你需要单独调试 Codex remote compaction v2 payload/SSE 包装时才用：

```bash
python compact.py \
  --compact_mode codex_remote_v2 \
  --input codex-request.json \
  --summary-file summary.md \
  --output compact-response.sse
```

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q compact.py main.py compact_mode tests
```
