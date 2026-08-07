# Observability

Three independent layers. Each degrades on its own; none is required.

## Logs

Structured, to stdout (the Space's **Logs** tab) and to `logs/agent.log`
(rotating, 5 MB × 3).

```
2026-08-06 17:15:48 | INFO | agent.core.graph | step 2/4 -> web_agent (needs a fact lookup)
2026-08-06 17:15:49 | INFO | agent.tools.web  | web_search: moons of Mars
2026-08-06 17:15:52 | WARNING | agent.core.graph | Supervisor step budget (4) exhausted - finishing.
```

Set `LOG_LEVEL=DEBUG` for more. Third-party loggers are pinned to `WARNING` so
they do not drown the signal.

## Traces

Set `LANGSMITH_API_KEY` and every task becomes one trace named `task:<task_id>`,
tagged `agent`/`supervisor`, with `task_id`, `model` and `provider` in metadata.

Both the legacy `LANGCHAIN_*` and current `LANGSMITH_*` variable families are
written, because which one the installed LangChain reads depends on its version.

A trace shows each supervisor decision, each tool call and its result, and where
the time and tokens went — which is the fastest way to find out why a task was
slow or wrong.

## Metrics

One JSON row per task appended to `logs/metrics.jsonl`:

```json
{"task_id": "8e867cd7", "status": "ok", "latency_s": 34.2,
 "tokens": {"input_tokens": 8412, "output_tokens": 210, "total_tokens": 8622},
 "model": "llama-3.3-70b-versatile", "answer": "3"}
```

Aggregate with `MetricsRecorder.summary()` — count, ok/error/timeout split,
median and max latency, total tokens. Visible in the app's **Metrics** tab and
printed at the end of `agent run`.

## Debugging a bad run

1. `agent doctor` — is the tool you expected actually available?
2. `logs/metrics.jsonl` — which tasks failed, and were they slow or wrong?
3. The LangSmith trace for that `task_id` — which decision went wrong?
4. `agent run --task-id <id>` — reproduce that one task in isolation.
