# Configuration

All configuration is read from the environment exactly once into a frozen
`Settings` object (`src/agent/config.py`). No other module calls `os.getenv`.

Run `agent doctor` to see what resolved.

## Credentials

| Variable | Required | Effect if missing |
|---|---|---|
| `ANTHROPIC_API_KEY` | one of these four | `MissingCredentialsError` at first model call |
| `GROQ_API_KEY` | " | " |
| `OPENAI_API_KEY` | " | " |
| `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` | " | " (also gates GAIA attachments and gold answers) |
| `TAVILY_API_KEY` | no | `web_search` returns an "unavailable" message |
| `E2B_API_KEY` | no | `python_repl` returns an "unavailable" message |
| `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY` | no | no traces; logs and metrics unaffected |

Provider selection is first-match in the order above.

## Models

| Variable | Default |
|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-5` |
| `GROQ_MODEL` | `openai/gpt-oss-120b` |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `HUGGINGFACE_MODEL` | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `LLM_BASE_URL` | provider default |
| `LLM_TEMPERATURE` | `0.0` |

!!! warning "`LLM_TEMPERATURE` is not universal"
    Sonnet 5 rejects `temperature` with a 400, so the Anthropic client never
    sends it; depth is controlled by the effort settings below. The field
    still applies to the OpenAI-compatible providers.

## Reasoning effort

Anthropic only. `low|medium|high|xhigh|max`; an unrecognised value is
dropped with a warning rather than sent.

| Variable | Default | Why |
|---|---|---|
| `ROUTER_EFFORT` | `medium` | picks one name and writes a sentence. At `low` it
  returned an empty object and lost a task |
| `SPECIALIST_EFFORT` | `medium` | level-1 tasks are lookups and small computations |
| `FINALIZER_EFFORT` | `low` | formats an answer it has already been handed |

## Budgets

| Variable | Default | Bounds |
|---|---|---|
| `MAX_SUPERVISOR_STEPS` | `4` | delegation rounds per task |
| `MAX_WEB_ITERATIONS` | `5` | web specialist **tool calls**, not turns |
| `MAX_CODE_ITERATIONS` | `6` | code specialist **tool calls**, not turns |
| `HISTORY_WINDOW` | `8` | messages replayed per model call |
| `PER_QUESTION_TIMEOUT_S` | `300` | hard cap per task |
| `TOTAL_BUDGET_S` | `6000` | hard cap for a whole run |
| `MAX_ANSWER_TOKENS` | `128` | finalizer output |
| `MAX_ROUTER_TOKENS` | `512` | router output; generous because thinking spends it |
| `MAX_SPECIALIST_TOKENS` | `1024` | client-wide default |
| `REFUSAL_FALLBACK_MODEL` | `claude-haiku-4-5` | retried here when a classifier
  declines a request. Empty disables the retry |
| `TOKENS_PER_MINUTE` | `0` | inter-task pacing; 0 disables. Set it only for a
  provider with a known tight ceiling |
| `MAX_TASK_COST_USD` | `0.50` | stops the run if one task costs more |
| `MAX_RUN_COST_USD` | `5.00` | stops the run at this total. 0 disables |
| `LLM_TIMEOUT_S` | `60` | single request |
| `LLM_MAX_RETRIES` | `2` | fail fast rather than sit in backoff |

## Tools

| Variable | Default | Purpose |
|---|---|---|
| `MAX_SCRAPE_CHARS` | `30000` | main driver of token spend |
| `MAX_FILE_CHARS` | `60000` | attachment read size |
| `MAX_CODE_OUTPUT_CHARS` | `15000` | sandbox output cap |
| `SCRAPE_TIMEOUT_S` | `20` | HTTP timeout |
| `SANDBOX_TIMEOUT_S` | `60` | sandbox lifetime |
| `SEARCH_RESULTS` | `3` | results per search |

## Paths and benchmark

| Variable | Default |
|---|---|
| `AGENT_LOG_DIR` | `logs` |
| `AGENT_DOWNLOAD_DIR` | `logs/downloads` |
| `SCORING_API_URL` | the course scoring API |
| `LOG_LEVEL` | `INFO` |
