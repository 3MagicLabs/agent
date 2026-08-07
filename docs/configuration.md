# Configuration

All configuration is read from the environment exactly once into a frozen
`Settings` object (`src/agent/config.py`). No other module calls `os.getenv`.

Run `agent doctor` to see what resolved.

## Credentials

| Variable | Required | Effect if missing |
|---|---|---|
| `GROQ_API_KEY` | one of these three | `MissingCredentialsError` at first model call |
| `OPENAI_API_KEY` | " | " |
| `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` | " | " |
| `TAVILY_API_KEY` | no | `web_search` returns an "unavailable" message |
| `E2B_API_KEY` | no | `python_repl` returns an "unavailable" message |
| `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY` | no | no traces; logs and metrics unaffected |

Provider selection is first-match in the order above.

## Models

| Variable | Default |
|---|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `HUGGINGFACE_MODEL` | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `LLM_BASE_URL` | provider default |
| `LLM_TEMPERATURE` | `0.0` |

!!! warning "Small models and tool calling"
    `llama-3.1-8b-instant` has a much larger daily token quota but is
    unreliable at structured output and tool calls, which shows up as routing
    failures. Prefer the 70B model unless you are quota-bound.

## Budgets

| Variable | Default | Bounds |
|---|---|---|
| `MAX_SUPERVISOR_STEPS` | `4` | delegation rounds per task |
| `MAX_WEB_ITERATIONS` | `3` | web specialist tool loops |
| `MAX_CODE_ITERATIONS` | `3` | code specialist tool loops |
| `HISTORY_WINDOW` | `8` | messages replayed per model call |
| `PER_QUESTION_TIMEOUT_S` | `180` | hard cap per task |
| `TOTAL_BUDGET_S` | `2400` | hard cap for a whole run |
| `LLM_TIMEOUT_S` | `60` | single request |
| `LLM_MAX_RETRIES` | `2` | fail fast rather than sit in backoff |

## Tools

| Variable | Default | Purpose |
|---|---|---|
| `MAX_SCRAPE_CHARS` | `6000` | main driver of token spend |
| `MAX_FILE_CHARS` | `12000` | attachment read size |
| `MAX_CODE_OUTPUT_CHARS` | `4000` | sandbox output cap |
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
