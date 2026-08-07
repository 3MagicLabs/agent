# 0002 — Bounded budgets over unbounded retries

**Status:** Accepted · 2026-08-06

## Context

The first deployed version had no step counter, no recursion limit, and
`max_retries=5` on the model client. On a free-tier provider with a
tokens-per-minute cap, a 15,000-character scrape replayed through full history
produced sustained 429s. The client sat in exponential backoff; the supervisor
looped between specialists; a single question could run for many minutes. The
host connection dropped long before the run finished, and every answer was lost.

The failure was not any one of those choices. It was that *nothing* in the
system had an upper bound.

## Decision

Every loop gets an explicit cap, and retries are cut to 2.

| Bound | Default |
|---|---|
| `max_supervisor_steps` | 4 |
| specialist `max_iterations` | 3 |
| `recursion_limit` | derived, `steps * 3 + 6` |
| `per_question_timeout_s` | 180 |
| `total_budget_s` | 2400 |
| `LLM_MAX_RETRIES` | 2 |

History is trimmed to a window before every model call, and scrapes are capped
at 6,000 characters — bounding token spend is what stops the rate limiting that
started the cascade.

## Consequences

**Good.** Termination is guaranteed by several independent mechanisms. A run has
a predictable worst-case duration you can plan a host timeout around. Hitting a
budget is logged loudly rather than being silently absorbed as latency.

**Bad.** A genuinely hard question that needed six delegation rounds now gets
four and answers from partial information. This is the right trade for a
benchmark scored per-question: one unanswerable question must not consume the
run. Budgets are all environment variables, so raising them for a specific run
costs nothing.
