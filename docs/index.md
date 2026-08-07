# 3MagicLabs Agent

A supervisor multi-agent system for autonomous research, reasoning and code
execution, built on LangGraph.

## Start here

```bash
pip install -e ".[dev,tools,app]"
cp .env.example .env       # add at least one provider key
agent doctor               # confirm what is configured
agent run --limit 3        # answer three benchmark tasks
```

- [Architecture](architecture.md) — how the supervisor, specialists and budgets fit together
- [Configuration](configuration.md) — every environment variable
- [Observability](observability.md) — logs, traces and metrics
- [GAIA benchmark](gaia.md) — running and scoring against the course API
- [Roadmap](roadmap.md) — Level 1 now, Level 3 later

## Design commitments

1. **Nothing at import time.** No client, credential, or network call runs on
   import. A missing key degrades a tool, it never crashes the process.
2. **Everything is bounded.** Every loop has an iteration cap; every network
   call has a timeout; every run has a wall-clock budget.
3. **Work is never lost.** Answers are cached to disk as they are produced;
   submission is a separate action.
4. **Immutable by default.** Configuration and metrics are frozen dataclasses.
