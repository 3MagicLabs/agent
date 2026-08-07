# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-08-06

First structured release. The project moved from a flat script to an installable
package with CI, docs and observability.

### Fixed

- **Startup crash**: a search client was constructed at import time and raised a
  `ValidationError` when `TAVILY_API_KEY` was unset, killing the application
  before it served a request.
- **Startup crash**: the sandbox SDK's `except ImportError` fallback re-imported
  from the same missing module.
- **Sandbox never worked**: the code specialist called the pre-1.0 E2B API
  (`sandbox.notebook.exec_cell`) against a 2.x SDK, so every execution silently
  returned a connection error.
- **Non-terminating supervisor**: no step counter and no recursion limit, so the
  supervisor could delegate until LangGraph aborted.
- **Rate-limit stalls**: 15,000-character scrapes replayed through unbounded
  history exhausted free-tier token quotas; `max_retries=5` then sat in backoff.
- **Runs lost on disconnect**: answering and submitting were a single
  synchronous action with no caching, timeout, or progress output.
- **Unscored answers**: the raw `[Web Agent Output]: ...` string was submitted to
  an exact-match benchmark.

### Added

- `src/agent` package: `config`, `core`, `agents`, `tools`, `obs`, `eval`.
- Capability-based tool registry; specialists request capabilities, not imports.
- Attachment tools: `download_task_file`, `read_file`, `list_downloaded_files`.
- `wikipedia_lookup` tool.
- Finalizer node producing exact-match answers.
- `agent` CLI: `doctor`, `run`, `score`, `submit`.
- Observability: structured logging, LangSmith tracing per task, JSONL metrics.
- 122 tests at 89% coverage, all offline.
- CI (lint, strict mypy, 3.11–3.13 matrix, build, Space-parity import check),
  CodeQL, dependency review, pip-audit, OSSF Scorecard, workflow static analysis.
- Documentation site, three ADRs, roadmap to GAIA Level 3.

### Changed

- Default Groq model is now `llama-3.3-70b-versatile`; `llama-3.1-8b-instant`
  has a larger quota but is unreliable at structured output and tool calls.
- All configuration consolidated into a single frozen `Settings` object.
- `LLM_MAX_RETRIES` reduced from 5 to 2 to fail fast instead of hanging.
- Scrape cap reduced from 15,000 to 6,000 characters.

### Removed

- Flat modules `agent.py`, `state.py`, `observability.py`, `evaluate.py`.
- Empty placeholder modules for unimplemented agents and tools.
- `langchain` and `langchain-community` dependencies, no longer used.

[Unreleased]: https://github.com/3MagicLabs/agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/3MagicLabs/agent/releases/tag/v0.1.0
