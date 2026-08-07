# Contributing

## Setup

```bash
git clone https://github.com/3MagicLabs/agent
cd agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,tools,app]"
pre-commit install
cp .env.example .env      # then fill in the keys you have
```

Check your setup:

```bash
agent doctor
```

## The loop

```bash
pytest -m unit            # fast, offline, no credentials needed
ruff check . && ruff format .
mypy
pytest -m integration     # only if you have provider credentials
```

`pre-commit install` runs the first two automatically on commit.

## Conventions

**Commits** follow [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `perf:`, `ci:`.
The release notes are generated from them.

**Tests come first.** For a bug, write the failing test before the fix — the
regression suite in `tests/unit/test_imports.py` exists because two outages were
caused by import-time side effects, and each test there pins one of them shut.

**No module-level side effects.** Never build a client, read a credential, or
call a network at import time. `tests/unit/test_imports.py` enforces this and
will fail your PR.

**Everything is bounded.** Any new loop needs an iteration cap, and any new
network call needs a timeout. An unbounded retry is a production incident.

**Immutability.** `Settings`, `TaskMetric` and friends are frozen dataclasses.
Derive a new value with `dataclasses.replace` instead of mutating.

## Adding a tool

1. Write it in `src/agent/tools/`, decorated with `@tool`.
2. Handle a missing credential by **returning a message**, never raising —
   include what the model should do instead.
3. Register it: `register(ToolSpec(name=..., capability=..., factory=..., requires=...))`.
4. Add it to a specialist by capability in `src/agent/agents/`.
5. Test the degraded path and the happy path with a mocked transport.

## Adding a specialist

Add a `spec()` builder in `src/agent/agents/` and append it to `SPEC_BUILDERS`.
The router schema, the graph nodes and the prompt roster all derive from that
list — you should not need to touch `core/graph.py`.

## Review

PRs need a green CI run and one approval. CI runs lint, types, tests on
Python 3.11–3.13, a build, and an import check against the exact dependency set
the Hugging Face Space installs.
