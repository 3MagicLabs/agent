## What and why

<!-- What does this change, and what problem does it solve? Link the issue. -->

Closes #

## How

<!-- The approach, and any alternative you rejected. -->

## Verification

- [ ] `pytest -m unit` passes
- [ ] `ruff check . && ruff format --check .` passes
- [ ] `mypy` passes
- [ ] New behaviour has a test that fails without the change
- [ ] `python -c "import app"` still works with no credentials set

<!-- If this touches agent behaviour, paste before/after from `agent run --limit 3`. -->

## Risk

- [ ] No new required credential (or: documented in README and `.env.example`)
- [ ] No new unbounded loop or unbounded retry
- [ ] No secret, token, or personal data in the diff
