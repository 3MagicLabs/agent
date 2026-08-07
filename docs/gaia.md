# GAIA benchmark

[GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) grades by
**exact match** after normalization. A correct answer formatted wrongly scores
zero, which is why the finalizer node and `agent.eval.scorers` exist.

## Running

```bash
agent run --limit 3                  # smoke test
agent run --task-id <uuid>           # one task, full logs
agent run                            # everything, cached as it goes
agent submit --username <hf-user>    # submit the cache
```

Answers land in `logs/answers.json`. `agent run` skips anything already cached,
so an interrupted run resumes.

## Scoring locally

Build a `gold.json` of `task_id -> expected answer` as you confirm answers:

```json
{ "8e867cd7-cff9-4e6c-867a-ff5ddc2550be": "3" }
```

```bash
agent score --gold gold.json
```

This is the regression suite for agent behaviour — run it after any prompt or
routing change to see what you broke.

## Answer format

The finalizer enforces:

- **Numbers**: digits only, no separators, no currency symbols, no units unless asked.
- **Strings**: as few words as possible, no leading article.
- **Lists**: the above per element, joined by `", "`.

## Level 1 notes

Level 1 is single-hop, but many tasks carry an attachment. `build_prompt`
detects `file_name` and instructs the model to call `download_task_file` with
the task ID, then `read_file` on the result. Spreadsheets come back as a table
summary; source code comes back verbatim.

Still missing for full Level 1 coverage: audio transcription, image
understanding, and YouTube transcripts. See the [roadmap](roadmap.md).
