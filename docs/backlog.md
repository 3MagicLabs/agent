# Product backlog — DRAFT for review

Every item below is grounded in something observed on 2026-08-13/14, not imagined.
Format follows CS 130: user story + Given/When/Then acceptance criteria; quality
attributes as measurable scenarios rather than adjectives.

**Nothing here has been created on GitHub yet.** Delete, merge or rewrite freely;
I will create only what survives review.

---

## 0. Cost and waste audit (do this first)

GitHub Actions minutes are **free and unlimited for public repositories**, and this
repo is public — so CI/CD costs nothing in money. The real costs are LLM tokens,
review attention, and maintenance of things nobody uses. Cutting comes before adding.

| Item | Observed | Proposal |
|---|---|---|
| **LLM tokens** | ~8,900/task; Groq free tier is 100,000/day | The single real cost. See **CAP-1**. |
| `Release` workflow | Never triggered; one tag `v0.1.0` | Remove unless you intend to publish releases |
| Dependabot | 4 PRs raised, 2 open and unreviewed since Aug 7 | Keep security updates, drop version-bump noise, or set a monthly interval |
| `requirements.txt` | Duplicates `pyproject.toml`; must be hand-synced | Generate it, or accept the duplication and pin the CI parity job that guards it |
| `eval/scorers.py` | 53 lines, 4 known bugs, unreachable in the submission path | Delete, or give it a gold file — see **ENG-6** |
| `gitleaks` hook | Panics with a wasm error; skipped on every commit today | Fix, replace, or remove — a scanner that never runs is worse than none |
| `TaskMetric.supervisor_steps` | Always `0`; never populated | Wire it up or drop the field |

**Rule adopted:** no new workflow, package extra, or dependency without an issue
naming the quality attribute it serves.

---

## 1. Quality attributes

Measurable scenarios. Each is a fitness function the backlog is judged against;
vague adjectives ("fast", "reliable") are deliberately absent.

| ID | Attribute | Scenario | Response measure |
|---|---|---|---|
| **QA-1** | Cost efficiency | A full 20-task benchmark run on a free-tier key | Median ≤ **5,000 tokens/task**; total ≤ **100,000** (one Groq day) |
| **QA-2** | Failure visibility | Any provider, tool or parsing failure during a run | **100%** surface as `status != "ok"` with a non-empty `error`; **0** fabricated answers reach the cache |
| **QA-3** | Completion | A 20-task run under normal quota | Completes within `total_budget_s`; **0** tasks fail with HTTP 429 |
| **QA-4** | Answer conformance | Any answer written to the cache | **0** contain a specialist tag, preamble, or exceed `MAX_ANSWER_CHARS` |
| **QA-5** | Testability | The unit suite on a clean machine | Runs with **no credentials and no network**; ≥ **85%** line coverage; < 30s |
| **QA-6** | Security | Untrusted model output reaching a tool | No path escape outside `download_dir`; no code execution outside the sandbox; no secret in git history |
| **QA-7** | Maintainability | Any source file | ≤ **400 lines**; `mypy --strict` and `ruff` clean; every public function documented |
| **QA-8** | Benchmark accuracy | GAIA Level 1 submission | ≥ **6/20** (certificate); stretch **15/20** (all non-multimodal tasks) |

**Known trade-off:** QA-1 (fewer tokens) opposes QA-8 (accuracy) — smaller scrapes and
fewer iterations mean less evidence per answer. Resolve empirically, not by argument:
measure accuracy at each budget setting.

---

## 2. Milestones

| Milestone | Goal | Contains |
|---|---|---|
| **M1 — Certificate (6/20)** | Pass the course threshold | CAP-1, CAP-2, CAP-3, ENG-1 |
| **M2 — Level 1 complete (20/20)** | All five modalities working | CAP-4 … CAP-8 |
| **M3 — Engineering baseline** | Practices that make M2 safe | ENG-2 … ENG-8 |
| **M4 — Toward Level 3** | Deferred; opened after M2 | — |

---

## 3. Capability backlog

### CAP-1 · Fit a full run inside one day's token quota
`area:eval` `gaia:level-1` `enhancement` · **M1** · serves QA-1, QA-3

> As an operator on a free-tier key, I want a full 20-task run to fit inside one day's
> token allowance, so that I can evaluate the agent without paying or waiting a day.

- **Given** the default budgets, **when** a 20-task run completes, **then** median
  per-task usage is ≤ 5,000 tokens and the run total is ≤ 100,000.
- **Given** a run that would exceed the daily cap, **when** the cap is reached,
  **then** the run stops with a message naming TPD, and cached answers remain submittable.
- **Given** reduced budgets, **when** accuracy is compared against the previous run,
  **then** the change in correct answers is recorded in the issue.

*Evidence:* measured 8,900 tokens/task × 20 = 178,000 against a 100,000 TPD limit.
*Levers:* `MAX_SCRAPE_CHARS`, `MAX_SUPERVISOR_STEPS`, `HISTORY_WINDOW`, `MAX_WEB_ITERATIONS`;
specialist output summarisation before it re-enters the supervisor transcript.

### CAP-2 · Stop re-delegating to the same specialist
`area:orchestration` `enhancement` · **M1** · serves QA-1

> As an operator, I want the supervisor not to send the same task to one specialist
> repeatedly, so that budget is not spent replaying work already done.

- **Given** a specialist has already returned output, **when** the supervisor routes again,
  **then** it does not select that specialist unless its previous attempt errored.
- **Given** the router still requests a spent specialist, **when** that happens,
  **then** the run continues without a provider-side 400.

*Evidence:* one run routed to `code_agent` on steps 2, 3 and 4. A first attempt at this —
narrowing the router schema — caused Groq 400s, because `with_structured_output`
validates rather than constrains. Reverted; see the comment in `graph.py`.

### CAP-3 · Spread a run across providers or days
`area:eval` `enhancement` · **M1** · serves QA-3

> As an operator, I want to resume a partially-completed run against a different provider
> or on a later day, so that a daily cap delays me rather than blocking me.

- **Given** a run stopped by a daily cap, **when** I re-run with a different `LLM_PROVIDER`,
  **then** only unanswered tasks are attempted and prior answers are preserved.
- **Given** answers gathered across several sessions, **when** I submit,
  **then** all of them are sent as one set.

*Note:* `AnswerCache` already provides this; the issue is to verify, document and test it,
not to build it. Depends on **ENG-3** (the cache is not crash-safe).

### CAP-4 · Audio transcription tool
`area:tools` `gaia:level-1` `enhancement` · **M2** · serves QA-8

> As the agent, I want to transcribe an attached audio file, so that I can answer tasks
> whose content is only available as speech.

- **Given** a task with an `.mp3` attachment, **when** the agent calls the tool,
  **then** it receives a text transcript.
- **Given** no transcription credentials, **when** the tool is called,
  **then** it returns an explanatory message and the run continues.
- **Given** the two audio tasks, **when** run, **then** both produce a non-empty answer.

*Evidence:* tasks `99c9cc74`, `1f975693`. `files.py` already points at a
`transcribe_audio` tool that does not exist.

### CAP-5 · YouTube transcript tool
`area:tools` `gaia:level-1` `enhancement` · **M2** · serves QA-8

> As the agent, I want the transcript and metadata of a YouTube video, so that I can answer
> questions about its content without watching it.

- **Given** a task containing a YouTube URL, **when** the agent calls the tool,
  **then** it receives the transcript or a clear reason none is available.
- **Given** the two video tasks, **when** run, **then** neither answers by guessing.

*Evidence:* tasks `a1e91b78`, `9d191bce`. `a1e91b78` currently answers `2` having never
seen the video.

### CAP-6 · Image understanding
`area:tools` `area:agents` `gaia:level-1` `enhancement` · **M2** · serves QA-8

> As the agent, I want to answer questions about an attached image, so that visual tasks
> are not automatic failures.

- **Given** a task with a `.png` attachment, **when** the agent processes it,
  **then** the answer is derived from image content rather than declining.
- **Given** no vision-capable model configured, **when** an image task runs,
  **then** it fails with a cause naming the missing capability.

*Evidence:* task `cca530fc` answers "No image provided, unable to determine the next move."
*Design note:* likely a second model rather than a tool — record an ADR.

### CAP-7 · Web research depth
`area:agents` `gaia:level-1` `enhancement` · **M2** · serves QA-8

> As the agent, I want enough research iterations to follow a multi-source question,
> so that cross-referencing tasks are answerable.

- **Given** a task requiring two or more sources, **when** the web specialist runs,
  **then** it does not stop solely because of its iteration cap.
- **Given** the ten web tasks, **when** run, **then** at least four produce a
  correct answer.

*Evidence:* `web_agent hit its iteration cap (3) - stopping` on the first task of the
first run. **Conflicts with CAP-1** — resolve by measurement.

### CAP-8 · Answer-format conformance
`area:eval` `gaia:level-1` `bug` · **M2** · serves QA-4

> As an operator, I want submitted answers to match the grader's exact-match format,
> so that correct answers are not scored wrong.

- **Given** a finalizer answer with conversational wrapping, **when** it is recorded,
  **then** the wrapping is removed and the value is unchanged.
- **Given** a numeric answer, **when** cleaned, **then** decimals and minus signs survive.

*Evidence:* a run answered `Therefore, the answer is 5.` — partially addressed by
`clean_answer`; this issue covers measuring it against real submissions.

---

## 4. Engineering backlog

### ENG-1 · Document the quality attributes
`documentation` `area:devex` · **M1** · serves QA-7

- **Given** a new contributor, **when** they read `docs/`, **then** they find §1 of this
  file as a maintained page with each attribute's current measured value.

### ENG-2 · Coverage floor enforced in CI
`ci` `test` · **M3** · serves QA-5

- **Given** a PR dropping coverage below 85%, **when** CI runs, **then** it fails.
- **Given** the unit suite, **when** run without credentials or network, **then** it passes.

### ENG-3 · Make the answer cache crash-safe
`area:eval` `bug` · **M3** · serves QA-2

> As an operator, I want a crash mid-write not to destroy answers I already paid for.

- **Given** a write interrupted partway, **when** the cache is next read, **then** the
  previous contents are intact.

*Evidence:* `AnswerCache.save` calls `write_text` on the live path — truncate-then-write,
with a window where both copies are gone. The run/submit split exists precisely to survive
crashes, and its storage does not.

### ENG-4 · Security review of the tool boundary
`area:tools` `enhancement` · **M3** · serves QA-6

> As a maintainer, I want model-controlled tool inputs treated as untrusted, so that a
> hallucinated path or URL cannot read or reach something it shouldn't.

- **Given** a path outside `download_dir`, **when** `read_file` is called, **then** it refuses.
- **Given** a `file://` or internal-network URL, **when** `scrape_webpage` is called,
  **then** it refuses (SSRF).
- **Given** the repo history, **when** scanned, **then** no secret is present.

*Note:* `_resolve` already guards traversal; this issue is to test it deliberately and
review the scrape and sandbox paths to the same standard.

### ENG-5 · Fix or remove the gitleaks hook
`ci` `area:devex` `bug` · **M3** · serves QA-6

- **Given** a commit, **when** pre-commit runs, **then** secret scanning either completes
  or is absent by decision — never skipped by habit.

*Evidence:* wasm panic in `go-re2`; skipped on every commit on 2026-08-14.

### ENG-6 · Decide the fate of `eval/scorers.py`
`area:eval` `refactor` · **M3** · serves QA-7

- **Given** the module, **when** this issue closes, **then** it is either deleted with its
  CLI flags, or has a gold file, correct denominator, and tests.

*Evidence:* unreachable from the submission path; `score()` reports 100% when 5 of 20 tasks
are answered correctly; `normalize` destroys decimal points and minus signs.

### ENG-7 · Backfill ADRs for decisions already made
`documentation` · **M3** · serves QA-7

- **Given** each load-bearing decision below, **when** this issue closes, **then** an ADR
  records context, alternatives considered, and the trade-off:
  1. Fail loudly rather than fall back to prior messages
  2. Validate answers at the harness boundary rather than in the graph
  3. Pace by measured tokens rather than a fixed delay
  4. Explicit `LLM_PROVIDER` over first-key-wins
  5. Tool-less `reason_agent`, and routing character work to `code_agent`
  6. Rejected: narrowing the router schema to spent specialists

### ENG-8 · Use the PR workflow for real
`area:devex` `documentation` · **M3** · serves QA-7

- **Given** any change, **when** it lands on `main`, **then** it arrived via a PR that
  passed CI and was reviewed.
- **Given** the two open Dependabot PRs, **when** this issue closes, **then** both are
  merged or closed with a reason.

*Evidence:* four PRs exist, all from Dependabot; today's three commits sit on an unpushed
branch.

---

## 5. Definition of Done

An issue is done when:

1. Acceptance criteria pass, demonstrated by a test or a recorded measurement
2. `make check` is green (ruff, `mypy --strict`, unit suite, format)
3. Coverage did not decrease
4. The quality attribute it serves was re-measured and the value recorded
5. It landed through a reviewed PR
6. An ADR exists if a load-bearing decision was made

---

## 6. Sequencing

```
M1 (certificate)      CAP-1 → CAP-3 → CAP-2 → ENG-1
M3 (baseline)         ENG-3, ENG-5, ENG-8 in parallel with M1
M2 (level 1 complete) CAP-4, CAP-5, CAP-6 in parallel; CAP-7 and CAP-8 after CAP-1
```

CAP-1 is first because every other measurement is unreliable until a run can complete:
under throttling the reversed-text task degraded from a correct answer to garbage, so
token pressure corrupts accuracy data as well as blocking it.
