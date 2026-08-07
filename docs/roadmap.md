# Roadmap

The near-term goal is **passing GAIA Level 1** on the Hugging Face Agents
course (30% of 20 questions is the certificate threshold). The architecture is
built for Level 3, but capability is added in the order the benchmark rewards.

## Phase 0 — Foundations ✅

Supervisor topology, bounded budgets, graceful degradation, observability,
89% test coverage, CI. Done.

## Phase 1 — Pass GAIA Level 1 (current)

Level 1 tasks are single-hop but frequently carry an **attachment**, which is
where most of the score is lost.

| Work | Why it scores | Status |
|---|---|---|
| `download_task_file` + `read_file` | Several L1 tasks are unanswerable without the file | ✅ done |
| Spreadsheet/CSV reading | Sales-total style tasks | ✅ done |
| `wikipedia_lookup` | Exact encyclopedic facts, cheaper and more reliable than search | ✅ done |
| Exact-match finalizer | A right answer formatted wrongly scores zero | ✅ done |
| **Audio transcription** | Several L1 tasks attach `.mp3` | ⬜ next |
| **Vision / image reading** | Chess-position and image tasks | ⬜ next |
| **YouTube transcript tool** | Video-comprehension tasks | ⬜ next |
| **Reverse/puzzle handling** | At least one L1 task is a reversed string | ⬜ next |
| Gold answer set + `agent score` regression run | Stops a fix from silently breaking another task | ⬜ next |

**Exit criteria:** ≥ 30% on the live scoring API, reproducibly, twice.

## Phase 2 — Level 2 (multi-hop)

- **Planner node.** Decompose a question into a checklist before routing;
  the supervisor works the list instead of re-deciding from scratch each turn.
- **Working memory.** A scratchpad channel in state that survives history
  trimming, so facts found on step 1 are still present on step 6.
- **Self-verification.** A critic pass that re-derives the answer independently
  and flags disagreement.
- **Wider budgets**, justified by measured per-level latency.

## Phase 3 — Level 3 (long-horizon)

- **Parallel fan-out.** Independent sub-questions dispatched concurrently.
- **Durable execution.** Checkpoint state so a run resumes after a crash.
- **Tool learning.** Cache successful tool sequences per task archetype.
- **Multi-model routing.** Cheap model for extraction, strong model for
  synthesis, chosen per node rather than per run.

## Phase 4 — Beyond the benchmark

- Long-term memory across sessions
- MCP server integration for third-party tools
- Cost/latency budgets as a first-class scheduling constraint
- An eval suite beyond GAIA

## Non-goals

- Training or fine-tuning models
- A hosted multi-tenant service
- Supporting every LLM provider — three is enough
