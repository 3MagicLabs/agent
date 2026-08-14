# Codebase tour

A reading order for `src/agent/`, bottom-up. Each stage only uses things defined
in earlier stages, so you never meet a symbol you haven't already seen.

Total: 2,173 lines across 25 files. Most files are under 100 lines.

---

## Stage 0 — the whole program in one paragraph

A GAIA task arrives as a question. The **harness** hands it to the **orchestrator**.
The orchestrator's **supervisor** decides which **specialist** should act — web
research or code execution. That specialist runs a **ReAct loop**: think, call a
**tool**, read the result, think again, until it stops or hits its iteration cap.
Its output goes back to the supervisor, which either delegates again or stops.
A **finalizer** compresses the whole transcript into one bare answer. The harness
records a **metric**, caches the answer to disk, and later submits it.

Everything else — config, logging, tracing — is plumbing that serves that path.

## The three ideas that explain most of the code

Once you see these, most files stop being surprising.

1. **Nothing is built at import time.** Model clients, tool clients, and the
   compiled graph are all constructed on first *use*, behind `@lru_cache`. An
   earlier version built a model client at import; one missing API key crashed
   the entire application before it started.

2. **Things that vary are declared as data, not written as code.** A specialist
   is a `SpecialistSpec` (name, prompt, tools, budget). A tool is a `ToolSpec`
   (name, capability, factory, requirements). The code that consumes them is
   written once.

3. **Every loop has a hard bound.** Specialists have `max_iterations`, the
   supervisor has `max_supervisor_steps`, the graph has `recursion_limit`, tasks
   have `per_question_timeout_s`, runs have `total_budget_s`. Agents loop
   forever by default; every one of these numbers exists because something
   didn't terminate.

---

## Stage 1 — the leaves

No internal imports. Nothing here calls anything else in the project.

### `core/prompts.py` (45 lines)

Four strings: `SUPERVISOR`, `WEB_SPECIALIST`, `CODE_SPECIALIST`, `FINALIZER`.

**Concept.** Prompts are behaviour. They live apart from control flow so they can
be changed and compared without touching graph code.

**Read for.** `FINALIZER` is the one that decides your benchmark score — GAIA
grades by exact match, so a correct answer in the wrong format scores zero.

**You understand it when** you can say what each prompt is trying to prevent.

### `core/state.py` (39 lines)

Two `TypedDict`s and two constructors.

**Concept.** LangGraph state is a set of named channels. A node returns a
*partial* update, not a whole state.

**Technical.** `Annotated[Sequence[BaseMessage], operator.add]` attaches a
**reducer**. When a node returns `{"messages": [msg]}`, LangGraph *appends*
rather than replaces. Same for `steps`: returning `{"steps": 1}` means "add
one", not "set to one". Fields without an annotation (`next_agent`) are
last-write-wins.

**You understand it when** you can explain why a node returning `{"steps": 0}`
is meaningful rather than pointless.

### `config.py` (195 lines)

One frozen `Settings` dataclass, a provider resolution order, and derived properties.

**Concept.** Single source of truth. No other module in the package reads the
environment — check it yourself:

```bash
grep -rn "os.getenv" src/          # only config.py
```

Two deliberate exceptions, both outside the package: `app.py` reads `SPACE_ID`
and `OAUTH_CLIENT_ID` to detect whether it is running on Hugging Face, and
`obs/tracing.py` *writes* `LANGSMITH_*` variables because the LangChain
libraries read them from the environment rather than taking arguments.

**Technical.** `@dataclass(frozen=True, slots=True)` makes settings immutable,
so nothing can mutate configuration halfway through a run. Derived values are
`@property`, so they can't drift from their inputs:

```python
@property
def recursion_limit(self) -> int:
    return self.max_supervisor_steps * 3 + 6
```

**Read for.** The budget numbers. These are the knobs that decide whether you
fit inside a free-tier rate limit.

**You understand it when** you can trace `GROQ_API_KEY` in your shell to
`settings.model` being `llama-3.3-70b-versatile`.

---

## Stage 2 — the model client

### `core/llm.py` (56 lines)

**Concept.** A factory with a process-wide cache. Also: credentials are checked
at *call* time, and the failure is a named exception rather than a crash.

**Technical.** `@lru_cache(maxsize=1)` on a zero-argument function is the
idiomatic Python singleton. `reset_llm()` exists so tests can clear it.

**Note.** All providers go through `ChatOpenAI` — Groq and others expose
OpenAI-compatible endpoints, so `base_url` is all that changes.

**You understand it when** you can explain why `get_llm()` is called inside
every node rather than stored on the class at construction.

---

## Stage 3 — tools

Read `registry.py` first for the pattern, then `files.py` because it's the most
concrete, then `web.py` and `code.py`.

### `tools/registry.py` (89 lines)

**Concept.** Agents ask for *capabilities* (`get_tools("search", "files")`), not
for named tools. Adding a tool is a registration, never a change to agent code.

**Technical.** `ToolSpec.requires` lists settings attributes that must be truthy.
`is_available()` checks them. Note the deliberate choice in `get_tools`:
unavailable tools are **included by default** and degrade to an explanatory
message at call time — that teaches the model to stop retrying better than the
tool silently not existing.

### `tools/files.py` (153 lines) → `web.py` (163) → `code.py` (102)

**Concept.** A tool is a function the model can call. Its docstring is not
documentation — it is the description sent to the model, and it is how the model
decides whether to call it.

**Technical.** `@tool` turns a function into a `BaseTool`; the signature becomes
a JSON schema. Notice that every external client (Tavily, E2B) is constructed
*inside* the function, not at module level — an import-time Tavily client once
crashed the whole app when its key was missing.

**You understand it when** you can add a tool without touching any file in
`agents/`.

---

## Stage 4 — one agent

### `agents/base.py` (102 lines) — the most important file in the project

**Concept: ReAct.** Reason → Act → Observe → repeat. The model thinks, optionally
emits tool calls, the tools run, their results come back as messages, and it
thinks again. This loop is the entire idea behind "agent".

**Technical.** A four-step LangGraph pattern that repeats everywhere:

1. `builder = StateGraph(SpecialistState)` — declare the state shape
2. `builder.add_node("reason", reason)` — nodes are functions: state in, partial state out
3. `builder.add_conditional_edges("reason", route, {...})` — a function returns the next node's name
4. `builder.compile()` — produces something with `.invoke()`

Read `route` (line 82) closely. Two ways out: the iteration cap, or no tool calls
in the last message. The cap is checked **first**, which is what guarantees
termination even if the model calls tools forever.

**The key insight.** Every specialist is this same loop. Only the prompt, the
tools and the budget differ — so the loop is written once and specialists are
declared as data.

### `agents/web.py` and `agents/code.py` (22 lines each)

A `SpecialistSpec`: name, description, prompt, tools, iteration cap. That's it.
The `description` is what the supervisor's router reads to choose between them.

**You understand this stage when** you can write a third specialist without
opening `base.py`.

---

## Stage 5 — the orchestrator

### `core/graph.py` (207 lines)

**Concept: supervisor topology.** One router, N specialists, hub-and-spoke.
Specialists never call each other and never end the run; they always return to
the supervisor. That means exactly one place to bound and one prompt to fix when
routing goes wrong.

```
START → supervisor ──┬→ web_agent  ──┐
                     ├→ code_agent ──┤ (back to supervisor)
                     └→ finalize → END
```

**Technical, in reading order:**

- `build_route_model` (39) — builds a pydantic model at runtime whose
  `next_agent` is a `Literal` of exactly the live specialist names. That schema
  is sent to the provider as a tool definition, so an invalid name is
  *impossible*, not merely discouraged. Constraint by schema, not by prompt.
- `trim` (62) — keeps `messages[0]` (the question) plus the most recent turns.
  Your only defence against context growth, and therefore against token cost.
- `_supervise` (85) — budget check first, then one structured-output call.
- `_make_specialist_node` (115) — note line 121: each invocation builds a *fresh*
  payload with `iterations: 0`. A specialist has no memory of its own previous
  run. The supervisor's transcript is the only memory in the system.
- `_finalize` (141) — one call with `FINALIZER` to compress the transcript into
  a bare answer.
- `_route` (155) — deliberately permissive: an unknown name falls through to
  `finalize` rather than raising.

**Read for.** The token arithmetic. One task can make up to ~17 LLM calls
(4 supervisor rounds + up to 3 reasoning turns per specialist invocation +
finalizer), each replaying the trimmed history. A 6,000-character scrape is
~1,500 tokens carried along every time. That is where ~12,700 tokens per task
comes from — arithmetic, not a bug.

**You understand it when** you can predict how many LLM calls a given task will
make before running it.

---

## Stage 6 — the outside

### `eval/harness.py` (243 lines)

Fetch tasks → answer each under a hard timeout → record a metric → cache to disk.
Submission is a **separate** step reading that cache, so a crash mid-run never
costs answers already produced.

**Technical.** `run()` is a generator yielding `Progress`, which is why the same
code drives both the Gradio table and the CLI. The timeout uses a thread because
`signal.alarm` only works on the main thread — but note that Python cannot kill a
thread, so a timeout means "stop waiting", not "stop working".

### `obs/` — `logging.py`, `tracing.py`, `metrics.py`

Structured logs, LangSmith wiring, and an append-only JSONL metric per task.
Read `metrics.py` first: `TaskMetric` is the record that tells you what happened.

### `cli.py` (144) and `app.py` (188)

Two front-ends over the same harness. `app.py` is a shim — it exists because
Hugging Face Spaces requires an `app.py` at the repo root, and it contains no
logic of its own.

---

## How to read any LangGraph file

Transferable, and it collapses most of the mystery:

1. **Find the state class.** What channels exist? Which have reducers?
2. **List the nodes.** Each is `state -> partial state`.
3. **Find the edges.** Fixed edges are `add_edge`; branches are
   `add_conditional_edges` with a function returning the next node's name.
4. **Find the exits.** Every path to `END`, and every bound that guarantees you
   reach one.

Apply that to `agents/base.py` and then `core/graph.py` and both become the same
shape at different scales.

---

## Trace one request

When the stages above make sense, follow a single question through the whole
system and name the file and function at each hop:

```
app.py / cli.py
  → BenchmarkRunner.run              (harness.py:165)
  → BenchmarkRunner.run_one          (harness.py:126)
  → answer_question                  (graph.py:203)
  → Orchestrator.answer              (graph.py:181)
  → graph.invoke → _supervise        (graph.py:85)
  → specialist node                  (graph.py:115)
  → subgraph reason/tools loop       (base.py:67)
  → back to _supervise
  → _finalize                        (graph.py:141)
  → TaskMetric + AnswerCache         (harness.py:154, 77)
  → BenchmarkRunner.submit           (harness.py:224)
```

If you can narrate that path out loud without looking, you understand the
codebase.
