# 0001 — Supervisor topology over a single ReAct agent

**Status:** Accepted · 2026-08-06

## Context

The task is GAIA: questions needing web research, file handling and computation.
Two obvious designs: one ReAct agent holding every tool, or a supervisor
delegating to specialists.

A single agent is simpler and cheaper — one prompt, one loop, no routing calls.
But every tool description sits in every request, prompts become a compromise
across all task types, and tool-selection accuracy falls as the toolset grows.
That last point is the binding constraint: we intend to keep adding tools.

## Decision

A supervisor routes to specialists. Each specialist has a focused prompt and a
capability-scoped toolset. Specialists are declared as data (`SpecialistSpec`)
over one shared ReAct loop, so the topology costs one file, not one file per
agent.

## Consequences

**Good.** Tool count per request stays flat as the system grows. Prompts stay
specific. A failing specialist is contained — the supervisor sees an error
message and can route elsewhere. Adding a specialist touches two files.

**Bad.** One extra model call per delegation round, so a trivial question costs
more than it would with a single agent. Routing itself can be wrong, and a weak
model routes badly — which is why the router schema is generated from the live
specialist list and why routing failure falls through to FINISH rather than
raising.

**Rejected:** a fully autonomous agent-to-agent mesh. Non-deterministic control
flow is much harder to bound and to debug, and bounding is what made this system
work at all.
