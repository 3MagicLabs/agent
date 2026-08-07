# 0003 — Lazy initialisation and graceful degradation

**Status:** Accepted · 2026-08-06

## Context

Two separate production outages had the same shape.

A search client was constructed at module scope:

```python
tavily_search_tool = TavilySearchResults(max_results=3)  # at import time
```

Without `TAVILY_API_KEY` this raises a pydantic `ValidationError`. The import
chain `app → agent → web_agent → web_tools` turned a missing optional
credential into a dead application that never served a request.

The second was an SDK import whose `except ImportError` fallback re-imported
from the same missing module.

In both cases the failure was maximally distant from its cause: an optional
tool's missing key presented as a total startup crash with a stack trace in a
module the operator had never heard of.

## Decision

1. **Nothing constructed at import time.** No clients, no credential reads, no
   network. Tools resolve their dependencies inside the call.
2. **Missing credentials return a message, not an exception**, and the message
   says what to do instead: *"web_search is unavailable: TAVILY_API_KEY is not
   configured. Use scrape_webpage on a known URL, or answer from context."*
3. **Unavailable tools are still bound to the agent** by default. The model
   learns far faster from an explicit refusal than from a tool's absence.
4. **`tests/unit/test_imports.py` imports every module with no credentials set**
   and is a required CI check. A regression fails the PR, not production.

## Consequences

**Good.** The application always starts. Capability is a runtime property you
can inspect (`agent doctor`) rather than a deploy-time gamble. Partial
credentials give partial capability instead of nothing.

**Bad.** A misconfiguration is quieter — you get a degraded agent rather than a
loud crash. `agent doctor`, startup logging of the capability report, and the
app's Configuration tab exist to compensate. A genuinely required credential
(the model provider) still raises, but at call time and with a message naming
every accepted variable.
