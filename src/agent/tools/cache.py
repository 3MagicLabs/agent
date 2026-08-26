"""Memoise tool results within a task.

A run issued 22 tool calls of which only 14 were distinct: the same Wikipedia
article was fetched three times and the same YouTube page scraped three times,
each costing ten seconds against a per-task timeout that killed two tasks.

Repeats happen because a specialist gets a fresh state on every delegation, so
it has no memory of what an earlier one already looked up.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from agent.obs.logging import get_logger

log = get_logger("tools.cache")

#: Phrases the tools use when reporting their own failure ("Search failed with
#: error: ...", "Failed to scrape URL ...", "No Wikipedia article found ...").
#: Only the opening of a result is checked, so a page whose body discusses a
#: failure is not mistaken for one.
#:
#: A false positive costs a refetch; a false negative caches a failure and
#: disables the tool for the rest of the task. The asymmetry is deliberate -
#: when in doubt, do not cache. That this predicate is needed at all is a smell
#: pointing at tools reporting failure in-band as ordinary text.
_FAILURE_MARKERS = (
    "failed",
    "unavailable",
    "could not",
    "cannot ",
    "no file is available",
    "no wikipedia article",
    "refusing to",
    "error:",
)
_FAILURE_WINDOW = 200


def looks_like_failure(result: str) -> bool:
    """Whether a tool's result reads as its own error message."""
    head = result[:_FAILURE_WINDOW].lower()
    return any(marker in head for marker in _FAILURE_MARKERS)


class ToolCache:
    """Tool results for the current generation.

    Tools are constructed once, when the orchestrator is built, so anything
    stored on them would otherwise live as long as the process - and in a
    long-running Space that means a page scraped on Monday served on Friday.
    The generation counter gives the cache a shorter life than its container:
    ``new_generation()`` makes every prior entry unreachable without touching
    them, and the harness bumps it once per task.
    """

    def __init__(self) -> None:
        self._generation = 0
        self._entries: dict[tuple[int, str, str], str] = {}

    @property
    def generation(self) -> int:
        return self._generation

    def new_generation(self) -> None:
        self._generation += 1

    def get(self, tool: str, key: str) -> str | None:
        return self._entries.get((self._generation, tool, key))

    def put(self, tool: str, key: str, result: str) -> None:
        self._entries[(self._generation, tool, key)] = result

    def clear(self) -> None:
        self._entries.clear()


#: Process-wide, because the tools that consult it are built once.
_CACHE = ToolCache()


def get_cache() -> ToolCache:
    return _CACHE


def _key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Canonical form of a call's arguments.

    Sorted, because ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` are the same
    call and must not occupy two entries.
    """
    return json.dumps({"a": args, "k": kwargs}, sort_keys=True, default=str)


def memoized(tool: BaseTool, cache: ToolCache | None = None) -> BaseTool:
    """Return a copy of ``tool`` that serves repeat calls from ``cache``.

    A new tool rather than a mutated one: the original stays usable, and this
    module never reaches into an object it did not create.

    A hit returns the full cached text with a marker rather than a pointer.
    Across delegations the earlier result may have been trimmed out of the
    transcript, so a pointer could refer to something the model can no longer
    see; the marker still tells it that it is repeating itself.
    """
    store = cache if cache is not None else _CACHE
    inner = getattr(tool, "func", None)
    if inner is None:  # pragma: no cover - every registered tool is a StructuredTool
        log.warning("%s has no .func and cannot be memoised", tool.name)
        return tool

    def wrapper(*args: Any, **kwargs: Any) -> str:
        key = _key(args, kwargs)
        hit = store.get(tool.name, key)
        if hit is not None:
            log.info("cache hit: %s", tool.name)
            return f"[already retrieved earlier in this task]\n{hit}"

        result = str(inner(*args, **kwargs))
        if looks_like_failure(result):
            log.info("not caching a failed %s call", tool.name)
            return result
        store.put(tool.name, key, result)
        return result

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=wrapper,
    )
