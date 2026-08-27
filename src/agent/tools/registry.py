"""Capability-based tool registry.

Tools declare a capability and their credential requirements. Specialists ask
for capabilities rather than importing concrete tools, so adding a tool is a
one-line registration and never a change to agent code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

from agent.config import Settings, get_settings
from agent.obs.logging import get_logger
from agent.tools.cache import memoized

log = get_logger("tools.registry")

Capability = str  # "search" | "scrape" | "files" | "code" | "media"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A registered tool and the conditions under which it is usable."""

    name: str
    capability: Capability
    factory: Callable[[], BaseTool]
    #: Settings properties that must be truthy for this tool to be useful.
    requires: tuple[str, ...] = ()
    #: Whether repeat calls with identical arguments may be served from
    #: cache within a task. Opt-in, never opt-out: a new tool is safe until
    #: someone has thought about it. python_repl must stay False - code can
    #: be nondeterministic and rerunning it can be intentional.
    cacheable: bool = False

    def is_available(self, settings: Settings) -> bool:
        return all(bool(getattr(settings, attr, False)) for attr in self.requires)


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    """Register a tool spec. Later registrations of the same name win."""
    _REGISTRY[spec.name] = spec
    return spec


def registered() -> tuple[ToolSpec, ...]:
    return tuple(_REGISTRY.values())


def get_tools(
    *capabilities: Capability,
    settings: Settings | None = None,
    include_unavailable: bool = True,
) -> tuple[BaseTool, ...]:
    """Return tools for the requested capabilities.

    Unavailable tools are included by default: each one degrades to an
    explanatory message at call time, which teaches the model to stop retrying
    far better than the tool silently not existing.
    """
    resolved = settings or get_settings()
    wanted = set(capabilities)
    selected: list[BaseTool] = []

    for spec in _REGISTRY.values():
        if spec.capability not in wanted:
            continue
        if not spec.is_available(resolved):
            if not include_unavailable:
                continue
            log.warning("Tool %r registered but its credentials are missing.", spec.name)
        built = spec.factory()
        selected.append(memoized(built) if spec.cacheable else built)

    return tuple(selected)


def capability_report(settings: Settings | None = None) -> dict[str, bool]:
    """Name -> availability, for startup logging and the UI status panel."""
    resolved = settings or get_settings()
    return {spec.name: spec.is_available(resolved) for spec in _REGISTRY.values()}


def load_builtin_tools() -> None:
    """Import the built-in tool modules so their registrations run."""
    from agent.tools import code, files, web  # noqa: F401  (import for side effects)


def iter_capabilities() -> Iterable[Capability]:
    return {spec.capability for spec in _REGISTRY.values()}
