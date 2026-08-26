"""Specialist agents.

Register a new specialist by adding its ``spec`` builder to ``SPEC_BUILDERS``;
the supervisor graph and its router pick it up automatically.
"""

from __future__ import annotations

from collections.abc import Callable

from agent.agents import code, reason, web
from agent.agents.base import SpecialistSpec, build_specialist, last_text, tool_evidence
from agent.config import Settings, get_settings

SpecBuilder = Callable[[Settings | None], SpecialistSpec]

#: Order matters only for how the roster reads to the router. Cheapest first.
SPEC_BUILDERS: tuple[SpecBuilder, ...] = (reason.spec, web.spec, code.spec)


def all_specs(settings: Settings | None = None) -> tuple[SpecialistSpec, ...]:
    """Build every specialist spec against the given settings."""
    resolved = settings or get_settings()
    return tuple(builder(resolved) for builder in SPEC_BUILDERS)


__all__ = [
    "SPEC_BUILDERS",
    "SpecialistSpec",
    "all_specs",
    "build_specialist",
    "last_text",
    "tool_evidence",
]
