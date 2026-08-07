"""Web research specialist."""

from __future__ import annotations

from agent.agents.base import SpecialistSpec
from agent.config import Settings, get_settings
from agent.core.prompts import WEB_SPECIALIST
from agent.tools import get_tools


def spec(settings: Settings | None = None) -> SpecialistSpec:
    resolved = settings or get_settings()
    return SpecialistSpec(
        name="web_agent",
        description=(
            "search the internet, look up encyclopedic facts, read a specific webpage "
            "or document URL, or download and read a file attached to the task"
        ),
        prompt=WEB_SPECIALIST,
        tools=get_tools("search", "scrape", "files", settings=resolved),
        max_iterations=resolved.max_web_iterations,
    )
