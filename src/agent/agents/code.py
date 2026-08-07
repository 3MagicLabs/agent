"""Code execution specialist."""

from __future__ import annotations

from agent.agents.base import SpecialistSpec
from agent.config import Settings, get_settings
from agent.core.prompts import CODE_SPECIALIST
from agent.tools import get_tools


def spec(settings: Settings | None = None) -> SpecialistSpec:
    resolved = settings or get_settings()
    return SpecialistSpec(
        name="code_agent",
        description=(
            "write and execute Python for calculation, data processing, parsing a "
            "downloaded spreadsheet, or any algorithmic logic"
        ),
        prompt=CODE_SPECIALIST,
        tools=get_tools("code", "files", settings=resolved),
        max_iterations=resolved.max_code_iterations,
    )
