"""Reasoning specialist: no tools, just thought."""

from __future__ import annotations

from agent.agents.base import SpecialistSpec
from agent.config import Settings, get_settings
from agent.core.prompts import REASON_SPECIALIST


def spec(_settings: Settings | None = None) -> SpecialistSpec:
    _ = _settings or get_settings()
    return SpecialistSpec(
        name="reason_agent",
        description=(
            "solve a question that is entirely contained in its own text - logic and word "
            "puzzles, a table printed in the prompt, classification from ordinary "
            "knowledge, or small arithmetic. No internet, no files, and no character-level "
            "manipulation such as reversing or decoding text, which belongs to code_agent"
        ),
        prompt=REASON_SPECIALIST,
        tools=(),
        # A tool-less specialist wires reason -> END, so the loop runs exactly once
        # and the iteration cap is never consulted. See agents/base.py.
        max_iterations=1,
    )
