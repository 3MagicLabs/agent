"""Graph state schemas.

``Annotated[..., operator.add]`` makes a channel accumulate across nodes;
plain fields are last-write-wins.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage


class SupervisorState(TypedDict, total=False):
    """Top-level orchestrator state."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    #: Delegation rounds consumed. Bounded by Settings.max_supervisor_steps.
    steps: Annotated[int, operator.add]


class SpecialistState(TypedDict, total=False):
    """Shared state for every ReAct-style specialist subgraph."""

    messages: Annotated[list[BaseMessage], operator.add]
    #: Reasoning turns consumed. Bounded by the specialist's iteration cap.
    iterations: Annotated[int, operator.add]
    last_error: str


def initial_supervisor_state(messages: list[BaseMessage]) -> SupervisorState:
    return {"messages": messages, "next_agent": "", "steps": 0}


def initial_specialist_state(messages: list[BaseMessage]) -> SpecialistState:
    return {"messages": messages, "iterations": 0, "last_error": ""}
