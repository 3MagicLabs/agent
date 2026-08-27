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
    #: The task being answered. Scopes the attachment inventory: the download
    #: directory outlives a task, and an unscoped listing let one task read
    #: another's files.
    task_id: str
    #: The router's justification for the current delegation, passed through to
    #: the specialist so it knows its task instead of inferring one.
    instruction: str
    #: Delegation rounds consumed. Bounded by Settings.max_supervisor_steps.
    steps: Annotated[int, operator.add]


class SpecialistState(TypedDict, total=False):
    """Shared state for every ReAct-style specialist subgraph."""

    messages: Annotated[list[BaseMessage], operator.add]
    #: Reasoning turns consumed. Bounded by the specialist's iteration cap.
    iterations: Annotated[int, operator.add]
    last_error: str


def initial_supervisor_state(messages: list[BaseMessage], task_id: str = "") -> SupervisorState:
    return {"messages": messages, "next_agent": "", "task_id": task_id, "steps": 0}


def initial_specialist_state(messages: list[BaseMessage]) -> SpecialistState:
    return {"messages": messages, "iterations": 0, "last_error": ""}
