"""Specialist factory.

Every specialist is the same ReAct loop — reason, optionally call tools, repeat
until it stops calling tools or burns its iteration budget. Only the prompt,
the toolset and the budget differ, so the loop is written once here and each
specialist is declared as data in ``SpecialistSpec``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.core.llm import get_llm
from agent.core.state import SpecialistState
from agent.obs.logging import get_logger

log = get_logger("agents.base")


@dataclass(frozen=True, slots=True)
class SpecialistSpec:
    """Declarative description of a specialist agent."""

    name: str
    #: One line shown to the supervisor's router to explain when to pick this agent.
    description: str
    prompt: str
    tools: tuple[BaseTool, ...]
    max_iterations: int

    def label(self) -> str:
        return self.name


def last_text(messages: Sequence[BaseMessage], default: str = "(no output produced)") -> str:
    """Most recent message that actually carries text.

    A force-quit on the iteration cap can leave a bare tool-call message with
    empty content, so we walk backwards rather than taking ``messages[-1]``.
    """
    for message in reversed(messages):
        text = str(message.content).strip()
        if text:
            return text
    return default


def build_specialist(
    spec: SpecialistSpec,
    llm_factory: Callable[[], Any] = get_llm,
) -> Any:
    """Compile a ReAct subgraph for one specialist.

    ``llm_factory`` is injected rather than imported so tests can substitute a
    stub without patching module globals.
    """
    system_message = SystemMessage(content=spec.prompt)
    tool_list = list(spec.tools)

    def reason(state: SpecialistState) -> dict[str, Any]:
        """Decide the next action; increments the iteration counter."""
        messages: list[BaseMessage] = [system_message, *state["messages"]]
        if state.get("last_error"):
            messages.append(SystemMessage(content=f"Previous error to fix: {state['last_error']}"))

        try:
            model = llm_factory().bind_tools(tool_list) if tool_list else llm_factory()
            response: BaseMessage = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - a provider failure must not kill the run
            log.error("%s reasoning failed: %s", spec.name, exc)
            response = AIMessage(content=f"{spec.name} failed: {exc}")

        return {"messages": [response], "iterations": 1, "last_error": ""}

    def route(state: SpecialistState) -> str:
        """Continue to tools, or stop — including a hard stop on the budget."""
        if state.get("iterations", 0) >= spec.max_iterations:
            log.warning("%s hit its iteration cap (%d) - stopping.", spec.name, spec.max_iterations)
            return END
        if getattr(state["messages"][-1], "tool_calls", None):
            return "tools"
        return END

    builder: StateGraph[SpecialistState] = StateGraph(SpecialistState)
    builder.add_node("reason", reason)
    builder.add_edge(START, "reason")

    if tool_list:
        builder.add_node("tools", ToolNode(tool_list))
        builder.add_conditional_edges("reason", route, {"tools": "tools", END: END})
        builder.add_edge("tools", "reason")
    else:
        builder.add_edge("reason", END)

    return builder.compile()
