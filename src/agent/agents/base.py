"""Specialist factory.

Every specialist is the same ReAct loop — reason, optionally call tools, repeat
until it stops calling tools or burns its iteration budget. Only the prompt,
the toolset and the budget differ, so the loop is written once here and each
specialist is declared as data in ``SpecialistSpec``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.config import get_settings
from agent.core.conversation import normalize, text_of
from agent.core.llm import get_llm, with_effort
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


def tool_evidence(messages: Sequence[BaseMessage], *, has_tools: bool = True) -> str:
    """Which tools actually ran, as one line the supervisor can read.

    The supervisor sees only a specialist's final text, so a researched answer
    and an invented one are indistinguishable to it. Faced with that it does the
    only sensible thing - delegates again to verify - which is how one solved
    task became four rounds and 34,185 tokens, its router noting each time that
    the claim "wasn't confirmed with a search" while eight searches sat in the
    log.

    ``has_tools`` distinguishes the two ways of running no tools. A specialist
    that could have searched and did not has produced a claim; one that has no
    tools at all has done exactly its job. Reporting both as "unverified" made
    the supervisor re-delegate after every single reason_agent turn, since that
    specialist is tool-less by design and can never satisfy the check.

    ``ToolMessage`` is the evidence rather than ``AIMessage.tool_calls``: a call
    can be requested and still never run.
    """
    counts = Counter(
        str(message.name or "unknown") for message in messages if isinstance(message, ToolMessage)
    )
    if not counts:
        if not has_tools:
            return "reasoned directly - this specialist has no tools by design"
        return "no tools were used - this answer is unverified"
    return ", ".join(
        f"{name} x{count}" if count > 1 else name for name, count in sorted(counts.items())
    )


def last_text(messages: Sequence[BaseMessage], default: str = "(no output produced)") -> str:
    """Most recent message that actually carries text.

    A force-quit on the iteration cap can leave a bare tool-call message with
    empty content, so we walk backwards rather than taking ``messages[-1]``.
    """
    for message in reversed(messages):
        text = text_of(message).strip()
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
            # A human turn, not a system one: this is an observation about what
            # just happened, and a second system message part-way down the list
            # is what Anthropic rejects as "multiple non-consecutive system
            # messages". normalize() would hoist it to the front regardless,
            # losing the "this just failed" positioning that makes it useful.
            messages.append(HumanMessage(content=f"Previous error to fix: {state['last_error']}"))

        error = ""
        try:
            base = llm_factory()
            paced = with_effort(base, get_settings().specialist_effort)
            model = paced.bind_tools(tool_list) if tool_list else paced
            response: BaseMessage = model.invoke(normalize(messages))
        except Exception as exc:  # noqa: BLE001 - a provider failure must not kill the run
            log.error("%s reasoning failed: %s", spec.name, exc)
            # Recorded, not swallowed: `route` sends it back here with the error
            # quoted, which is usually enough to fix a malformed tool call.
            error = str(exc)
            response = AIMessage(content=f"{spec.name} failed: {exc}")

        return {"messages": [response], "iterations": 1, "last_error": error}

    def route(state: SpecialistState) -> str:
        """Continue to tools, retry a failed call, or stop on the budget."""
        if state.get("iterations", 0) >= spec.max_iterations:
            log.warning("%s hit its iteration cap (%d) - stopping.", spec.name, spec.max_iterations)
            return END
        if state.get("last_error"):
            log.info("%s retrying after: %s", spec.name, state["last_error"][:120])
            return "reason"
        if getattr(state["messages"][-1], "tool_calls", None):
            return "tools"
        return END

    builder: StateGraph[SpecialistState] = StateGraph(SpecialistState)
    builder.add_node("reason", reason)
    builder.add_edge(START, "reason")

    if tool_list:
        builder.add_node("tools", ToolNode(tool_list))
        builder.add_conditional_edges(
            "reason", route, {"tools": "tools", "reason": "reason", END: END}
        )
        builder.add_edge("tools", "reason")
    else:
        builder.add_edge("reason", END)

    return builder.compile()
