"""Supervisor orchestrator.

A supervisor routes each turn to one specialist, or to FINISH. Two budgets keep
it terminating: ``max_supervisor_steps`` bounds delegation rounds, and
``recursion_limit`` is a backstop at the LangGraph level. Without them the
supervisor can ping-pong between specialists indefinitely.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, create_model

from agent.agents import SpecialistSpec, all_specs, build_specialist, last_text
from agent.config import Settings, get_settings
from agent.core.llm import get_llm
from agent.core.prompts import FINALIZER, FINALIZER_REQUEST, SUPERVISOR
from agent.core.state import SupervisorState, initial_supervisor_state
from agent.obs.logging import get_logger
from agent.obs.tracing import trace_config
from agent.tools.files import downloaded_inventory

log = get_logger("core.graph")

FINISH = "FINISH"
FINAL_ANSWER = "final_answer"


class RouteDecision(BaseModel):
    """Fallback schema used when no specialists are registered."""

    next_agent: Literal["FINISH"] = Field(description="Always FINISH.")
    reasoning: str = Field(description="Why this routing choice was made.")


def build_route_model(specs: tuple[SpecialistSpec, ...]) -> type[BaseModel]:
    """Create a router schema whose choices are exactly the live specialists."""
    if not specs:
        return RouteDecision

    choices = (*(spec.name for spec in specs), FINISH)
    return create_model(
        "RouteDecision",
        next_agent=(
            Literal[choices],
            Field(description="The specialist to call next, or FINISH."),
        ),
        reasoning=(str, Field(description="Brief justification for this choice.")),
        __doc__="Select the next specialist, or FINISH when the task is solved.",
    )


def routing_prompt(specs: tuple[SpecialistSpec, ...]) -> str:
    """Supervisor prompt with the live specialist roster appended."""
    roster = "\n".join(f"- '{spec.name}': {spec.description}." for spec in specs)
    return f"{SUPERVISOR}\n\nAvailable specialists:\n{roster}"


def trim(messages: list[BaseMessage], keep: int) -> list[BaseMessage]:
    """Keep the original question plus the most recent turns.

    Specialist output is verbose; replaying all of it every round is what
    exhausts tokens-per-minute quotas and triggers long provider backoffs.
    """
    if len(messages) <= keep:
        return list(messages)
    return [messages[0], *messages[-(keep - 1) :]]


#: Conversational lead-ins the model emits despite being told not to. Observed:
#: "Therefore, the answer is 5." on a task whose graded answer was "5".
_PREAMBLE = re.compile(
    r"^\s*(?:therefore|thus|so|hence)?[,\s]*" r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*",
    re.IGNORECASE,
)


def clean_answer(text: str) -> str:
    """Strip wrapping the grader would count as a wrong answer.

    Exact match makes "Therefore, the answer is 5." and "5" as different as
    right and wrong. The prompt already forbids preamble; this is the
    deterministic backstop for when the model does it anyway.

    Only unambiguous wrapping is removed - never punctuation inside the answer,
    so decimals and comma-separated lists survive intact.
    """
    cleaned = _PREAMBLE.sub("", text.strip()).strip()
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    if cleaned.endswith(".") and not cleaned.endswith(".."):
        cleaned = cleaned[:-1].strip()
    return cleaned or text.strip()


class Orchestrator:
    """Compiled supervisor graph bound to a settings snapshot."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.specs = all_specs(self.settings)
        self._subgraphs = {spec.name: build_specialist(spec) for spec in self.specs}
        # The roster is fixed for the whole run. Narrowing it as specialists are
        # used looks tempting, but with_structured_output does not constrain
        # generation on Groq - it validates afterwards and returns a hard 400.
        # Removing a choice the model still wants turns a retry into a failure.
        self._route_model = build_route_model(self.specs)
        self._system = SystemMessage(content=routing_prompt(self.specs))
        self.graph = self._compile()

    # --- nodes ---------------------------------------------------------
    def _supervise(self, state: SupervisorState) -> dict[str, Any]:
        step = state.get("steps", 0)
        budget = self.settings.max_supervisor_steps

        if step >= budget:
            log.warning("Supervisor step budget (%d) exhausted - finishing.", budget)
            return {"next_agent": FINISH, "steps": 1}

        messages = [self._system, *trim(list(state["messages"]), self.settings.history_window)]
        try:
            router = get_llm().with_structured_output(self._route_model, method="function_calling")
            decision = router.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - a bad tool call must not kill the run
            log.error("Routing failed (%s) - finishing with what we have.", exc)
            return {"next_agent": FINISH, "steps": 1}

        if decision is None:
            log.error("Router returned no decision - finishing.")
            return {"next_agent": FINISH, "steps": 1}

        target = str(getattr(decision, "next_agent", FINISH))
        log.info(
            "step %d/%d -> %s (%s)",
            step + 1,
            budget,
            target,
            getattr(decision, "reasoning", ""),
        )
        return {"next_agent": target, "steps": 1}

    def _make_specialist_node(self, name: str) -> Callable[[SupervisorState], dict[str, Any]]:
        """Wrap a specialist subgraph as a supervisor node."""
        subgraph = self._subgraphs[name]

        def node(state: SupervisorState) -> dict[str, Any]:
            seeded = trim(list(state["messages"]), 4)
            # A specialist gets a fresh state on every delegation, so it has no
            # memory of work it already did. Pushing the inventory is what stops
            # the second delegation re-fetching what the first one downloaded.
            inventory = downloaded_inventory()
            if inventory:
                seeded = [*seeded, SystemMessage(content=inventory)]
            payload = {"messages": seeded, "iterations": 0, "last_error": ""}
            try:
                result = subgraph.invoke(
                    payload, config={"recursion_limit": self.settings.recursion_limit}
                )
                # Only what the subgraph appended. Running last_text over the whole
                # list walks back into the seed messages, so a specialist that
                # produced nothing echoes its own input back - double-tagged.
                content = last_text(list(result["messages"])[len(seeded) :])
            except Exception as exc:  # noqa: BLE001 - one specialist failing is recoverable
                log.error("%s failed: %s", name, exc)
                content = f"{name} failed with error: {exc}"

            return {
                "messages": [AIMessage(content=f"[{name}]\n{content}", name=name)],
                "steps": 0,
            }

        return node

    def _finalize(self, state: SupervisorState) -> dict[str, Any]:
        # The trailing user turn is load-bearing. Specialist output is an
        # AIMessage, and when it happens to end with its own "Answer: ..."
        # block the model reads the conversation as already complete and
        # returns a single stop token - measured 3/3 empty without this line
        # and 3/3 correct with it, on the same captured conversation.
        messages = [
            SystemMessage(content=FINALIZER),
            *trim(list(state["messages"]), self.settings.history_window),
            HumanMessage(content=FINALIZER_REQUEST),
        ]
        # Capped: the answer is a few words, and an uncapped repetition loop
        # once emitted 4,344 tokens of a single sentence repeated.
        finalizer = get_llm().bind(max_tokens=self.settings.max_answer_tokens)
        try:
            content = clean_answer(str(finalizer.invoke(messages).content))
        except Exception as exc:
            log.error("Finalizer failed: %s", exc)
            raise

        log.info("final answer: %s", content[:200])
        return {"messages": [AIMessage(content=content, name=FINAL_ANSWER)], "steps": 0}

    def _route(self, state: SupervisorState) -> str:
        target = state.get("next_agent", FINISH)
        return "finalize" if target not in self._subgraphs else target

    # --- assembly ------------------------------------------------------
    def _compile(self) -> Any:
        builder: StateGraph[SupervisorState] = StateGraph(SupervisorState)
        builder.add_node("supervisor", self._supervise)
        builder.add_node("finalize", self._finalize)
        for name in self._subgraphs:
            # langgraph's stubs narrow node callables to Never under a
            # parameterised StateGraph; the runtime signature is correct.
            builder.add_node(name, self._make_specialist_node(name))  # type: ignore[arg-type]

        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._route,
            {**{name: name for name in self._subgraphs}, "finalize": "finalize"},
        )
        for name in self._subgraphs:
            builder.add_edge(name, "supervisor")
        builder.add_edge("finalize", END)
        return builder.compile()

    # --- public API ----------------------------------------------------
    def answer(
        self, question: str, task_id: str = "local", callbacks: list[Any] | None = None
    ) -> str:
        """Run the graph on one question and return the final answer text."""
        final_state = self.graph.invoke(
            initial_supervisor_state([HumanMessage(content=question)]),
            config=trace_config(task_id, callbacks),
        )
        for message in reversed(list(final_state["messages"])):
            if getattr(message, "name", "") == FINAL_ANSWER:
                return str(message.content).strip()
        return ""


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    """Process-wide orchestrator, compiled on first use."""
    return Orchestrator()


def reset_orchestrator() -> None:
    """Test hook: drop the cached orchestrator."""
    get_orchestrator.cache_clear()


def answer_question(
    question: str, task_id: str = "local", callbacks: list[Any] | None = None
) -> str:
    """Convenience entry point used by the app, CLI and eval harness."""
    return get_orchestrator().answer(question, task_id=task_id, callbacks=callbacks)
