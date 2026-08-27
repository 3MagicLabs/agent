"""Supervisor orchestrator.

A supervisor routes each turn to one specialist, or to FINISH. Two budgets keep
it terminating: ``max_supervisor_steps`` bounds delegation rounds, and
``recursion_limit`` is a backstop at the LangGraph level. Without them the
supervisor can ping-pong between specialists indefinitely.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, create_model

from agent.agents import SpecialistSpec, all_specs, build_specialist, last_text, tool_evidence
from agent.config import Settings, get_settings
from agent.core.conversation import as_data, normalize, text_of
from agent.core.llm import get_llm, with_effort
from agent.core.prompts import FINALIZER, FINALIZER_REQUEST, ROUTER_REQUEST, SUPERVISOR
from agent.core.state import SupervisorState, initial_supervisor_state
from agent.obs.logging import get_logger
from agent.obs.tracing import trace_config
from agent.tools.files import downloaded_inventory

log = get_logger("core.graph")

FINISH = "FINISH"
FINAL_ANSWER = "final_answer"

#: A safety classifier declined the request. Arrives as a normal 200 with an
#: empty body, so it is only visible in stop_reason - and it is deterministic,
#: unlike a malformed reply, so retrying only buys another refusal.
REFUSAL = "refusal"

#: Where to send a task the router was not permitted to read. The refusal is
#: on the router's call; a specialist prompts differently and may not trip the
#: same classifier. code_agent because text the router could not parse is
#: usually encoded, and decoding is what it is for.
REFUSAL_FALLBACK = "code_agent"

REFUSAL_INSTRUCTION = (
    "The router was not permitted to read this task, so it could not be "
    "classified. Work directly from the task text."
)


def refusal_category(raw: Any) -> str:
    """The category when a reply was declined by policy, else "".

    A refusal is a *successful* response - HTTP 200, empty content, zero
    output tokens - with the outcome carried in stop_reason. Read content
    first and it is indistinguishable from an empty reply, which is how a
    policy decision reached this code as "next_agent Field required".
    """
    metadata = getattr(raw, "response_metadata", None) or {}
    if metadata.get("stop_reason") != REFUSAL:
        return ""
    details = metadata.get("stop_details") or {}
    return str(details.get("category") or "unspecified")


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


@dataclass(frozen=True, slots=True)
class Solution:
    """An answer together with what it cost to reach.

    ``steps`` is the delegation count. It was declared on ``TaskMetric`` from
    the start but stayed 0 in all 59 recorded runs, because the only way out of
    the graph returned a bare string and the count lived in state that was
    thrown away.
    """

    text: str
    steps: int = 0


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

        messages = normalize(
            [
                self._system,
                *as_data(trim(list(state["messages"]), self.settings.history_window)),
            ],
            ROUTER_REQUEST,
        )
        # Capped like the finalizer: the router emits one schema selection
        # and a short justification, so it never needs a specialist's room.
        capped = get_llm().bind(max_tokens=self.settings.max_router_tokens)
        # include_raw, because the default discards the reply and raises on a
        # parse failure - so a policy refusal, which carries its cause in
        # stop_reason, arrived here as a pydantic "field required" error.
        router = with_effort(capped, self.settings.router_effort).with_structured_output(
            self._route_model, method="function_calling", include_raw=True
        )
        # Retried once, because a malformed reply is usually a hiccup. A refusal
        # is not: it is deterministic, and the first version of this loop spent
        # two round trips being declined identically before giving up.
        #
        # Typed Any deliberately. with_structured_output declares a non-Optional
        # return, which would make the None checks unreachable - but that is a
        # promise about a well-behaved provider, and this codebase exists
        # because providers return things their type signatures did not predict.
        decision: Any = None
        for attempt in (1, 2):
            try:
                result: Any = router.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - a bad tool call must not kill the run
                log.warning("Routing attempt %d failed: %s", attempt, exc)
                continue

            decision = (result or {}).get("parsed")
            if decision is not None:
                break

            category = refusal_category((result or {}).get("raw"))
            if category:
                target = self._refusal_route()
                log.warning(
                    "Routing declined by policy (%s) - sending to %s unclassified.",
                    category,
                    target,
                )
                return {
                    "next_agent": target,
                    "instruction": REFUSAL_INSTRUCTION,
                    "steps": 1,
                }
            log.warning(
                "Routing attempt %d produced no decision: %s",
                attempt,
                (result or {}).get("parsing_error"),
            )

        if decision is None:
            log.error("Router returned no usable decision - finishing with what we have.")
            return {"next_agent": FINISH, "steps": 1}

        target = str(getattr(decision, "next_agent", FINISH))
        instruction = str(getattr(decision, "reasoning", ""))
        log.info("step %d/%d -> %s (%s)", step + 1, budget, target, instruction)
        return {"next_agent": target, "instruction": instruction, "steps": 1}

    def _refusal_route(self) -> str:
        """Where an unclassifiable task goes. FINISH only if nothing can run."""
        names = [spec.name for spec in self.specs]
        if REFUSAL_FALLBACK in names:
            return REFUSAL_FALLBACK
        return names[0] if names else FINISH

    def _make_specialist_node(self, name: str) -> Callable[[SupervisorState], dict[str, Any]]:
        """Wrap a specialist subgraph as a supervisor node."""
        subgraph = self._subgraphs[name]
        has_tools = bool(next(s for s in self.specs if s.name == name).tools)

        def node(state: SupervisorState) -> dict[str, Any]:
            seeded = trim(list(state["messages"]), 4)
            # A specialist gets a fresh state on every delegation, so it has no
            # memory of work it already did. Pushing the inventory is what stops
            # the second delegation re-fetching what the first one downloaded.
            inventory = downloaded_inventory(state.get("task_id", ""))
            if inventory:
                seeded = [*seeded, SystemMessage(content=inventory)]
            # The router already generated a justification for this delegation
            # and we already paid for it; it used to be logged and discarded,
            # leaving the specialist to infer its task from the raw transcript.
            instruction = state.get("instruction", "").strip()
            if instruction:
                seeded = [*seeded, HumanMessage(content=f"Your task: {instruction}")]
            payload = {"messages": seeded, "iterations": 0, "last_error": ""}
            try:
                result = subgraph.invoke(
                    payload, config={"recursion_limit": self.settings.recursion_limit}
                )
                # Only what the subgraph appended. Running last_text over the whole
                # list walks back into the seed messages, so a specialist that
                # produced nothing echoes its own input back - double-tagged.
                appended = list(result["messages"])[len(seeded) :]
                content = last_text(appended)
                evidence = tool_evidence(appended, has_tools=has_tools)
            except Exception as exc:  # noqa: BLE001 - one specialist failing is recoverable
                log.error("%s failed: %s", name, exc)
                content = f"{name} failed with error: {exc}"
                evidence = "the specialist failed before producing evidence"

            return {
                "messages": [AIMessage(content=f"[{name}] ({evidence})\n{content}", name=name)],
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
        capped = get_llm().bind(max_tokens=self.settings.max_answer_tokens)
        finalizer = with_effort(capped, self.settings.finalizer_effort)
        try:
            # text_of, not str(...content): with thinking enabled the content is
            # a list of typed blocks, and str() over it yields the repr - which
            # once shipped `[{'signature': 'EsEECpAB...` as a final answer.
            content = clean_answer(text_of(finalizer.invoke(normalize(messages))))
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
    def solve(
        self, question: str, task_id: str = "local", callbacks: list[Any] | None = None
    ) -> Solution:
        """Run the graph on one question and return the answer with its cost.

        ``answer()`` returns only the text, which is all the app and CLI need.
        The harness needs the delegation count too: iteration caps and timeouts
        should be set from the distribution of successful runs, and that was
        unobservable while every metric record reported zero steps.
        """
        final_state = self.graph.invoke(
            initial_supervisor_state([HumanMessage(content=question)], task_id),
            config=trace_config(task_id, callbacks),
        )
        steps = int(final_state.get("steps", 0))
        for message in reversed(list(final_state["messages"])):
            if getattr(message, "name", "") == FINAL_ANSWER:
                return Solution(text=str(message.content).strip(), steps=steps)
        return Solution(text="", steps=steps)

    def answer(
        self, question: str, task_id: str = "local", callbacks: list[Any] | None = None
    ) -> str:
        """Run the graph on one question and return the final answer text."""
        return self.solve(question, task_id=task_id, callbacks=callbacks).text


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


def solve_question(
    question: str, task_id: str = "local", callbacks: list[Any] | None = None
) -> Solution:
    """Like answer_question, but keeps the delegation count."""
    return get_orchestrator().solve(question, task_id=task_id, callbacks=callbacks)
