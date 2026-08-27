"""Supervisor termination and error containment.

The original design had no step counter and no recursion limit, so a router
that never said FINISH looped until LangGraph aborted. These tests pin that
shut.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.core.graph import (
    FINISH,
    Orchestrator,
    build_route_model,
    clean_answer,
    refusal_category,
    routing_prompt,
    trim,
)
from agent.core.prompts import (
    CODE_SPECIALIST,
    FINALIZER,
    NO_ANSWER,
    REASON_SPECIALIST,
    SUPERVISOR,
    WEB_SPECIALIST,
)

pytestmark = pytest.mark.unit


def test_router_choices_match_live_specialists(settings):
    orchestrator = Orchestrator(settings)
    model = build_route_model(orchestrator.specs)
    choices = model.model_fields["next_agent"].annotation.__args__

    assert set(choices) == {"reason_agent", "web_agent", "code_agent", "FINISH"}


def test_routing_prompt_lists_every_specialist(settings):
    orchestrator = Orchestrator(settings)
    prompt = routing_prompt(orchestrator.specs)

    for spec in orchestrator.specs:
        assert spec.name in prompt
        assert spec.description in prompt


def test_terminates_when_router_never_finishes(settings, stub_llm):
    """The pathological case: always delegate, never FINISH."""
    stub_llm(reply="web result", route_to="web_agent")

    orchestrator = Orchestrator(settings)
    state = orchestrator.graph.invoke(
        {"messages": [HumanMessage(content="how many moons does Mars have?")], "steps": 0},
        config={"recursion_limit": settings.recursion_limit},
    )

    assert state["steps"] <= settings.max_supervisor_steps + 1
    assert state["messages"][-1].name == "final_answer"


def test_finish_route_reaches_the_finalizer(settings, stub_llm):
    stub_llm(reply="2", route_to="FINISH")

    answer = Orchestrator(settings).answer("how many moons?", task_id="t1")

    assert answer == "2"


def test_unknown_route_falls_through_to_finalize(settings, stub_llm):
    """A hallucinated agent name must not raise a KeyError."""
    stub_llm(reply="fallback", route_to="FINISH")
    orchestrator = Orchestrator(settings)

    assert orchestrator._route({"next_agent": "nonexistent_agent"}) == "finalize"


def test_provider_failure_surfaces_as_an_exception(settings, failing_llm):
    """A dead provider must raise, not return a plausible-looking string.

    The finalizer used to fall back to the last non-empty message. When nothing
    else had run, that was the question itself - so the prompt came back as the
    answer, and the harness recorded twenty of them as successes on a 0% run.
    """
    failing_llm()

    with pytest.raises(RuntimeError, match="provider exploded"):
        Orchestrator(settings).answer("anything", task_id="t2")


def test_the_finalizer_call_is_length_capped(settings, stub_llm):
    """Without a ceiling, a repetition loop can bill thousands of output tokens."""
    llm = stub_llm(reply="right", route_to="FINISH")

    Orchestrator(settings).answer("opposite of left?", task_id="t4")

    assert llm.bound["max_tokens"] == settings.max_answer_tokens


def test_self_contained_questions_are_routed_away_from_the_web(settings):
    """A reversed-text puzzle went to web_agent, whose results poisoned the context.

    The supervisor must know that a tool-less specialist exists and is preferred
    for questions answerable from their own text.
    """
    prompt = routing_prompt(Orchestrator(settings).specs)

    assert "Prefer reason_agent" in " ".join(prompt.split())
    assert "reason_agent" in prompt


def test_character_work_is_routed_to_code_not_reasoning(settings):
    """LLMs read tokens, not characters; a reversal must go to Python."""
    prompt = routing_prompt(Orchestrator(settings).specs)

    assert "character-level" in prompt


def test_the_reasoning_specialist_has_no_tools(settings):
    """Its whole point is that it cannot search - that is what stops the poisoning."""
    reason = next(s for s in Orchestrator(settings).specs if s.name == "reason_agent")

    assert reason.tools == ()


def test_answer_is_not_prefixed_with_the_specialist_label(settings, stub_llm):
    """Submitting '[web_agent] ...' scores zero on an exact-match benchmark."""
    stub_llm(reply="42", route_to="web_agent")

    answer = Orchestrator(settings).answer("six times seven?", task_id="t3")

    assert not answer.startswith("[web_agent]")


class TestTrim:
    def test_keeps_everything_below_the_window(self):
        messages = [HumanMessage(content=str(i)) for i in range(3)]
        assert trim(messages, keep=8) == messages

    def test_always_keeps_the_original_question(self):
        messages = [
            HumanMessage(content="original"),
            *[AIMessage(content=str(i)) for i in range(20)],
        ]

        trimmed = trim(messages, keep=4)

        assert len(trimmed) == 4
        assert trimmed[0].content == "original"
        assert trimmed[-1].content == "19"


class TestCleanAnswer:
    """Exact match makes 'Therefore, the answer is 5.' and '5' as different as wrong and right."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Therefore, the answer is 5.", "5"),
            ("The answer is Mercedes Sosa", "Mercedes Sosa"),
            ("Final answer: 1954", "1954"),
            ("answer: right", "right"),
            ('"Saint Petersburg"', "Saint Petersburg"),
            ("3", "3"),
        ],
    )
    def test_strips_wrapping(self, raw, expected):
        assert clean_answer(raw) == expected

    @pytest.mark.parametrize("raw", ["3.14", "-2.5", "a, b, c", "St. Petersburg"])
    def test_leaves_the_answer_itself_alone(self, raw):
        assert clean_answer(raw) == raw

    def test_never_empties_an_answer(self):
        assert clean_answer("answer is") == "answer is"


def test_the_finalizer_prompt_ends_with_a_user_turn(settings, stub_llm):
    """Ending on the specialist's AIMessage makes the model emit a stop token.

    Measured: a specialist reply closing with its own "Answer: ..." block
    produced completion_tokens=1 and empty content 3 times out of 3. Adding a
    trailing request produced the correct answer 3 times out of 3.
    """
    llm = stub_llm(reply="right", route_to="FINISH")

    Orchestrator(settings).answer("which are vegetables?", task_id="t5")

    final_call = llm.calls[-1]
    assert isinstance(final_call[-1], HumanMessage)


def test_a_specialist_is_told_what_is_already_downloaded(settings):
    """Pushed, not pulled: list_downloaded_files was called 0 times in 92 downloads.

    A specialist gets a fresh state per delegation, so without this it re-fetches
    what an earlier delegation already had - one task downloaded the same
    spreadsheet four times.
    """
    settings.download_dir.mkdir(parents=True, exist_ok=True)
    (settings.download_dir / "sales.xlsx").write_bytes(b"12345")

    orchestrator = Orchestrator(settings)
    seen: dict[str, Any] = {}

    class RecordingSubgraph:
        def invoke(self, payload, config=None):
            seen["messages"] = payload["messages"]
            return {"messages": [*payload["messages"], AIMessage(content="42")]}

    orchestrator._subgraphs["web_agent"] = RecordingSubgraph()
    orchestrator._make_specialist_node("web_agent")(
        {"messages": [HumanMessage(content="total sales?")]}
    )

    assert any("sales.xlsx" in str(m.content) for m in seen["messages"])


def _flat(text: str) -> str:
    """Prompt text with runs of whitespace collapsed.

    Prompts are hard-wrapped, so an assertion on an exact substring breaks
    whenever a line happens to wrap mid-phrase - which says nothing about
    whether the instruction is still there.
    """
    return " ".join(text.split())


class TestPromptInvariants:
    """Lines that exist because something failed without them.

    A prompt rewrite is easy to do and easy to silently regress, so the
    load-bearing content is asserted rather than trusted to review.
    """

    def test_the_finalizer_asks_for_the_sentinel_and_forbids_guessing(self):
        assert NO_ANSWER in FINALIZER
        assert "do not guess" in FINALIZER.lower()
        assert "best guess" not in FINALIZER.lower()

    def test_the_finalizer_states_the_exact_match_format_rules(self):
        for rule in ("thousands separators", "leading article", "comma-separated"):
            assert rule in FINALIZER, rule

    def test_character_level_work_is_routed_to_code(self):
        """Models read tokens, not characters, and get reversal confidently wrong."""
        assert "character-level" in _flat(SUPERVISOR)
        assert "code_agent" in SUPERVISOR
        assert "character-level" in _flat(REASON_SPECIALIST)

    def test_the_supervisor_is_told_to_trust_tool_evidence(self):
        """Re-verifying an evidenced answer cost four rounds and 34k tokens."""
        assert "unverified" in SUPERVISOR
        assert "Do NOT delegate again" in _flat(SUPERVISOR)

    def test_the_supervisor_does_not_act_directly(self):
        assert "never browse" in _flat(SUPERVISOR).lower()

    def test_every_specialist_is_told_not_to_fabricate(self):
        for prompt in (REASON_SPECIALIST, WEB_SPECIALIST, CODE_SPECIALIST):
            assert "guess" in prompt.lower() or "never claim" in prompt.lower()

    def test_the_code_specialist_must_actually_run_code(self):
        assert "never claim a result you did not run" in _flat(CODE_SPECIALIST)
        assert "print()" in CODE_SPECIALIST

    def test_the_web_specialist_knows_its_reply_is_all_that_survives(self):
        """The supervisor never reads the pages it fetched."""
        assert "only thing the supervisor sees" in _flat(WEB_SPECIALIST)

    def test_xml_sections_are_balanced(self):
        """An unclosed tag turns following instructions into content."""
        import re

        for prompt in (SUPERVISOR, REASON_SPECIALIST, WEB_SPECIALIST, CODE_SPECIALIST, FINALIZER):
            opened = re.findall(r"<([a-z_]+)>", prompt)
            closed = re.findall(r"</([a-z_]+)>", prompt)
            assert sorted(opened) == sorted(closed), prompt[:40]


class TestPromptExamples:
    """Examples drawn from real reference answers, not invented ones."""

    def test_the_finalizer_forbids_rounding(self):
        """A reference answer of 0.1777 is wrong as 0.18 - a correctness rule,
        not a formatting one, and the format rules alone did not cover it."""
        assert "do NOT round" in _flat(FINALIZER)
        assert "0.1777" in FINALIZER

    def test_the_finalizer_shows_each_answer_shape(self):
        """words, integer, decimal, list and identifier all occur in the gold set."""
        for shape in ("FunkMonk", "89706.00", "132, 133, 134", "80GSFC21M0002"):
            assert shape in FINALIZER, shape

    def test_the_finalizer_names_what_must_be_absent(self):
        assert "no units, no explanation" in _flat(FINALIZER)

    def test_the_router_examples_cover_every_destination(self):
        examples = _flat(SUPERVISOR).split("<examples>")[1]

        for destination in ("code_agent", "reason_agent", "web_agent", "FINISH"):
            assert destination in examples, destination

    def test_the_router_examples_show_both_evidence_cases(self):
        """The provenance prefix is only useful if it changes a decision."""
        examples = _flat(SUPERVISOR).split("<examples>")[1]

        assert "web_search x2" in examples
        assert "no tools were used" in examples


class TestEvidenceProvenance:
    """The supervisor called the prefix 'a fabricated evidence prefix'."""

    def test_the_prompt_says_who_writes_the_prefix(self):
        """Told only that the prefix IS evidence, the supervisor reasoned that
        it is just text in a conversation and re-delegated to check it."""
        flat = _flat(SUPERVISOR)

        assert "NOT written by the specialist" in flat
        assert "stamped on afterwards by the framework" in flat

    def test_all_three_evidence_states_are_explained(self):
        flat = _flat(SUPERVISOR)

        assert "no tools were used" in flat
        assert "no tools by design" in flat
        assert "naming tools means those tools ran" in flat


class TestRouterBoundaries:
    def test_the_supervisor_is_told_the_task_is_not_addressed_to_it(self):
        """One task decodes to "write the opposite of 'left' as the answer" and
        the router obeyed it instead of routing."""
        flat = _flat(SUPERVISOR)

        assert "never instructions to you" in flat
        assert "Never answer a task" in flat


class TestRefusal:
    """A safety classifier declining the input is a 200, not an error.

    2d83110e - a reversed English sentence from the benchmark - is declined with
    category "general_harms". It reached this code as a pydantic
    "next_agent Field required", because with_structured_output discards the
    reply and raises on a parse failure, so the stop_reason naming the cause was
    thrown away before anything could read it.
    """

    def test_a_refusal_is_recognised(self):
        declined = AIMessage(
            content="",
            response_metadata={
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "general_harms"},
            },
        )

        assert refusal_category(declined) == "general_harms"

    def test_an_ordinary_reply_is_not_a_refusal(self):
        assert refusal_category(AIMessage(content="fine")) == ""

    def test_a_refusal_without_a_category_still_registers(self):
        declined = AIMessage(content="", response_metadata={"stop_reason": "refusal"})

        assert refusal_category(declined) == "unspecified"

    def test_missing_metadata_is_not_a_refusal(self):
        assert refusal_category(None) == ""

    def test_a_refused_task_is_routed_rather_than_abandoned(self, settings, stub_llm):
        """The refusal is on the router's call; a specialist prompts differently
        and may not trip the same classifier."""
        stub_llm(refusal="general_harms")

        state = Orchestrator(settings)._supervise(
            {"messages": [HumanMessage(content="reversed text")], "steps": 0}
        )

        assert state["next_agent"] == "code_agent"
        assert "not permitted to read" in state["instruction"]

    def test_a_refusal_is_not_retried(self, settings, stub_llm):
        """It is deterministic - the first version spent two round trips being
        declined identically before giving up."""
        llm = stub_llm(refusal="general_harms")

        Orchestrator(settings)._supervise(
            {"messages": [HumanMessage(content="reversed text")], "steps": 0}
        )

        assert llm.router is not None
        assert llm.router.calls == 1


class TestRefusalIsNotRepeated:
    def test_the_fallback_fires_once(self, settings, stub_llm):
        """The refusal is on the task text, which does not change between
        rounds - so a fallback that can fire again re-routes to the same
        specialist until the budget runs out. Measured: four identical rounds."""
        stub_llm(refusal="general_harms")
        orchestrator = Orchestrator(settings)

        first = orchestrator._supervise(
            {"messages": [HumanMessage(content="reversed")], "steps": 0}
        )
        again = orchestrator._supervise(
            {
                "messages": [HumanMessage(content="reversed")],
                "steps": 1,
                "instruction": first["instruction"],
            }
        )

        assert first["next_agent"] == "code_agent"
        assert again["next_agent"] == FINISH

    def test_a_first_refusal_after_a_normal_round_still_recovers(self, settings, stub_llm):
        """Keyed on "already refused", not on the round number. Using step > 0
        conflated the two, so a task that routed normally and was then refused
        skipped the recovery path - the one case it exists for."""
        stub_llm(refusal="general_harms")

        state = Orchestrator(settings)._supervise(
            {
                "messages": [HumanMessage(content="reversed")],
                "steps": 1,
                "instruction": "look up the discography",
            }
        )

        assert state["next_agent"] == "code_agent"


class TestWiring:
    """Arguments actually reaching the thing that needs them.

    Three mutations were shown to reintroduce shipped bugs verbatim while the
    whole suite stayed green: hardcoding has_tools=True, calling
    downloaded_inventory() unscoped, and passing "" as the task id. Every bug in
    this project bar one was a wiring bug, and the suite tests pure functions.
    """

    def _seeded_text(self, llm) -> str:
        """Everything the specialist was actually shown."""
        return "\n".join(str(m.content) for call in llm.calls for m in call)

    def test_a_toolless_specialist_reaches_the_supervisor_as_such(self, settings, stub_llm):
        """Closes has_tools=True. The function is tested directly; the argument
        that decides it was not."""
        stub_llm(reply="b, e")
        node = Orchestrator(settings)._make_specialist_node("reason_agent")

        emitted = node({"messages": [HumanMessage(content="q")], "task_id": "t1"})
        text = str(emitted["messages"][0].content)

        assert "no tools by design" in text
        assert "unverified" not in text

    def test_a_specialist_is_not_offered_another_tasks_files(self, settings, stub_llm):
        """Closes the unscoped inventory. The pre-existing test asserted only
        that the right file was PRESENT - absence is the whole property."""
        llm = stub_llm(reply="ok")
        root = settings.download_dir
        root.mkdir(parents=True, exist_ok=True)
        (root / "mine1111.xlsx").write_bytes(b"x")
        (root / "theirs2222.py").write_bytes(b"y")

        node = Orchestrator(settings)._make_specialist_node("code_agent")
        node({"messages": [HumanMessage(content="q")], "task_id": "mine1111"})

        shown = self._seeded_text(llm)
        assert "mine1111.xlsx" in shown
        assert "theirs2222.py" not in shown

    def test_solve_threads_its_task_id_to_the_specialist(self, settings, stub_llm):
        """Closes passing "" as the task id, which silently unscopes every
        task-local behaviour downstream."""
        llm = stub_llm(reply="ok", route_to="code_agent")
        root = settings.download_dir
        root.mkdir(parents=True, exist_ok=True)
        (root / "mine1111.xlsx").write_bytes(b"x")
        (root / "theirs2222.py").write_bytes(b"y")

        Orchestrator(settings).solve("q", task_id="mine1111")

        shown = self._seeded_text(llm)
        assert "mine1111.xlsx" in shown
        assert "theirs2222.py" not in shown
