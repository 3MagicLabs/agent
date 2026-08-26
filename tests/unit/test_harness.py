"""Benchmark runner: budgets, caching and failure isolation."""

from __future__ import annotations

import time

import pytest

from agent.config import Settings, set_settings
from agent.core.prompts import FINALIZER, NO_ANSWER
from agent.eval.harness import AnswerCache, BenchmarkRunner, build_prompt, rejection_reason
from agent.obs.metrics import TaskMetric

pytestmark = pytest.mark.unit

QUESTIONS = [
    {"task_id": "t1", "question": "first question"},
    {"task_id": "t2", "question": "second question"},
]


def make_runner(answer_fn, settings=None) -> BenchmarkRunner:
    return BenchmarkRunner(answer_fn=answer_fn, settings=settings)


class TestBuildPrompt:
    def test_plain_question_is_unchanged(self):
        assert build_prompt({"task_id": "t", "question": "hi"}) == "hi"

    def test_attachment_is_flagged_with_its_name_and_id(self):
        prompt = build_prompt({"task_id": "abc", "question": "sum it", "file_name": "data.xlsx"})

        assert "data.xlsx" in prompt
        assert "abc" in prompt

    def test_the_named_tools_exist(self):
        """The prompt names tools, so a rename must break a test, not a run.

        Dropping the names entirely was tried and measured: the xlsx task
        stopped fetching its attachment and answered 0.00 instead of 17949.59.
        The instruction stays; this guards the coupling it creates.
        """
        from agent.eval.harness import DOWNLOAD_TOOL, READ_TOOL
        from agent.tools import registered

        names = {spec.name for spec in registered()}

        assert {DOWNLOAD_TOOL, READ_TOOL} <= names

    def test_the_prompt_uses_the_registered_names(self):
        from agent.eval.harness import DOWNLOAD_TOOL, READ_TOOL

        prompt = build_prompt({"task_id": "abc", "question": "sum it", "file_name": "data.xlsx"})

        assert DOWNLOAD_TOOL in prompt
        assert READ_TOOL in prompt


class TestRun:
    def test_answers_every_task_and_caches(self, settings):
        runner = make_runner(lambda q, tid, cb: f"answer-{tid}")

        list(runner.run(QUESTIONS, reuse_cache=False))

        assert runner.cache.load() == {"t1": "answer-t1", "t2": "answer-t2"}

    def test_cached_tasks_are_skipped(self, settings):
        calls: list[str] = []

        def answer(_q, task_id, _cb):
            calls.append(task_id)
            return "x"

        runner = make_runner(answer)
        runner.cache.save({"t1": "already done"})

        list(runner.run(QUESTIONS, reuse_cache=True))

        assert calls == ["t2"]

    def test_no_cache_reruns_everything(self, settings):
        calls: list[str] = []
        runner = make_runner(lambda _q, tid, _cb: calls.append(tid) or "x")
        runner.cache.save({"t1": "already done"})

        list(runner.run(QUESTIONS, reuse_cache=False))

        assert calls == ["t1", "t2"]

    def test_one_failure_does_not_stop_the_batch(self, settings):
        def answer(_q, task_id, _cb):
            if task_id == "t1":
                raise RuntimeError("boom")
            return "second"

        runner = make_runner(answer)
        list(runner.run(QUESTIONS, reuse_cache=False))

        statuses = {m.task_id: m.status for m in runner.recorder.rows}
        assert statuses == {"t1": "error", "t2": "ok"}
        assert runner.cache.load() == {"t2": "second"}

    def test_malformed_items_are_skipped(self, settings):
        runner = make_runner(lambda _q, tid, _cb: "x")
        items = [{"question": "no id"}, {"task_id": "t9", "question": "fine"}]

        list(runner.run(items, reuse_cache=False))

        assert runner.cache.load() == {"t9": "x"}

    def test_final_progress_is_marked_done(self, settings):
        runner = make_runner(lambda _q, _t, _c: "x")

        events = list(runner.run(QUESTIONS, reuse_cache=False))

        assert events[-1].done is True


class TestBudgets:
    def test_a_hung_task_times_out(self, tmp_path):
        set_settings(Settings(log_dir=tmp_path, per_question_timeout_s=0.2, total_budget_s=60))

        def hang(_q, _t, _c):
            time.sleep(5)
            return "never"

        metric = make_runner(hang).run_one(QUESTIONS[0])

        assert metric.status == "timeout"
        assert "exceeded" in metric.error

    def test_timeout_does_not_block_later_tasks(self, tmp_path):
        set_settings(Settings(log_dir=tmp_path, per_question_timeout_s=0.2, total_budget_s=60))

        def slow_first(_q, task_id, _c):
            if task_id == "t1":
                time.sleep(5)
            return "fast"

        runner = make_runner(slow_first)
        started = time.monotonic()
        list(runner.run(QUESTIONS, reuse_cache=False))

        assert time.monotonic() - started < 4
        assert runner.cache.load() == {"t2": "fast"}

    def test_total_budget_stops_the_run(self, tmp_path):
        set_settings(Settings(log_dir=tmp_path, per_question_timeout_s=30, total_budget_s=0.05))

        def slow(_q, _t, _c):
            time.sleep(0.1)
            return "x"

        events = list(make_runner(slow).run(QUESTIONS, reuse_cache=False))

        assert events[-1].done is True
        assert "budget" in events[-1].message


class TestAnswerCache:
    def test_missing_file_loads_empty(self, tmp_path):
        assert AnswerCache(tmp_path / "nope.json").load() == {}

    def test_corrupt_file_loads_empty(self, tmp_path):
        path = tmp_path / "answers.json"
        path.write_text("{not json")

        assert AnswerCache(path).load() == {}

    def test_round_trips(self, tmp_path):
        cache = AnswerCache(tmp_path / "a.json")
        cache.save({"t": "value"})

        assert AnswerCache(tmp_path / "a.json").load() == {"t": "value"}


class TestSubmit:
    def test_empty_cache_refuses_to_submit(self, settings):
        with pytest.raises(ValueError, match="empty"):
            make_runner(lambda *_: "x").submit("me", "code-url")


class TestRejectionReason:
    """The predicate alone: string in, reason out. No runner, no fixtures."""

    @pytest.mark.parametrize("answer", ["", "   ", "\n\t "])
    def test_blank_answers_are_rejected(self, answer):
        assert rejection_reason(answer) == "empty answer"

    def test_specialist_tag_means_the_finalizer_never_ran(self):
        assert "web_agent" in rejection_reason("[web_agent]\n1954")

    @pytest.mark.parametrize("answer", ["1954", "Mercedes Sosa", "3, 4, 5", "-2.5"])
    def test_real_answers_are_accepted(self, answer):
        assert rejection_reason(answer) == ""

    def test_runaway_generation_is_rejected(self):
        """A repetition loop once emitted 7,876 characters of one Thai sentence."""
        assert "too long" in rejection_reason("word " * 400)

    def test_a_long_but_plausible_list_is_accepted(self):
        assert rejection_reason(", ".join(str(n) for n in range(100))) == ""


class TestAnswerValidation:
    """The predicate wired in: a returned string is not by itself a success.

    ``run_one`` used to stamp "ok" whenever no exception escaped the worker
    thread, so an empty or unfinalized answer was cached and submitted.
    """

    def test_empty_answer_is_recorded_as_an_error(self, settings):
        metric = make_runner(lambda _q, _t, _c: "").run_one(QUESTIONS[0])

        assert metric.status == "error"
        assert metric.error == "empty answer"

    def test_unfinalized_output_is_recorded_as_an_error(self, settings):
        metric = make_runner(lambda _q, _t, _c: "[web_agent]\n1954").run_one(QUESTIONS[0])

        assert metric.status == "error"
        # Quarantined, not deleted: the rejected text stays readable in the metrics.
        assert metric.answer == "[web_agent]\n1954"

    def test_a_real_answer_still_passes(self, settings):
        metric = make_runner(lambda _q, _t, _c: "1954").run_one(QUESTIONS[0])

        assert metric.status == "ok"
        assert metric.error == ""

    def test_a_rejected_answer_is_never_cached(self, settings):
        runner = make_runner(lambda _q, _t, _c: "")

        list(runner.run(QUESTIONS, reuse_cache=False))

        assert runner.cache.load() == {}


class TestPacing:
    """A 20-task run spent its whole per-minute quota on the first six tasks."""

    def _metric(self, tokens: int, latency: float) -> TaskMetric:
        return TaskMetric(
            task_id="t", question="q", tokens={"total_tokens": tokens}, latency_s=latency
        )

    def test_waits_for_the_rest_of_the_token_window(self, tmp_path):
        set_settings(Settings(log_dir=tmp_path, tokens_per_minute=12000))
        # 6000 tokens is half a minute of quota; 10s were already spent running.
        assert make_runner(None).pause_for(self._metric(6000, 10.0)) == pytest.approx(20.0)

    def test_a_slow_task_needs_no_extra_wait(self, tmp_path):
        set_settings(Settings(log_dir=tmp_path, tokens_per_minute=12000))
        assert make_runner(None).pause_for(self._metric(1000, 90.0)) == 0.0

    def test_pacing_is_disabled_by_a_zero_rate(self, tmp_path):
        set_settings(Settings(log_dir=tmp_path, tokens_per_minute=0))
        assert make_runner(None).pause_for(self._metric(60000, 0.0)) == 0.0

    def test_unmeasured_tokens_do_not_stall_the_run(self, tmp_path):
        set_settings(Settings(log_dir=tmp_path, tokens_per_minute=12000))
        assert make_runner(None).pause_for(self._metric(0, 1.0)) == 0.0


class TestNoAnswerSentinel:
    """The finalizer needs a legal way to fail.

    Without one it was instructed to guess, and a fluent guess passes every
    shape check - it differs from a real answer only in provenance, which
    rejection_reason cannot see.
    """

    def test_the_sentinel_is_rejected(self):
        assert rejection_reason(NO_ANSWER) == "no answer found"

    def test_surrounding_whitespace_does_not_smuggle_it_through(self):
        assert rejection_reason(f"  {NO_ANSWER}\n") == "no answer found"

    def test_an_answer_that_merely_mentions_it_is_still_accepted(self):
        """Only the bare sentinel means failure; the word may appear in prose."""
        assert rejection_reason(f"the file contained {NO_ANSWER}") == ""

    def test_the_prompt_asks_for_the_sentinel_rather_than_a_guess(self):
        """Guard against the 'give your single best guess anyway' line returning."""
        assert NO_ANSWER in FINALIZER
        assert "best guess" not in FINALIZER.lower()
