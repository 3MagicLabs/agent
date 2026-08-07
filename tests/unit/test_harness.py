"""Benchmark runner: budgets, caching and failure isolation."""

from __future__ import annotations

import time

import pytest

from agent.config import Settings, set_settings
from agent.eval.harness import AnswerCache, BenchmarkRunner, build_prompt

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

    def test_attachment_instructs_the_download_tool(self):
        prompt = build_prompt({"task_id": "abc", "question": "sum it", "file_name": "data.xlsx"})

        assert "data.xlsx" in prompt
        assert "download_task_file" in prompt
        assert "task_id='abc'" in prompt


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
