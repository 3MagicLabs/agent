"""Observability plumbing: tracing config, token totals, metrics sink."""

from __future__ import annotations

import os

import pytest

from agent.config import Settings
from agent.obs.metrics import MetricsRecorder, TaskMetric, read_metrics
from agent.obs.tracing import configure_tracing, total_tokens, trace_config

pytestmark = pytest.mark.unit


class TestTracing:
    def test_disabled_without_a_key(self, settings):
        assert configure_tracing(settings) is False

    def test_sets_both_env_var_families(self):
        """LangChain renamed these; the installed version decides which it reads."""
        assert configure_tracing(Settings(langsmith_api_key="k", langsmith_project="p")) is True

        assert os.environ["LANGSMITH_TRACING"] == "true"
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGSMITH_PROJECT"] == "p"
        assert os.environ["LANGCHAIN_PROJECT"] == "p"

    def test_trace_config_carries_the_task_id(self):
        config = trace_config("task-123")

        assert config["metadata"]["task_id"] == "task-123"
        assert config["run_name"] == "task:task-123"
        assert config["recursion_limit"] > 0

    def test_callbacks_are_only_added_when_present(self):
        assert "callbacks" not in trace_config("t")
        assert trace_config("t", ["cb"])["callbacks"] == ["cb"]


class TestTokenTotals:
    def test_none_handler_yields_empty(self):
        assert total_tokens(None) == {}

    def test_sums_across_models(self):
        class Handler:
            usage_metadata = {
                "model-a": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                "model-b": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            }

        assert total_tokens(Handler()) == {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        }

    def test_tolerates_missing_keys(self):
        class Handler:
            usage_metadata = {"m": {"input_tokens": 4}}

        assert total_tokens(Handler())["input_tokens"] == 4


class TestMetrics:
    def test_summary_of_an_empty_run(self, tmp_path):
        assert MetricsRecorder(tmp_path / "m.jsonl").summary() == {"count": 0}

    def test_summary_aggregates_statuses(self, tmp_path):
        recorder = MetricsRecorder(tmp_path / "m.jsonl")
        recorder.record(TaskMetric("a", "q", latency_s=1.0, tokens={"total_tokens": 10}))
        recorder.record(TaskMetric("b", "q", status="timeout", latency_s=3.0))
        recorder.record(TaskMetric("c", "q", status="error", latency_s=2.0))

        summary = recorder.summary()

        assert summary["count"] == 3
        assert summary["ok"] == 1
        assert summary["timeouts"] == 1
        assert summary["errors"] == 1
        assert summary["total_tokens"] == 10

    def test_rows_are_persisted_as_jsonl(self, tmp_path):
        path = tmp_path / "m.jsonl"
        recorder = MetricsRecorder(path)
        recorder.record(TaskMetric("a", "q", answer="1"))
        recorder.record(TaskMetric("b", "q", answer="2"))

        rows = read_metrics(path)

        assert [row["task_id"] for row in rows] == ["a", "b"]

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text('{"task_id": "a"}\nnot json\n{"task_id": "b"}\n')

        assert len(read_metrics(path)) == 2

    def test_metric_is_immutable(self):
        metric = TaskMetric("a", "q")
        with pytest.raises(AttributeError):
            metric.answer = "changed"  # type: ignore[misc]

    def test_recorder_rows_are_a_tuple_snapshot(self, tmp_path):
        recorder = MetricsRecorder(tmp_path / "m.jsonl")
        recorder.record(TaskMetric("a", "q"))
        snapshot = recorder.rows
        recorder.record(TaskMetric("b", "q"))

        assert len(snapshot) == 1
        assert len(recorder.rows) == 2
