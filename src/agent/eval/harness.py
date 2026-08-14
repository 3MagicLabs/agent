"""Benchmark runner.

Shared by the Gradio app, the CLI and the test suite so all three exercise the
same code path. Answers are cached to disk as they are produced, and
submission is a separate step — a dropped connection must never lose a run.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from agent.config import Settings, get_settings
from agent.obs.logging import get_logger
from agent.obs.metrics import MetricsRecorder, TaskMetric
from agent.obs.tracing import total_tokens, usage_callback

log = get_logger("eval.harness")

AnswerFn = Callable[..., str]

#: Floor for a per-task timeout derived from a nearly exhausted total budget.
MIN_TASK_TIMEOUT_S = 1.0

#: Marker the supervisor stamps on specialist output ("[web_agent]\n..."). Seeing
#: it in a final answer proves the finalizer never ran, so the text is
#: intermediate output rather than an answer.
_SPECIALIST_TAG = re.compile(r"^\[(\w+)\]")

#: Generous ceiling on a graded answer. GAIA answers are a number, a short
#: string or a comma-separated list; anything near this is runaway generation,
#: not an answer. Deliberately far above any legitimate value so the rule stays
#: structural - it never has to know what the task was.
MAX_ANSWER_CHARS = 1000


@dataclass(frozen=True, slots=True)
class Progress:
    """One step of a run, streamed to whatever is driving it."""

    index: int
    total: int
    message: str
    metric: TaskMetric | None = None
    done: bool = False


def build_prompt(item: dict[str, Any]) -> str:
    """Render a task into the prompt, flagging any attachment."""
    prompt = str(item.get("question", ""))
    task_id = item.get("task_id", "")
    if item.get("file_name"):
        prompt += (
            f"\n\n[This task has an attached file: {item['file_name']}. "
            f"Call download_task_file with task_id='{task_id}' to retrieve it, "
            f"then read_file on the returned path.]"
        )
    if item.get("file_url"):
        prompt += f"\n[File URL: {item['file_url']}]"
    return prompt


def rejection_reason(answer: str) -> str:
    """Why ``answer`` is not a usable answer, or "" when it is.

    Note the inversion: an empty return means accepted. Callers treat any
    non-empty string as both the rejection and its explanation, so it can be
    recorded directly as ``TaskMetric.error``.

    This asks only whether the agent produced an answer at all - never whether
    it is correct. Grading lives in ``agent.eval.scorers``.
    """
    stripped = answer.strip()
    if not stripped:
        return "empty answer"
    tag = _SPECIALIST_TAG.match(stripped)
    if tag:
        return f"unfinalized {tag.group(1)} output"
    if len(stripped) > MAX_ANSWER_CHARS:
        return f"answer too long ({len(stripped)} chars)"
    return ""


class AnswerCache:
    """Disk-backed task_id -> answer map. Reads and writes whole snapshots."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or get_settings().answer_cache

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, str]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def save(self, answers: dict[str, str]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Could not write answer cache: %s", exc)

    def clear(self) -> None:
        self._path.unlink(missing_ok=True)


class BenchmarkRunner:
    """Fetches tasks, answers them under hard budgets, and submits results."""

    def __init__(
        self,
        answer_fn: AnswerFn | None = None,
        settings: Settings | None = None,
        cache: AnswerCache | None = None,
        recorder: MetricsRecorder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or AnswerCache(self.settings.answer_cache)
        self.recorder = recorder or MetricsRecorder(self.settings.metrics_file)
        self._answer_fn = answer_fn

    @property
    def answer_fn(self) -> AnswerFn:
        """Resolved late so importing the harness never builds a model client."""
        if self._answer_fn is None:
            from agent.core.graph import answer_question

            resolved: AnswerFn = answer_question
            self._answer_fn = resolved
        return self._answer_fn

    # --- data ----------------------------------------------------------
    def fetch_questions(self) -> list[dict[str, Any]]:
        url = f"{self.settings.scoring_api_url}/questions"
        log.info("fetching questions from %s", url)
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        questions = response.json()
        if not questions:
            raise ValueError("Fetched questions list is empty.")
        return list(questions)

    # --- execution -----------------------------------------------------
    def run_one(self, item: dict[str, Any], timeout_s: float | None = None) -> TaskMetric:
        """Answer one task under a hard timeout, always returning a metric."""
        task_id = str(item.get("task_id", "unknown"))
        question = str(item.get("question", ""))
        limit = timeout_s if timeout_s is not None else self.settings.per_question_timeout_s
        handler = usage_callback()
        started = time.monotonic()

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                self.answer_fn,
                build_prompt(item),
                task_id,
                [handler] if handler else None,
            )
            answer = str(future.result(timeout=limit))
            status, error = "ok", ""
        except FutureTimeout:
            log.error("[%s] timed out after %.0fs", task_id, limit)
            answer, status, error = "", "timeout", f"exceeded {limit:.0f}s"
        except Exception as exc:
            log.exception("[%s] failed", task_id)
            answer, status, error = "", "error", f"{type(exc).__name__}: {exc}"
        finally:
            # Never wait: a hung task must not block the rest of the batch.
            executor.shutdown(wait=False)

        if status == "ok":
            reason = rejection_reason(answer)
            if reason:
                status, error = "error", reason

        return TaskMetric(
            task_id=task_id,
            question=question,
            answer=answer,
            status=status,
            error=error,
            latency_s=round(time.monotonic() - started, 2),
            tokens=total_tokens(handler),
            model=self.settings.model,
        )

    def run(
        self,
        questions: list[dict[str, Any]] | None = None,
        reuse_cache: bool = True,
    ) -> Iterator[Progress]:
        """Answer every task, yielding progress so callers can stream it."""
        items = questions if questions is not None else self.fetch_questions()
        answers = self.cache.load() if reuse_cache else {}
        started = time.monotonic()
        total = len(items)

        for index, item in enumerate(items, start=1):
            task_id = str(item.get("task_id", ""))
            if not task_id or item.get("question") is None:
                log.warning("skipping malformed item: %s", item)
                continue

            elapsed = time.monotonic() - started
            if elapsed > self.settings.total_budget_s:
                yield Progress(
                    index=index - 1,
                    total=total,
                    message=(
                        f"Stopped after {index - 1}/{total}: total budget "
                        f"({self.settings.total_budget_s:.0f}s) exhausted. "
                        f"Cached answers are still submittable."
                    ),
                    done=True,
                )
                return

            if task_id in answers:
                yield Progress(index, total, f"[{index}/{total}] {task_id}: cached")
                continue

            # The configured per-task timeout is the ceiling; shrink it when the
            # total budget is nearly gone, but never below a usable minimum.
            budget_left = max(self.settings.total_budget_s - elapsed, MIN_TASK_TIMEOUT_S)
            limit = min(self.settings.per_question_timeout_s, budget_left)
            metric = self.recorder.record(self.run_one(item, timeout_s=limit))
            if metric.status == "ok":
                answers = {**answers, task_id: metric.answer}
                self.cache.save(answers)

            yield Progress(
                index=index,
                total=total,
                message=f"[{index}/{total}] {task_id}: {metric.status} ({metric.latency_s:.0f}s)",
                metric=metric,
            )

        yield Progress(
            index=total,
            total=total,
            message=f"Run complete: {len(answers)} answers cached in {self.cache.path}.",
            done=True,
        )

    # --- submission ----------------------------------------------------
    def submit(self, username: str, agent_code: str) -> dict[str, Any]:
        """Submit the cached answers. Raises on transport failure."""
        answers = self.cache.load()
        if not answers:
            raise ValueError("Answer cache is empty. Run the agent first.")

        payload = {
            "username": username.strip(),
            "agent_code": agent_code,
            "answers": [
                {"task_id": task_id, "submitted_answer": answer}
                for task_id, answer in answers.items()
            ],
        }
        log.info("submitting %d answers for %s", len(answers), username)
        response = requests.post(
            f"{self.settings.scoring_api_url}/submit", json=payload, timeout=120
        )
        response.raise_for_status()
        return dict(response.json())
