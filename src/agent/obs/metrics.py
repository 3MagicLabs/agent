"""Per-task metrics: an immutable record plus an append-only JSONL sink."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.config import get_settings
from agent.obs.logging import get_logger

log = get_logger("obs.metrics")

Status = str  # "ok" | "error" | "timeout"


@dataclass(frozen=True, slots=True)
class TaskMetric:
    """One evaluated task. Immutable: build a new one to change anything."""

    task_id: str
    question: str
    answer: str = ""
    status: Status = "ok"
    error: str = ""
    latency_s: float = 0.0
    tokens: Mapping[str, int] = field(default_factory=dict)
    supervisor_steps: int = 0
    model: str = ""

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


class MetricsRecorder:
    """Append-only JSONL sink with an in-memory view of the current run."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else get_settings().metrics_file
        self._rows: tuple[TaskMetric, ...] = ()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def rows(self) -> tuple[TaskMetric, ...]:
        return self._rows

    def record(self, metric: TaskMetric) -> TaskMetric:
        self._rows = (*self._rows, metric)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric.as_row(), ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Could not persist metric for %s: %s", metric.task_id, exc)
        return metric

    def summary(self) -> dict[str, Any]:
        if not self._rows:
            return {"count": 0}
        latencies = sorted(row.latency_s for row in self._rows)
        return {
            "count": len(self._rows),
            "ok": sum(1 for row in self._rows if row.status == "ok"),
            "errors": sum(1 for row in self._rows if row.status == "error"),
            "timeouts": sum(1 for row in self._rows if row.status == "timeout"),
            "total_latency_s": round(sum(latencies), 1),
            "median_latency_s": round(latencies[len(latencies) // 2], 1),
            "max_latency_s": round(latencies[-1], 1),
            "total_tokens": sum(int(row.tokens.get("total_tokens", 0)) for row in self._rows),
        }


def read_metrics(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load a metrics JSONL file, skipping unreadable lines."""
    target = Path(path) if path else get_settings().metrics_file
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            log.warning("Skipping malformed metrics line in %s", target)
    return rows
