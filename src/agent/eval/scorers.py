"""Answer normalization and scoring.

The benchmark grades by exact match after normalization, so the normalizer is
part of the system under test: a correct answer formatted wrongly scores zero.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

#: Preambles models habitually emit despite being told not to.
_PREFIX = re.compile(r"^\s*(final\s+answer\s*:|answer\s*:)\s*", re.IGNORECASE)
_PUNCTUATION = str.maketrans("", "", string.punctuation.replace(",", ""))
_NUMBER = re.compile(r"^-?\d[\d,]*\.?\d*$")


def normalize(answer: str) -> str:
    """Canonical form used for comparison."""
    text = _PREFIX.sub("", str(answer).strip()).strip().lower()
    text = text.translate(_PUNCTUATION)
    text = re.sub(r"\s+", " ", text).strip()
    if _NUMBER.match(text.replace(" ", "")):
        text = text.replace(",", "").replace(" ", "")
    return text


def exact_match(predicted: str, expected: str) -> bool:
    """True when the two answers agree after normalization."""
    return normalize(predicted) == normalize(expected)


@dataclass(frozen=True, slots=True)
class ScoreReport:
    """Aggregate accuracy over a graded run."""

    correct: int
    graded: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.graded if self.graded else 0.0

    def __str__(self) -> str:
        return f"{self.correct}/{self.graded} ({100 * self.accuracy:.0f}%)"


def score(predictions: dict[str, str], gold: dict[str, str]) -> ScoreReport:
    """Score predictions against gold answers, keyed by task_id."""
    graded = [(task_id, answer) for task_id, answer in predictions.items() if task_id in gold]
    correct = sum(1 for task_id, answer in graded if exact_match(answer, gold[task_id]))
    return ScoreReport(correct=correct, graded=len(graded))
