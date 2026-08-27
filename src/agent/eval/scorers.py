"""Answer normalization and scoring.

The benchmark grades by exact match after normalization, so the normalizer is
part of the system under test: a correct answer formatted wrongly scores zero.

Reference answers come from the GAIA validation split, which ships them
alongside the attachments the tools already download. Grading is therefore
local, instant and free - the alternative is submitting to the leaderboard and
learning a single percentage with no indication of which tasks failed.
"""

from __future__ import annotations

import io
import re
import string
from dataclasses import dataclass

import requests

from agent.config import get_settings
from agent.obs.logging import get_logger
from agent.tools.files import GAIA_DATASET, GAIA_SPLIT

log = get_logger("eval.scorers")


class GoldUnavailableError(RuntimeError):
    """Reference answers could not be loaded.

    Raised rather than returning an empty mapping. An empty gold set scores
    every run 0/0, which reads as a result instead of a failure to obtain one -
    the same laundering of an error into a plausible output that this codebase
    exists to remove.
    """


#: Populated only on success. A failed fetch must not be memoised: one transient
#: error would otherwise convince the process for its whole lifetime that GAIA
#: has no reference answers.
_GOLD: dict[int, dict[str, str]] = {}


def gold_answers(level: int = 1) -> dict[str, str]:
    """task_id -> reference answer for one GAIA validation level.

    Requires ``HF_TOKEN``: the dataset is gated. Reading parquet needs pandas,
    which is an optional extra, so it is imported lazily and its absence is
    reported as an actionable message rather than an ImportError traceback.
    """
    if level in _GOLD:
        return _GOLD[level]

    settings = get_settings()
    if not settings.hf_token:
        raise GoldUnavailableError(
            "HF_TOKEN is not set. The GAIA dataset is gated; reference answers "
            "cannot be fetched without it."
        )

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on the install extras
        raise GoldUnavailableError(
            "Reading the reference answers needs pandas. Install it with "
            "`pip install -e '.[app]'`."
        ) from exc

    name = f"metadata.level{level}.parquet"
    url = f"https://huggingface.co/datasets/{GAIA_DATASET}/resolve/main/{GAIA_SPLIT}/{name}"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            timeout=settings.scrape_timeout_s,
        )
        response.raise_for_status()
        frame = pd.read_parquet(io.BytesIO(response.content))
    except Exception as exc:
        raise GoldUnavailableError(f"Could not fetch {name}: {exc}") from exc

    answers = {
        str(row["task_id"]): str(row["Final answer"])
        for _, row in frame.iterrows()
        if row.get("task_id") and row.get("Final answer") is not None
    }
    if not answers:
        raise GoldUnavailableError(f"{name} contained no reference answers.")

    log.info("GAIA level %d reference answers: %d tasks", level, len(answers))
    _GOLD[level] = answers
    return answers


#: Preambles models habitually emit despite being told not to.
_PREFIX = re.compile(r"^\s*(final\s+answer\s*:|answer\s*:)\s*", re.IGNORECASE)
#: Comma is kept so a list survives splitting; the decimal point and minus sign
#: are removed only from text that is NOT a number - see below.
_PUNCTUATION = str.maketrans("", "", string.punctuation.replace(",", ""))
#: Stripped before a numeric reading: thousands separators, currency, percent.
_NUMERIC_NOISE = str.maketrans("", "", ",$%  ")


def _as_number(text: str) -> str | None:
    """``text`` as a canonical number, or None when it is not one."""
    try:
        return f"{float(text.translate(_NUMERIC_NOISE)):g}"
    except (ValueError, OverflowError):
        return None


def normalize(answer: str) -> str:
    """Canonical form used for comparison.

    Numbers are read *before* punctuation is stripped, and this ordering is the
    whole point. Stripping first deleted the decimal point and the minus sign,
    so the grader scored "3.14" equal to "314" and "-5" equal to "5" - crediting
    wrong answers - while judging a correct "89706" unequal to the reference
    "89706.00", because one had been flattened and the other had not.

    Two of the 53 level-1 reference answers are decimals and fifteen are
    integers, so this was not hypothetical: it is the instrument that produced
    every score in this project until it was checked.
    """
    text = _PREFIX.sub("", str(answer).strip()).strip()

    number = _as_number(text)
    if number is not None:
        return number

    return re.sub(r"\s+", " ", text.lower().translate(_PUNCTUATION)).strip()


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
