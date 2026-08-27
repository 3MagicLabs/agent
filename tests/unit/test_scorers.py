"""Answer normalization — part of the system under test, since the benchmark
grades by exact match."""

from __future__ import annotations

import pytest

from agent.config import Settings
from agent.eval import scorers
from agent.eval.scorers import exact_match, normalize, score

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FINAL ANSWER: 42", "42"),
        ("Answer: Paris", "paris"),
        ("  Paris  ", "paris"),
        ("1,234", "1234"),
        # Was asserted as "123450" - the decimal point stripped before the
        # value was read as a number, which is precisely the bug that made
        # the grader score 3.14 equal to 314.
        ("$1,234.50", "1234.5"),
        ("The Eiffel Tower.", "the eiffel tower"),
        ("a, b, c", "a, b, c"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("FINAL ANSWER: 3", "3"),
        ("Paris", "paris"),
        ("1,000", "1000"),
        ("Cheese.", "cheese"),
    ],
)
def test_exact_match_accepts_formatting_differences(predicted, gold):
    assert exact_match(predicted, gold)


@pytest.mark.parametrize(("predicted", "gold"), [("4", "3"), ("London", "Paris"), ("", "x")])
def test_exact_match_rejects_wrong_answers(predicted, gold):
    assert not exact_match(predicted, gold)


class TestScore:
    def test_counts_only_graded_tasks(self):
        report = score({"a": "1", "b": "2", "c": "3"}, {"a": "1", "b": "9"})

        assert report.graded == 2
        assert report.correct == 1
        assert report.accuracy == 0.5

    def test_empty_gold_is_not_a_division_error(self):
        assert score({"a": "1"}, {}).accuracy == 0.0

    def test_string_form_is_human_readable(self):
        assert str(score({"a": "1"}, {"a": "1"})) == "1/1 (100%)"


class TestGoldAnswers:
    """Loading reference answers from the GAIA validation split."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        scorers._GOLD.clear()
        yield
        scorers._GOLD.clear()

    def test_a_missing_token_is_an_error_not_an_empty_result(self, monkeypatch):
        """An empty gold set scores every run 0/0, which reads as a result."""
        monkeypatch.setattr(scorers, "get_settings", lambda: Settings(hf_token=""))

        with pytest.raises(scorers.GoldUnavailableError, match="HF_TOKEN"):
            scorers.gold_answers()

    def test_a_fetch_failure_raises(self, monkeypatch):
        monkeypatch.setattr(scorers, "get_settings", lambda: Settings(hf_token="t"))
        monkeypatch.setattr(
            scorers.requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
        )

        with pytest.raises(scorers.GoldUnavailableError, match="boom"):
            scorers.gold_answers()

    def test_a_failure_is_never_memoised(self, monkeypatch):
        """One transient error must not disable grading for the process lifetime."""
        monkeypatch.setattr(scorers, "get_settings", lambda: Settings(hf_token="t"))
        calls: list[int] = []

        def flaky(*_args, **_kwargs):
            calls.append(1)
            raise OSError("transient")

        monkeypatch.setattr(scorers.requests, "get", flaky)

        for _ in range(2):
            with pytest.raises(scorers.GoldUnavailableError):
                scorers.gold_answers()

        assert len(calls) == 2, "a failed fetch was cached"

    def test_a_successful_fetch_is_cached(self, monkeypatch):
        monkeypatch.setattr(scorers, "get_settings", lambda: Settings(hf_token="t"))
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame(
            [
                {"task_id": "abc", "Final answer": "FunkMonk"},
                {"task_id": "def", "Final answer": "3"},
            ]
        )
        calls: list[int] = []

        class Response:
            content = b""

            def raise_for_status(self) -> None:
                return None

        def fetch(*_args, **_kwargs):
            calls.append(1)
            return Response()

        monkeypatch.setattr(scorers.requests, "get", fetch)
        monkeypatch.setattr(pd, "read_parquet", lambda _buffer: frame)

        assert scorers.gold_answers() == {"abc": "FunkMonk", "def": "3"}
        assert scorers.gold_answers() == {"abc": "FunkMonk", "def": "3"}
        assert len(calls) == 1, "a successful fetch was not cached"
