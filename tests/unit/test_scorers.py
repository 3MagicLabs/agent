"""Answer normalization — part of the system under test, since the benchmark
grades by exact match."""

from __future__ import annotations

import pytest

from agent.eval.scorers import exact_match, normalize, score

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FINAL ANSWER: 42", "42"),
        ("Answer: Paris", "paris"),
        ("  Paris  ", "paris"),
        ("1,234", "1234"),
        ("$1,234.50", "123450"),
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
