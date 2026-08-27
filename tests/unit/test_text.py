"""Trimming tool output so the end survives."""

from __future__ import annotations

import pytest

from agent.tools.text import elide

pytestmark = pytest.mark.unit


class TestElide:
    def test_short_text_is_untouched(self):
        assert elide("abc", 100) == "abc"

    def test_text_exactly_at_the_limit_is_untouched(self):
        assert elide("x" * 100, 100) == "x" * 100

    def test_the_result_respects_the_limit(self):
        assert len(elide("x" * 5000, 500)) <= 500

    def test_both_ends_survive(self):
        """A head slice discards the end, which is where the answer usually is."""
        text = "TOTAL_IS_AT_THE_START" + "x" * 5000 + "TOTAL_IS_AT_THE_END"

        trimmed = elide(text, 400)

        assert trimmed.startswith("TOTAL_IS_AT_THE_START")
        assert trimmed.endswith("TOTAL_IS_AT_THE_END")

    def test_it_says_how_much_was_dropped(self):
        trimmed = elide("x" * 5000, 400)

        assert "characters of content elided" in trimmed

    def test_the_note_is_caller_supplied(self):
        assert "output elided" in elide("x" * 5000, 400, note="output elided")

    def test_a_limit_too_small_to_split_keeps_the_head_and_still_says_so(self):
        """Two useless fragments are worse than one readable one - but
        silent truncation is worse than both: a partial result then looks
        complete."""
        trimmed = elide("x" * 500, 20)

        assert trimmed.startswith("x" * 20)
        assert "480 characters of content elided" in trimmed

    def test_a_zero_limit_disables_trimming(self):
        assert elide("x" * 500, 0) == "x" * 500

    def test_a_real_table_keeps_its_total_row(self):
        rows = "\n".join(f"item-{i},{i}" for i in range(2000))
        table = f"name,amount\n{rows}\nTOTAL,89706.00"

        trimmed = elide(table, 600)

        assert "TOTAL,89706.00" in trimmed
        assert trimmed.startswith("name,amount")
