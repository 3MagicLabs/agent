"""Dollar ceilings on a run."""

from __future__ import annotations

import pytest

from agent.obs.budget import Budget, cost_of, rate_for

pytestmark = pytest.mark.unit


class TestRates:
    def test_a_known_model_is_priced(self):
        assert rate_for("claude-sonnet-5") == (2.00, 10.00)

    def test_a_dated_snapshot_inherits_its_family_rate(self):
        assert rate_for("claude-sonnet-5-20260101") == (2.00, 10.00)

    def test_an_unknown_model_is_unpriced(self):
        assert rate_for("some-local-llama") is None


class TestCostOf:
    def test_input_and_output_are_priced_separately(self):
        tokens = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}

        assert cost_of(tokens, "claude-sonnet-5") == pytest.approx(12.00)

    def test_a_real_measured_task(self):
        """The reference task: 17,704 in / 1,157 out on Sonnet 5."""
        tokens = {"input_tokens": 17_704, "output_tokens": 1_157}

        assert cost_of(tokens, "claude-sonnet-5") == pytest.approx(0.0470, abs=0.001)

    def test_an_unpriced_model_costs_nothing(self):
        """A wrong price is worse than no price; the clock budget still bounds it."""
        tokens = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}

        assert cost_of(tokens, "some-local-llama") == 0.0

    def test_missing_token_counts_are_free(self):
        assert cost_of({}, "claude-sonnet-5") == 0.0


class TestBudget:
    def test_charging_returns_a_new_budget(self):
        """Immutable, so a caller can weigh a prospective total before committing."""
        budget = Budget(max_run_usd=1.0)

        charged = budget.charge(0.25)

        assert budget.spent_usd == 0.0
        assert charged.spent_usd == 0.25

    def test_a_run_under_its_ceiling_may_continue(self):
        assert Budget(max_run_usd=1.0).charge(0.99).run_overspend() == ""

    def test_a_run_at_its_ceiling_stops(self):
        assert "at the $1.00 ceiling" in Budget(max_run_usd=1.0).charge(1.0).run_overspend()

    def test_one_expensive_task_is_caught_on_its_own(self):
        """A per-run ceiling alone would let a single runaway through."""
        budget = Budget(max_run_usd=100.0, max_task_usd=0.50)

        assert "per-task ceiling" in budget.task_overspend(0.75)

    def test_an_ordinary_task_passes(self):
        assert Budget(max_run_usd=100.0, max_task_usd=0.50).task_overspend(0.047) == ""

    def test_zero_disables_a_ceiling(self):
        budget = Budget(max_run_usd=0.0, max_task_usd=0.0).charge(1000.0)

        assert budget.enabled is False
        assert budget.run_overspend() == ""
        assert budget.task_overspend(1000.0) == ""

    def test_either_ceiling_enables_accounting(self):
        assert Budget(max_task_usd=0.5).enabled is True
        assert Budget(max_run_usd=5.0).enabled is True
