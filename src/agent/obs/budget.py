"""Spend accounting for a run.

The project moved from a free provider with a hard daily token cap to a paid one
with no cap at all. The old ceiling was involuntary and absolute; the new one has
to be built, because nothing else stops a retry loop from spending real money.

Two ceilings, because they catch different failures: a per-task ceiling catches
one runaway task, a per-run ceiling catches many slightly-too-expensive ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from agent.obs.logging import get_logger

log = get_logger("obs.budget")

#: model -> (input $/1M tokens, output $/1M tokens). A prefix match, so dated
#: snapshots of a model inherit its rate. Unknown models cost nothing here,
#: which keeps an unpriced provider from halting a run - the wall-clock budget
#: still bounds it, and a wrong price is worse than no price.
RATES: Mapping[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o-mini": (0.15, 0.60),
}

_PER_MILLION = 1_000_000.0


def rate_for(model: str) -> tuple[float, float] | None:
    """Input and output rates for a model, or None when it is not priced."""
    for name, rates in RATES.items():
        if model.startswith(name):
            return rates
    return None


def cost_of(tokens: Mapping[str, int], model: str) -> float:
    """Dollar cost of one task's token usage, or 0.0 for an unpriced model."""
    rates = rate_for(model)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    inputs = int(tokens.get("input_tokens", 0))
    outputs = int(tokens.get("output_tokens", 0))
    return (inputs * input_rate + outputs * output_rate) / _PER_MILLION


@dataclass(frozen=True, slots=True)
class Budget:
    """What a run may spend, and what it has spent.

    Immutable: ``charge`` returns a new Budget rather than mutating this one, so
    a caller can compute a prospective total without committing to it.
    """

    max_run_usd: float = 0.0
    max_task_usd: float = 0.0
    spent_usd: float = 0.0

    @property
    def enabled(self) -> bool:
        """False when neither ceiling is configured, which disables accounting."""
        return self.max_run_usd > 0 or self.max_task_usd > 0

    def charge(self, amount: float) -> Budget:
        return replace(self, spent_usd=self.spent_usd + amount)

    def task_overspend(self, amount: float) -> str:
        """Why one task's cost is unacceptable, or "" when it is fine."""
        if self.max_task_usd > 0 and amount > self.max_task_usd:
            return (
                f"one task cost ${amount:.4f}, over the "
                f"${self.max_task_usd:.2f} per-task ceiling"
            )
        return ""

    def run_overspend(self) -> str:
        """Why the run may not continue, or "" when it may."""
        if self.max_run_usd > 0 and self.spent_usd >= self.max_run_usd:
            return f"run cost ${self.spent_usd:.4f}, at the ${self.max_run_usd:.2f} ceiling"
        return ""
