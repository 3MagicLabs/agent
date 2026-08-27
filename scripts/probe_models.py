"""Does any available model accept the reversed text?

The server-side ``fallbacks`` parameter is not supported on claude-sonnet-5 -
it is an Opus/Fable-tier feature - so the API cannot retry a declined request
for us. A client-side fallback needs no special parameter, though: on a refusal,
re-issue that one call to a different model. That only works if some model
accepts the input.

Refusals are classifier decisions and classifiers differ between models, so this
is a real question rather than a formality. It is also cheap: a decline before
any output is not billed, and each probe caps output at 16 tokens.

Run:
    cd ~/agentsCourse/Final_Assignment_Template
    set -a; source .env; set +a
    ~/agentsCourse/venv/bin/python scripts/probe_models.py

Reading it: any model answering the HARMLESS control is a viable fallback
target, and the cheapest one wins - it handles one call per refused task, not
the workload.
"""

from __future__ import annotations

import os

import anthropic

#: The benchmark task, and an innocuous question under the same obfuscation.
#: The control is the cleaner signal: content is already known to be irrelevant,
#: so a model refusing "the capital of France" backwards is refusing the
#: encoding itself.
QUESTION = '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'
HARMLESS = "?ecnarF fo latipac eht si tahW"

#: input $/1M, output $/1M - so a viable target can be chosen on price.
CANDIDATES = (
    ("claude-haiku-4-5", 1.00, 5.00),
    ("claude-sonnet-4-6", 3.00, 15.00),
    ("claude-sonnet-5", 2.00, 10.00),
    ("claude-opus-5", 5.00, 25.00),
)


def probe(client: anthropic.Anthropic, model: str, text: str) -> str:
    try:
        reply = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": text}],
        )
    except Exception as exc:  # noqa: BLE001 - an unavailable model is a result
        return f"ERROR {type(exc).__name__}: {str(exc)[:70]}"

    if reply.stop_reason == "refusal":
        details = getattr(reply, "stop_details", None)
        return f"refused ({getattr(details, 'category', '?')})"
    text_out = next((b.text for b in reply.content if getattr(b, "type", "") == "text"), "")
    return f"ANSWERED {text_out.strip()[:40]!r}"


def probe_knob(client: anthropic.Anthropic, model: str, **kwargs: object) -> str:
    """The control text with one generation parameter varied."""
    try:
        reply = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": HARMLESS}],
            **kwargs,  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001 - a rejected parameter is a result
        return f"ERROR {type(exc).__name__}: {str(exc)[:60]}"
    return "refused" if reply.stop_reason == "refusal" else "ANSWERED"


def knobs(client: anthropic.Anthropic) -> None:
    """Can a generation parameter change a decision made before generation?

    Expected no: every refusal reports output_tokens 0 and reasoning 0, so
    nothing was generated for effort or temperature to act on. Measured
    anyway - reasoning about this task has been wrong three times.
    """
    print("\n=== does a generation parameter move it? (harmless control) ===")
    for label, model, kwargs in (
        ("effort low", "claude-sonnet-5", {"output_config": {"effort": "low"}}),
        ("effort max", "claude-sonnet-5", {"output_config": {"effort": "max"}}),
        ("temperature 0", "claude-haiku-4-5", {"temperature": 0.0}),
        ("temperature 1", "claude-haiku-4-5", {"temperature": 1.0}),
    ):
        print(f"  {label:<16} {model:<20} {probe_knob(client, model, **kwargs)}")


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not in this shell - run with `set -a; source .env; set +a`")
        return 1

    client = anthropic.Anthropic()
    print(f"{'model':<20} {'$/1M in':>8}  {'benchmark task':<34} harmless control")
    print("-" * 100)
    for model, price_in, _ in CANDIDATES:
        on_task = probe(client, model, QUESTION)
        on_control = probe(client, model, HARMLESS)
        print(f"{model:<20} {price_in:>8.2f}  {on_task:<34} {on_control}")

    knobs(client)

    print(
        "\nAny model answering the control is a viable client-side fallback: on a\n"
        "refusal, that one call is re-issued there and the rest of the run is\n"
        "unaffected. If every model refuses, the encoding itself is universally\n"
        "declined and normalising the input before it is sent is the only option\n"
        "left - at the cost of doing in Python the character-level work the task\n"
        "exists to test.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
