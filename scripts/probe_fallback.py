"""Can a server-side fallback recover a refused request?

A classifier declines reversed text before the model runs - measured, and
content-independent: "What is the capital of France?" written backwards is
refused exactly like the benchmark task. Nothing the agent does can route
around it, because every component is handed the same string.

Anthropic ships ``fallbacks`` for this: on a policy decline the API re-runs the
request on a different model inside the same call, so the caller gets an answer
rather than a refusal. The refusal error text has recommended it on every
occurrence.

This checks three things, cheapest first:

  1. The raw SDK, which is where the parameter is documented. If this refuses
     too, fallbacks cannot help and the question is closed.
  2. Whether ChatAnthropic passes it through. LangChain was not built with this
     parameter in mind; it may reach the wire, be dropped silently, or fail to
     use the beta endpoint at all.
  3. What the refusal chain reports, so a fallback that fires is visible in a
     trace rather than silent.

Run:
    cd ~/agentsCourse/Final_Assignment_Template
    set -a; source .env; set +a
    ~/agentsCourse/venv/bin/python scripts/probe_fallback.py

Costs a few cents. A decline before any output is not billed; only a rescue is.
"""

from __future__ import annotations

import os
from typing import Any

#: The benchmark task, and a control proving content is irrelevant: an entirely
#: innocuous question is refused under the same obfuscation.
QUESTION = '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'
HARMLESS = "?ecnarF fo latipac eht si tahW"

BETA = "server-side-fallback-2026-07-01"


def report(label: str, reply: Any) -> None:
    meta = getattr(reply, "response_metadata", None)
    stop = getattr(reply, "stop_reason", None) or (meta or {}).get("stop_reason")
    print(f"\n--- {label} ---")
    print(f"  stop_reason : {stop}")

    content = getattr(reply, "content", None)
    blocks = content if isinstance(content, list) else []
    for block in blocks:
        kind = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )
        if kind == "fallback":
            print(f"  FALLBACK    : {block}")
        elif kind == "text":
            text = getattr(block, "text", None) or block.get("text", "")
            print(f"  text        : {str(text)[:200]!r}")
    if not blocks:
        print(f"  content     : {str(content)[:200]!r}")

    usage = getattr(reply, "usage", None)
    if usage is not None:
        served = [
            entry
            for entry in (getattr(usage, "iterations", None) or [])
            if getattr(entry, "type", "") == "fallback_message"
        ]
        print(f"  fallback ran: {bool(served)}")


def raw_sdk() -> None:
    """The parameter as documented, through the SDK that defines it."""
    import anthropic

    client = anthropic.Anthropic()
    for label, text in (("task", QUESTION), ("harmless control", HARMLESS)):
        print(f"\n=== raw SDK, fallbacks='default' ({label}) ===")
        try:
            reply = client.beta.messages.create(
                model="claude-sonnet-5",
                max_tokens=256,
                betas=[BETA],
                fallbacks="default",
                messages=[{"role": "user", "content": text}],
            )
        except Exception as exc:  # noqa: BLE001 - the answer either way
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        report(label, reply)


def through_langchain() -> None:
    """Whether ChatAnthropic carries the parameter to the wire."""
    from langchain_anthropic import ChatAnthropic

    print("\n=== via ChatAnthropic (betas + model_kwargs) ===")
    try:
        model = ChatAnthropic(
            model_name="claude-sonnet-5",
            max_tokens_to_sample=256,
            betas=[BETA],
            model_kwargs={"fallbacks": "default"},
        )
        report("langchain", model.invoke(QUESTION))
    except Exception as exc:  # noqa: BLE001 - a rejection here is the finding
        print(f"  FAILED: {type(exc).__name__}: {str(exc)[:300]}")


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not in this shell - run with `set -a; source .env; set +a`")
        return 1

    raw_sdk()
    through_langchain()

    print(
        "\nReading it:\n"
        "  stop_reason 'refusal' everywhere -> the fallback chain also declined;\n"
        "    fallbacks cannot recover this and normalising the input is the only\n"
        "    remaining option.\n"
        "  raw SDK answers, LangChain refuses -> the parameter works but does not\n"
        "    survive the wrapper; that one call needs the raw client.\n"
        "  both answer -> wire it into core/llm.py and the task is recoverable.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
