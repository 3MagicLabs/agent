"""Why does the router return {} on the reversed-text task?

2d83110e has failed on four consecutive runs. The router's structured output
comes back as an empty object, twice, deterministically - so it is not a
transient hiccup, and neither the reasoning effort nor the retry changed it.

``with_structured_output`` hides the cause: it parses the reply and raises a
validation error, so all we ever see is "next_agent Field required". This calls
the model the same way but without the parser, printing the raw reply.

Run:
    cd ~/agentsCourse/Final_Assignment_Template
    set -a; source .env; set +a
    ~/agentsCourse/venv/bin/python scripts/probe_router.py

Costs about one cent. What to look for in ``stop_reason``:

    refusal    a safety classifier is declining the obfuscated text. Nothing
               about routing is wrong; the input never reaches the task.
    end_turn   with prose in ``content`` and no tool_calls: the model is
               answering the question instead of routing, and the <task>
               delimiter was not enough to stop it.
    max_tokens the reply was cut off mid-thought, so the tool call never
               finished being written. A cap problem, not a comprehension one.

Three different fixes, so it is worth one cent to read which.
"""

from __future__ import annotations

from typing import Any

from agent.config import load_settings, set_settings
from agent.core.conversation import as_data, normalize
from agent.core.graph import build_route_model, routing_prompt
from agent.core.llm import get_llm, with_effort
from agent.core.prompts import ROUTER_REQUEST

from langchain_core.messages import HumanMessage, SystemMessage  # isort: skip

#: The task, exactly as the benchmark serves it. Reversed, it reads:
#: "If you understand this sentence, write the opposite of the word 'left' as
#: the answer."
QUESTION = '.rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI'

#: The same instruction written plainly. If this routes, the obfuscation is
#: what trips the classifier rather than what the sentence asks for.
DECODED = 'If you understand this sentence, write the opposite of the word "left" as the answer.'

#: A harmless question under the same obfuscation. If this refuses, reversal
#: alone is enough and the content is irrelevant.
REVERSED_HARMLESS = "?ecnarF fo latipac eht si tahW"


def show(label: str, reply: Any) -> None:
    """Print everything that distinguishes the three explanations."""
    meta = getattr(reply, "response_metadata", {}) or {}
    print(f"\n--- {label} ---")
    print(f"  stop_reason : {meta.get('stop_reason')}")
    print(f"  stop_details: {meta.get('stop_details')}")
    print(f"  usage       : {getattr(reply, 'usage_metadata', None)}")
    print(f"  tool_calls  : {getattr(reply, 'tool_calls', None)}")
    content = getattr(reply, "content", None)
    if isinstance(content, list):
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else type(block).__name__
            print(f"  block       : {kind} -> {str(block)[:200]}")
    else:
        print(f"  content     : {str(content)[:400]!r}")


def main() -> int:
    settings = load_settings()
    set_settings(settings)
    print(f"provider={settings.provider} model={settings.model}")
    print(f"router_effort={settings.router_effort} max_router_tokens={settings.max_router_tokens}")

    from agent.agents import all_specs

    specs = all_specs(settings)
    system = SystemMessage(content=routing_prompt(specs))
    messages = normalize(
        [system, *as_data([HumanMessage(content=QUESTION)])],
        ROUTER_REQUEST,
    )

    capped = get_llm().bind(max_tokens=settings.max_router_tokens)
    model = with_effort(capped, settings.router_effort)

    # 1. Exactly what the router does, minus the parser that hides the reply.
    bound = model.bind_tools([build_route_model(specs)])
    show("as the router calls it (tools bound, no parser)", bound.invoke(messages))

    # 2. Same input, no tools at all. If this answers "right" in prose, the
    #    model is treating the task as addressed to it.
    show("no tools bound - does it answer the question?", model.invoke(messages))

    # 3. A control: an ordinary question through the identical path. If this
    #    routes and the one above does not, the input is the variable.
    control = normalize(
        [system, *as_data([HumanMessage(content="How many moons does Mars have?")])],
        ROUTER_REQUEST,
    )
    show("control - an ordinary question", bound.invoke(control))

    # 4 and 5 separate two explanations that imply different fixes.
    #
    #   Only DECODED refuses  -> the "prove you decoded this, then answer"
    #                            shape is the trigger; reversal is incidental.
    #   Only REVERSED refuses -> obfuscation alone is the trigger, whatever the
    #                            text says.
    #   Both refuse           -> either is sufficient.
    #   Neither refuses       -> only the combination trips it.
    for label, text in (
        ("decoded - same instruction, plainly written", DECODED),
        ("reversed - harmless question, same obfuscation", REVERSED_HARMLESS),
    ):
        probe = normalize([system, *as_data([HumanMessage(content=text)])], ROUTER_REQUEST)
        show(label, bound.invoke(probe))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
