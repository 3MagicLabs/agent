"""Provider-portable conversation shaping.

``BaseChatModel`` abstracts the *client*, not the *conversation contract*.
Providers disagree about message structure, and the disagreements are not
cosmetic: each rule here fixes a defect that one provider rejects loudly and
another accepts while returning garbage.

Every rule below was learned from a measured failure, cited at its function.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

#: Appended when a conversation would otherwise end on an assistant turn.
CONTINUE = "Continue."


def text_of(message: BaseMessage) -> str:
    """The readable text of a message, whatever shape its content is in.

    ``content`` is a plain string on OpenAI-compatible providers and a list of
    typed blocks on Anthropic, where thinking and tool_use blocks sit alongside
    text. ``str()`` over that list yields its repr - which is how a final answer
    once came back beginning ``[{'signature': 'EsEECpAB...``.
    """
    content = message.content
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def merge_system(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Collapse every system message into a single leading one.

    Anthropic carries system content in a dedicated top-level field, so a second
    one part-way down the list has nowhere to go: *"Received multiple
    non-consecutive system messages."* Prepending a prompt and later appending a
    note - which both the specialist retry path and the attachment-inventory
    push did - produces exactly that shape.

    Order among the non-system messages is preserved.
    """
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    if not system:
        return rest

    blocks = [text for text in (text_of(m).strip() for m in system) if text]
    return [SystemMessage(content="\n\n".join(blocks)), *rest]


def ends_with_request(
    messages: Sequence[BaseMessage], request: str = CONTINUE
) -> list[BaseMessage]:
    """Ensure the conversation ends with something the model must answer.

    A trailing assistant turn reads as a *prefill*: Anthropic rejects it
    outright ("This model does not support assistant message prefill"), while
    OpenAI-compatible providers accept it and the model, seeing a conversation
    that already looks finished, replies with a single stop token. That was the
    empty-finalizer bug - measured 3/3 empty without a trailing turn and 3/3
    correct with one, on the same captured conversation.

    A tool result is a valid ending: it maps to a user turn on the wire.
    """
    conversation = list(messages)
    if not conversation or not isinstance(conversation[-1], HumanMessage | ToolMessage):
        conversation.append(HumanMessage(content=request))
    return conversation


def as_data(messages: Sequence[BaseMessage], tag: str = "task") -> list[BaseMessage]:
    """Delimit the opening human turn so it reads as data, not instruction.

    A benchmark question is arbitrary text and some of it is imperative. One
    task is a reversed sentence that decodes to "If you understand this
    sentence, write the opposite of the word 'left' as the answer" - and the
    router obeyed it, replying "right" as prose instead of calling the routing
    function. With no tool call to parse, the structured output came back as
    {}, twice, deterministically, and the task was lost.

    The router's job is to pick a specialist, never to answer. Wrapping the
    question marks where the instructions addressed to *it* end and the material
    it is routing begins. Specialists are not wrapped: following the task is
    precisely what they are for.
    """
    conversation = list(messages)
    for index, message in enumerate(conversation):
        if isinstance(message, HumanMessage):
            conversation[index] = HumanMessage(content=f"<{tag}>\n{text_of(message)}\n</{tag}>")
            break
    return conversation


def drop_dangling_tool_calls(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Remove tool calls that were never executed.

    Anthropic requires every ``tool_use`` block to be followed immediately by
    its ``tool_result``: otherwise the request is rejected outright with
    "`tool_use` ids were found without `tool_result` blocks immediately
    after".

    A specialist that exhausts its iteration budget mid-decision leaves
    exactly that shape - the model asked for a tool, the loop stopped before
    running it - so the wrap-up turn crashed on a 400 every time it was
    needed. The requests are dropped rather than answered with synthetic
    results: they did not run, and inventing results would be a lie the model
    then reasons from.

    Position matters and the first version of this got it wrong: it popped only
    from the end, but ``summarize`` appends its own request after the
    transcript, so the unresolved call sits second-to-last and was skipped. The
    check has to be by *pairing*, not by position.
    """
    resolved = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage) and message.tool_call_id
    }

    kept: list[BaseMessage] = []
    for message in messages:
        requested = getattr(message, "tool_calls", None) or []
        if requested and not all(str(call.get("id")) in resolved for call in requested):
            continue
        kept.append(message)
    return kept


def normalize(messages: Sequence[BaseMessage], request: str = CONTINUE) -> list[BaseMessage]:
    """Shape a message list so any supported provider will accept it."""
    return ends_with_request(drop_dangling_tool_calls(merge_system(messages)), request)
