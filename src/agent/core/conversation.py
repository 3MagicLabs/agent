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


def normalize(messages: Sequence[BaseMessage], request: str = CONTINUE) -> list[BaseMessage]:
    """Shape a message list so any supported provider will accept it."""
    return ends_with_request(merge_system(messages), request)
