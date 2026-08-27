"""Shared text shaping for tool output.

Whatever a tool returns is appended to the specialist's transcript and replayed
on every subsequent reasoning turn, so how it is trimmed decides both what the
model can see and what the run costs.
"""

from __future__ import annotations

#: Below this there is no room for a useful head and tail, so trimming the tail
#: is the honest thing to do rather than returning two useless fragments.
_MIN_ELIDE = 80


def elide(text: str, limit: int, note: str = "content elided") -> str:
    """Trim ``text`` to ``limit`` characters, dropping the middle.

    Every truncation in this codebase used to take a head slice, which discards
    the end - and the end is routinely where the answer is: a spreadsheet's
    total is its last row, a program prints its result last, and an article's
    tables sit below its prose. A page that mentions the right topic in its
    first paragraph and answers the question in its last was indistinguishable
    from one that never answered it at all.

    Keeping both ends costs the same tokens and loses only the middle, which is
    the part least likely to be load-bearing.
    """
    if limit <= 0 or len(text) <= limit:
        return text

    dropped = len(text) - limit
    marker = f"\n...[{dropped} characters of {note}]...\n"
    room = limit - len(marker)
    if room < _MIN_ELIDE:
        # Too tight to keep two useful ends. Keep the head, but still say that
        # something was dropped: silent truncation is how a partial result comes
        # to look like a complete one. The note may push slightly past the
        # limit, which is the right trade - the limit bounds cost, not bytes.
        return f"{text[:limit]}\n...[{dropped} characters of {note}]"

    head = room // 2
    return f"{text[:head]}{marker}{text[len(text) - (room - head) :]}"
