"""Sandboxed Python execution.

The E2B SDK is imported lazily so a missing or incompatible package degrades to
a tool message instead of killing the application at import time. Both SDK
generations are supported: ``Sandbox.run_code`` (>=1.0) and the older
``CodeInterpreter.notebook.exec_cell``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from agent.config import get_settings
from agent.obs.logging import get_logger
from agent.tools.registry import ToolSpec, register
from agent.tools.text import elide

log = get_logger("tools.code")


def _load_sandbox_class() -> Any:
    """Return the installed E2B sandbox class, raising ImportError if absent."""
    import e2b_code_interpreter as e2b

    sandbox_cls = getattr(e2b, "Sandbox", None) or getattr(e2b, "CodeInterpreter", None)
    if sandbox_cls is None:
        raise ImportError("e2b_code_interpreter exposes neither Sandbox nor CodeInterpreter")
    return sandbox_cls


def _open_sandbox(sandbox_cls: Any, timeout_s: int) -> Any:
    """Construct a sandbox across E2B SDK generations.

    2.x replaced ``Sandbox(timeout=...)`` with the ``Sandbox.create(...)``
    factory; calling the constructor directly now demands internal connection
    arguments (sandbox_id, envd_version, ...) and fails.
    """
    if hasattr(sandbox_cls, "create"):
        return sandbox_cls.create(timeout=timeout_s)
    return sandbox_cls(timeout=timeout_s)


def _close_sandbox(sandbox: Any) -> None:
    """Release a sandbox; a leaked one keeps billing against the quota."""
    killer = getattr(sandbox, "kill", None)
    if killer is None:
        return
    try:
        killer()
    except Exception as exc:  # noqa: BLE001 - cleanup must never mask the result
        log.warning("Could not release sandbox: %s", exc)


def _execute(sandbox: Any, code: str, timeout_s: float | None = None) -> Any:
    """Run a cell across E2B SDK generations.

    ``timeout`` moved from the sandbox constructor to ``run_code`` in the 2.x
    SDK; passing it to ``Sandbox(...)`` raises TypeError.
    """
    if hasattr(sandbox, "run_code"):
        return sandbox.run_code(code, timeout=timeout_s)
    if hasattr(sandbox, "notebook"):
        return sandbox.notebook.exec_cell(code)
    raise AttributeError("Unsupported E2B sandbox API: no run_code or notebook")


def _render(execution: Any, limit: int) -> str:
    if getattr(execution, "error", None):
        error = execution.error
        traceback = elide(error.traceback or "", limit, note="traceback elided")
        return f"Execution Error: {error.name}: {error.value}\n{traceback}"

    parts: list[str] = []
    logs = getattr(execution, "logs", None)
    if logs is not None:
        if getattr(logs, "stdout", None):
            parts.append("\n".join(logs.stdout))
        if getattr(logs, "stderr", None):
            parts.append("[stderr]\n" + "\n".join(logs.stderr))

    for result in getattr(execution, "results", []) or []:
        if getattr(result, "text", None):
            parts.append(f"[Result]: {result.text}")

    output = "\n".join(part for part in parts if part).strip()
    if not output:
        return "Executed successfully with no output. Did you forget to print()?"
    return elide(output, limit, note="output elided")


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code in a secure sandbox and return its output.
    Use print() to surface results. You may pip install libraries.
    """
    log.info("python_repl: executing %d chars", len(code))
    settings = get_settings()

    if not settings.has_sandbox:
        return (
            "Code execution is unavailable: E2B_API_KEY is not configured. "
            "Reason the answer out directly instead of retrying this tool."
        )

    try:
        sandbox_cls = _load_sandbox_class()
    except ImportError as exc:
        log.error("E2B SDK unavailable: %s", exc)
        return f"Code execution is unavailable: {exc}"

    sandbox = None
    try:
        sandbox = _open_sandbox(sandbox_cls, int(settings.sandbox_timeout_s))
        execution = _execute(sandbox, code, timeout_s=settings.sandbox_timeout_s)
        return _render(execution, settings.max_code_output_chars)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a message
        log.error("Sandbox execution failed: %s", exc)
        return f"System Error connecting to sandbox: {exc}"
    finally:
        if sandbox is not None:
            _close_sandbox(sandbox)


register(
    ToolSpec(
        name="python_repl",
        capability="code",
        factory=lambda: python_repl,
        requires=("has_sandbox",),
    )
)
