"""LangSmith tracing setup and per-run token accounting."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from agent.config import Settings, get_settings
from agent.obs.logging import get_logger

log = get_logger("obs.tracing")

#: LangChain renamed these variables; both families are still read in the wild,
#: so we set both rather than betting on the installed version.
_TRACING_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
_KEY_VARS = ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
_PROJECT_VARS = ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")


def configure_tracing(settings: Settings | None = None) -> bool:
    """Enable LangSmith tracing when an API key is configured.

    Returns True when tracing is active.
    """
    resolved = settings or get_settings()
    if not resolved.langsmith_api_key:
        log.info("LangSmith tracing disabled (no LANGSMITH_API_KEY)")
        return False

    for name in _KEY_VARS:
        os.environ[name] = resolved.langsmith_api_key
    for name in _TRACING_VARS:
        os.environ[name] = "true"
    for name in _PROJECT_VARS:
        os.environ[name] = resolved.langsmith_project

    log.info("LangSmith tracing enabled -> project %s", resolved.langsmith_project)
    return True


def usage_callback() -> Any | None:
    """Return a token-usage callback handler, or None on older langchain-core."""
    try:
        from langchain_core.callbacks import UsageMetadataCallbackHandler
    except ImportError:  # pragma: no cover - depends on installed version
        return None
    return UsageMetadataCallbackHandler()


def total_tokens(handler: Any | None) -> dict[str, int]:
    """Flatten a usage handler's per-model counts into one total."""
    if handler is None:
        return {}
    usage: Mapping[str, Mapping[str, Any]] = getattr(handler, "usage_metadata", {}) or {}
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for per_model in usage.values():
        for key in totals:
            totals[key] += int(per_model.get(key, 0) or 0)
    return totals


def trace_config(task_id: str, callbacks: list[Any] | None = None) -> dict[str, Any]:
    """Build the LangGraph run config that ties a trace back to a task."""
    settings = get_settings()
    config: dict[str, Any] = {
        "recursion_limit": settings.recursion_limit,
        "run_name": f"task:{task_id}",
        "metadata": {"task_id": task_id, "model": settings.model, "provider": settings.provider},
        "tags": ["agent", "supervisor"],
    }
    if callbacks:
        config["callbacks"] = callbacks
    return config
