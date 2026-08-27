"""Chat model factory.

One cached client per process. Building a client at import time (the previous
design) meant a missing key crashed the whole application before it started.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from agent.config import PROVIDER_KEYS, Settings, get_settings
from agent.obs.logging import get_logger

log = get_logger("core.llm")


class MissingCredentialsError(RuntimeError):
    """No provider key configured. Raised at call time, never at import time."""

    def __init__(self) -> None:
        names = ", ".join(name for _, keys in PROVIDER_KEYS for name in keys)
        super().__init__(f"No LLM credentials found. Set one of: {names}.")


def build_llm(settings: Settings | None = None) -> BaseChatModel:
    """Construct a chat client for the configured provider."""
    resolved = settings or get_settings()
    if resolved.provider is None or not resolved.api_key:
        raise MissingCredentialsError

    log.info("LLM provider=%s model=%s", resolved.provider, resolved.model)

    if resolved.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature: Sonnet 5 rejects the field with a 400.
        # max_tokens is the client-wide default; _finalize binds its own,
        # narrower cap per call.
        return ChatAnthropic(
            model=resolved.model,
            api_key=resolved.api_key,
            max_tokens=resolved.max_specialist_tokens,
            timeout=resolved.llm_timeout_s,
            max_retries=resolved.llm_max_retries,
        )

    kwargs: dict[str, Any] = {
        "model": resolved.model,
        "api_key": resolved.api_key,
        "temperature": resolved.temperature,
        "timeout": resolved.llm_timeout_s,
        "max_retries": resolved.llm_max_retries,
    }
    if resolved.base_url:
        kwargs["base_url"] = resolved.base_url

    return ChatOpenAI(**kwargs)


#: Values Anthropic accepts for reasoning effort. Anything else is ignored
#: rather than sent, so a typo degrades to the provider default instead of
#: failing every call in a run.
EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})

#: Bound to Runnable so ``.bind`` is known, and generic so the caller keeps
#: its concrete type - annotating this as Runnable erased ``bind_tools`` and
#: ``with_structured_output`` from everything it touched.
M = TypeVar("M", bound=Runnable[Any, Any])


def with_effort(model: M, effort: str) -> M:
    """Bind a reasoning effort, when the provider has one and it is valid.

    Non-Anthropic providers have no equivalent knob, so binding the field would
    be sent as an unknown parameter. Callers can therefore ask for an effort
    unconditionally and get the right thing per provider.
    """
    if effort not in EFFORTS:
        if effort:
            log.warning("ignoring unknown reasoning effort %r", effort)
        return model
    if get_settings().provider != "anthropic":
        return model
    return cast(M, model.bind(reasoning_effort=effort))


def build_for(model: str) -> BaseChatModel:
    """A client pinned to ``model``, for retrying a declined request.

    Measured across every available model on the same input: haiku-4-5
    answers text that sonnet-4-6, sonnet-5 and opus-5 all decline, including
    an innocuous question written backwards. The refusal is a classifier
    decision on the encoding and classifiers differ between models, so a
    second opinion is the whole remedy.
    """
    return build_llm(replace(get_settings(), model=model))


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Process-wide chat client."""
    return build_llm()


def reset_llm() -> None:
    """Test hook: drop the cached client."""
    get_llm.cache_clear()
