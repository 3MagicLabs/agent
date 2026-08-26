"""Chat model factory.

One cached client per process. Building a client at import time (the previous
design) meant a missing key crashed the whole application before it started.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
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


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Process-wide chat client."""
    return build_llm()


def reset_llm() -> None:
    """Test hook: drop the cached client."""
    get_llm.cache_clear()
