"""Single source of truth for runtime configuration.

Every tunable is read from the environment exactly once, into a frozen
``Settings`` object. Modules never call ``os.getenv`` directly, so the full
configuration surface is discoverable in one place and trivially overridable
in tests via ``Settings(...)``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final, Literal

Provider = Literal["groq", "openai", "huggingface"]

#: Provider selection order. First key present wins.
PROVIDER_KEYS: Final[tuple[tuple[Provider, tuple[str, ...]], ...]] = (
    ("groq", ("GROQ_API_KEY",)),
    ("openai", ("OPENAI_API_KEY",)),
    ("huggingface", ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")),
)

PROVIDER_DEFAULTS: Final[dict[Provider, tuple[str, str]]] = {
    # provider -> (default model, base_url); empty base_url means the SDK default
    "groq": ("llama-3.3-70b-versatile", "https://api.groq.com/openai/v1"),
    "openai": ("gpt-4o-mini", ""),
    "huggingface": ("Qwen/Qwen2.5-Coder-32B-Instruct", "https://router.huggingface.co/v1"),
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration. Use ``replace()`` to derive a variant."""

    # --- model ---
    provider: Provider | None = None
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    #: Hard ceiling on the finalizer's reply. A graded answer is a few words;
    #: without a cap a repetition loop can emit thousands of tokens of garbage.
    max_answer_tokens: int = 128

    # --- orchestration budgets ---
    max_supervisor_steps: int = 4
    max_web_iterations: int = 3
    max_code_iterations: int = 3
    history_window: int = 8

    # --- run budgets ---
    per_question_timeout_s: float = 180.0
    total_budget_s: float = 2400.0
    #: Provider tokens-per-minute allowance; the runner sleeps between tasks to
    #: stay under it. 0 disables pacing. Groq's free tier reports 12000 in its
    #: x-ratelimit-limit-tokens header.
    tokens_per_minute: int = 12000

    # --- tools ---
    tavily_api_key: str = ""
    e2b_api_key: str = ""
    #: Read independently of provider resolution: the GAIA dataset is gated,
    #: and its files are needed even when the LLM provider is not HuggingFace.
    hf_token: str = ""
    max_scrape_chars: int = 6000
    max_file_chars: int = 12000
    max_code_output_chars: int = 4000
    scrape_timeout_s: float = 20.0
    sandbox_timeout_s: int = 60
    search_results: int = 3

    # --- benchmark ---
    scoring_api_url: str = "https://agents-course-unit4-scoring.hf.space"
    download_dir: Path = Path("logs/downloads")

    # --- observability ---
    log_level: str = "INFO"
    log_dir: Path = Path("logs")
    langsmith_api_key: str = ""
    langsmith_project: str = "3magic-agent"

    @property
    def recursion_limit(self) -> int:
        """LangGraph node budget derived from the delegation budget."""
        return self.max_supervisor_steps * 3 + 6

    @property
    def metrics_file(self) -> Path:
        return self.log_dir / "metrics.jsonl"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "agent.log"

    @property
    def answer_cache(self) -> Path:
        return self.log_dir / "answers.json"

    @property
    def has_search(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_sandbox(self) -> bool:
        return bool(self.e2b_api_key)

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langsmith_api_key)

    def with_provider(self, provider: Provider, model: str = "") -> Settings:
        """Return a copy pinned to a provider (used by tests and the CLI)."""
        default_model, base_url = PROVIDER_DEFAULTS[provider]
        return replace(self, provider=provider, model=model or default_model, base_url=base_url)


def _resolve_provider() -> tuple[Provider | None, str, str, str]:
    """Pick the provider to use. Returns (provider, model, base_url, key).

    ``LLM_PROVIDER`` names one explicitly. Without it the first key present
    wins, in ``PROVIDER_KEYS`` order.

    An explicit choice whose key is missing resolves to nothing rather than
    falling through to the next provider. Silent fallthrough is how a spent or
    unfunded key keeps getting used while you believe you switched away from it.
    """
    requested = os.getenv("LLM_PROVIDER", "").strip().lower()
    candidates = (
        tuple(entry for entry in PROVIDER_KEYS if entry[0] == requested)
        if requested
        else PROVIDER_KEYS
    )

    for provider, env_names in candidates:
        key = next((os.environ[n] for n in env_names if os.environ.get(n)), "")
        if not key:
            continue
        default_model, base_url = PROVIDER_DEFAULTS[provider]
        model = os.getenv(f"{provider.upper()}_MODEL", default_model)
        return provider, model, os.getenv("LLM_BASE_URL", base_url), key
    return None, "", "", ""


def load_settings() -> Settings:
    """Build Settings from the process environment."""
    provider, model, base_url, api_key = _resolve_provider()
    return Settings(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=_env_float("LLM_TEMPERATURE", 0.0),
        llm_timeout_s=_env_float("LLM_TIMEOUT_S", 60.0),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        max_answer_tokens=_env_int("MAX_ANSWER_TOKENS", 128),
        max_supervisor_steps=_env_int("MAX_SUPERVISOR_STEPS", 4),
        max_web_iterations=_env_int("MAX_WEB_ITERATIONS", 3),
        max_code_iterations=_env_int("MAX_CODE_ITERATIONS", 3),
        history_window=_env_int("HISTORY_WINDOW", 8),
        per_question_timeout_s=_env_float("PER_QUESTION_TIMEOUT_S", 180.0),
        total_budget_s=_env_float("TOTAL_BUDGET_S", 2400.0),
        tokens_per_minute=_env_int("TOKENS_PER_MINUTE", 12000),
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        e2b_api_key=os.getenv("E2B_API_KEY", ""),
        hf_token=(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""),
        max_scrape_chars=_env_int("MAX_SCRAPE_CHARS", 6000),
        max_file_chars=_env_int("MAX_FILE_CHARS", 12000),
        max_code_output_chars=_env_int("MAX_CODE_OUTPUT_CHARS", 4000),
        scrape_timeout_s=_env_float("SCRAPE_TIMEOUT_S", 20.0),
        sandbox_timeout_s=_env_int("SANDBOX_TIMEOUT_S", 60),
        search_results=_env_int("SEARCH_RESULTS", 3),
        scoring_api_url=os.getenv(
            "SCORING_API_URL", "https://agents-course-unit4-scoring.hf.space"
        ),
        download_dir=Path(os.getenv("AGENT_DOWNLOAD_DIR", "logs/downloads")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_dir=Path(os.getenv("AGENT_LOG_DIR", "logs")),
        langsmith_api_key=(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or ""),
        langsmith_project=(
            os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "3magic-agent"
        ),
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings, loaded once."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Override the process-wide settings. Intended for tests and the CLI."""
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Drop cached settings so the next read re-reads the environment."""
    global _settings
    _settings = None
