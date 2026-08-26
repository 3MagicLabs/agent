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

Provider = Literal["anthropic", "groq", "openai", "huggingface"]

#: Provider selection order. First key present wins.
PROVIDER_KEYS: Final[tuple[tuple[Provider, tuple[str, ...]], ...]] = (
    ("anthropic", ("ANTHROPIC_API_KEY",)),
    ("groq", ("GROQ_API_KEY",)),
    ("openai", ("OPENAI_API_KEY",)),
    ("huggingface", ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")),
)

PROVIDER_DEFAULTS: Final[dict[Provider, tuple[str, str]]] = {
    # provider -> (default model, base_url); empty base_url means the SDK default
    "anthropic": ("claude-sonnet-5", ""),
    "groq": ("openai/gpt-oss-120b", "https://api.groq.com/openai/v1"),
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
    #: Ceiling on the router's reply. It emits one schema selection plus a
    #: short justification; generous because Sonnet 5 spends output tokens on
    #: adaptive thinking, and a cap that truncates mid-thought yields a
    #: malformed structured output rather than a cheaper one.
    max_router_tokens: int = 512
    #: Anthropic reasoning effort, one of low|medium|high|xhigh|max. Empty
    #: uses the provider default (high) and is what non-Anthropic providers
    #: get, since they have no equivalent knob.
    #:
    #: Lower effort means fewer and more-consolidated tool calls, which is
    #: why this is worth more than its effect on output tokens: fewer calls
    #: means fewer delegation rounds, and rounds drive the transcript replay
    #: that is 94% of a run's spend. Which setting is right is an empirical
    #: question - run the same tasks at two values and score both.
    #:
    #: The router picks one name from a fixed list and writes a sentence; the
    #: finalizer formats an answer it has already been handed. Neither is
    #: reasoning, so both run cheap. Specialists carry the actual work, but
    #: level-1 tasks are lookups and small computations rather than deep
    #: reasoning, so "medium" rather than the provider's "high". Set
    #: deliberately rather than left blank: a recorded "medium" is a
    #: configuration under test, whereas a blank is only "whatever the provider
    #: chose", which is not a thing an A/B can compare against.
    router_effort: str = "low"
    specialist_effort: str = "medium"
    finalizer_effort: str = "low"
    #: Ceiling for any call that does not bind its own. Anthropic requires
    #: max_tokens at construction, so this is the client-wide default and the
    #: finalizer narrows it per call. It must fit a specialist's reasoning plus
    #: a tool call - the finalizer's 128 would truncate one mid-thought.
    max_specialist_tokens: int = 1024

    # --- orchestration budgets ---
    max_supervisor_steps: int = 4
    max_web_iterations: int = 3
    max_code_iterations: int = 3
    history_window: int = 8

    # --- run budgets ---
    per_question_timeout_s: float = 300.0
    total_budget_s: float = 6000.0
    #: Provider tokens-per-minute allowance; the runner sleeps between tasks to
    #: stay under it. 0 disables pacing. Groq's free tier reports 12000 in its
    #: x-ratelimit-limit-tokens header.
    tokens_per_minute: int = 12000
    #: Dollar ceilings. The free provider had an involuntary daily token cap;
    #: a paid one has none, so this is the only thing standing between a
    #: retry loop and real money. 0 disables a ceiling.
    #: Two of them because they catch different failures: per-task catches
    #: one runaway, per-run catches many slightly-too-expensive ones.
    max_task_cost_usd: float = 0.50
    max_run_cost_usd: float = 5.00

    # --- tools ---
    tavily_api_key: str = ""
    e2b_api_key: str = ""
    #: Read independently of provider resolution: the GAIA dataset is gated,
    #: and its files are needed even when the LLM provider is not HuggingFace.
    hf_token: str = ""
    #: How much of a tool's output may enter the transcript. These were sized
    #: for Groq's 8,000 tokens-per-minute ceiling, where 12,000 characters was
    #: already most of a minute's allowance. Against Sonnet's 1M context that
    #: was 0.3% of what fits, and the middle of every document was being thrown
    #: away for no reason.
    #:
    #: The binding constraint is now replay, not context: a specialist resends
    #: its whole transcript each iteration, so a document costs its size times
    #: the number of iterations. At $2/1M input, 60,000 characters (~15k tokens)
    #: replayed three times is about $0.09 - comfortable inside the $0.50
    #: per-task ceiling.
    #:
    #: For data too large to be worth any of this, the answer is not a bigger
    #: limit or a retrieval index: it is python_repl computing over the file and
    #: returning the number.
    max_scrape_chars: int = 30000
    max_file_chars: int = 60000
    max_code_output_chars: int = 15000
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


#: The field defaults above, as an object. ``load_settings`` falls back to these
#: rather than repeating each literal, so every default is written exactly once.
#: When the two were separate, editing a field changed what tests construct and
#: nothing about what the agent actually ran.
_DEFAULTS: Final[Settings] = Settings()


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
        temperature=_env_float("LLM_TEMPERATURE", _DEFAULTS.temperature),
        llm_timeout_s=_env_float("LLM_TIMEOUT_S", _DEFAULTS.llm_timeout_s),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", _DEFAULTS.llm_max_retries),
        max_answer_tokens=_env_int("MAX_ANSWER_TOKENS", _DEFAULTS.max_answer_tokens),
        max_router_tokens=_env_int("MAX_ROUTER_TOKENS", _DEFAULTS.max_router_tokens),
        router_effort=os.getenv("ROUTER_EFFORT", _DEFAULTS.router_effort),
        specialist_effort=os.getenv("SPECIALIST_EFFORT", _DEFAULTS.specialist_effort),
        finalizer_effort=os.getenv("FINALIZER_EFFORT", _DEFAULTS.finalizer_effort),
        max_supervisor_steps=_env_int("MAX_SUPERVISOR_STEPS", _DEFAULTS.max_supervisor_steps),
        max_web_iterations=_env_int("MAX_WEB_ITERATIONS", _DEFAULTS.max_web_iterations),
        max_code_iterations=_env_int("MAX_CODE_ITERATIONS", _DEFAULTS.max_code_iterations),
        history_window=_env_int("HISTORY_WINDOW", _DEFAULTS.history_window),
        per_question_timeout_s=_env_float(
            "PER_QUESTION_TIMEOUT_S", _DEFAULTS.per_question_timeout_s
        ),
        total_budget_s=_env_float("TOTAL_BUDGET_S", _DEFAULTS.total_budget_s),
        tokens_per_minute=_env_int("TOKENS_PER_MINUTE", _DEFAULTS.tokens_per_minute),
        max_task_cost_usd=_env_float("MAX_TASK_COST_USD", _DEFAULTS.max_task_cost_usd),
        max_run_cost_usd=_env_float("MAX_RUN_COST_USD", _DEFAULTS.max_run_cost_usd),
        tavily_api_key=os.getenv("TAVILY_API_KEY", _DEFAULTS.tavily_api_key),
        e2b_api_key=os.getenv("E2B_API_KEY", _DEFAULTS.e2b_api_key),
        hf_token=(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""),
        max_scrape_chars=_env_int("MAX_SCRAPE_CHARS", _DEFAULTS.max_scrape_chars),
        max_file_chars=_env_int("MAX_FILE_CHARS", _DEFAULTS.max_file_chars),
        max_code_output_chars=_env_int("MAX_CODE_OUTPUT_CHARS", _DEFAULTS.max_code_output_chars),
        scrape_timeout_s=_env_float("SCRAPE_TIMEOUT_S", _DEFAULTS.scrape_timeout_s),
        sandbox_timeout_s=_env_int("SANDBOX_TIMEOUT_S", _DEFAULTS.sandbox_timeout_s),
        search_results=_env_int("SEARCH_RESULTS", _DEFAULTS.search_results),
        scoring_api_url=os.getenv("SCORING_API_URL", _DEFAULTS.scoring_api_url),
        download_dir=Path(os.getenv("AGENT_DOWNLOAD_DIR", str(_DEFAULTS.download_dir))),
        log_level=os.getenv("LOG_LEVEL", _DEFAULTS.log_level).upper(),
        log_dir=Path(os.getenv("AGENT_LOG_DIR", str(_DEFAULTS.log_dir))),
        langsmith_api_key=(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or ""),
        langsmith_project=(
            os.getenv("LANGSMITH_PROJECT")
            or os.getenv("LANGCHAIN_PROJECT")
            or _DEFAULTS.langsmith_project
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
