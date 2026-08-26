"""Configuration resolution and provider selection."""

from __future__ import annotations

import pytest

from agent.config import Settings, get_settings, load_settings, reset_settings
from agent.core.llm import MissingCredentialsError, build_llm

pytestmark = pytest.mark.unit


class TestProviderSelection:
    def test_no_credentials_yields_no_provider(self, monkeypatch):
        reset_settings()
        assert load_settings().provider is None

    def test_groq_wins_when_several_keys_are_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENAI_API_KEY", "o")

        assert load_settings().provider == "groq"

    def test_openai_used_when_groq_absent(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "o")

        settings = load_settings()
        assert settings.provider == "openai"
        assert settings.base_url == ""

    def test_huggingface_accepts_either_token_name(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf")

        assert load_settings().provider == "huggingface"

    def test_model_is_overridable_per_provider(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")

        assert load_settings().model == "llama-3.1-8b-instant"


class TestDerivedValues:
    def test_recursion_limit_scales_with_step_budget(self):
        assert Settings(max_supervisor_steps=4).recursion_limit == 18
        assert Settings(max_supervisor_steps=10).recursion_limit == 36

    def test_paths_are_derived_from_the_log_directory(self, tmp_path):
        settings = Settings(log_dir=tmp_path)

        assert settings.metrics_file == tmp_path / "metrics.jsonl"
        assert settings.answer_cache == tmp_path / "answers.json"

    def test_capability_flags_follow_credentials(self):
        assert Settings().has_search is False
        assert Settings(tavily_api_key="x").has_search is True
        assert Settings(e2b_api_key="x").has_sandbox is True

    def test_settings_are_immutable(self):
        settings = Settings()
        with pytest.raises(AttributeError):
            settings.model = "other"  # type: ignore[misc]

    def test_with_provider_returns_a_copy(self):
        original = Settings()
        derived = original.with_provider("openai")

        assert original.provider is None
        assert derived.provider == "openai"
        assert derived.model == "gpt-4o-mini"


class TestInvalidEnvironment:
    def test_non_numeric_values_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("MAX_SUPERVISOR_STEPS", "not-a-number")

        assert load_settings().max_supervisor_steps == 4

    def test_missing_credentials_raise_only_when_a_model_is_built(self):
        """Import must never fail; construction is where it surfaces."""
        with pytest.raises(MissingCredentialsError, match="No LLM credentials"):
            build_llm(Settings())


class TestProviderSelector:
    """Ordering alone is not enough once more than one key is real.

    With Groq's daily quota spent, GROQ_API_KEY still won the scan and the
    HF token sat unused - the failure looked like an outage, not a config bug.
    """

    def test_named_provider_wins_over_the_scan_order(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("HF_TOKEN", "h")
        monkeypatch.setenv("LLM_PROVIDER", "huggingface")
        reset_settings()

        assert get_settings().provider == "huggingface"

    def test_scan_order_still_applies_when_unset(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("HF_TOKEN", "h")
        reset_settings()

        assert get_settings().provider == "groq"

    def test_a_named_provider_without_a_key_does_not_fall_through(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("LLM_PROVIDER", "huggingface")
        reset_settings()

        assert get_settings().provider is None

    def test_an_unknown_provider_name_resolves_to_nothing(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("LLM_PROVIDER", "nonsense")
        reset_settings()

        assert get_settings().provider is None

    def test_the_name_is_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "h")
        monkeypatch.setenv("LLM_PROVIDER", "HuggingFace")
        reset_settings()

        assert get_settings().provider == "huggingface"


class TestDefaults:
    """The dataclass defaults and load_settings' fallbacks must not drift.

    They were separate literals once: editing a field changed what tests build
    and nothing about what the agent ran, so a 300s timeout stayed 180s in
    production while the suite stayed green.
    """

    #: (Settings field, environment variable that overrides it)
    TUNABLES = (
        ("temperature", "LLM_TEMPERATURE"),
        ("llm_timeout_s", "LLM_TIMEOUT_S"),
        ("llm_max_retries", "LLM_MAX_RETRIES"),
        ("max_answer_tokens", "MAX_ANSWER_TOKENS"),
        ("max_supervisor_steps", "MAX_SUPERVISOR_STEPS"),
        ("max_web_iterations", "MAX_WEB_ITERATIONS"),
        ("max_code_iterations", "MAX_CODE_ITERATIONS"),
        ("history_window", "HISTORY_WINDOW"),
        ("per_question_timeout_s", "PER_QUESTION_TIMEOUT_S"),
        ("total_budget_s", "TOTAL_BUDGET_S"),
        ("tokens_per_minute", "TOKENS_PER_MINUTE"),
        ("max_scrape_chars", "MAX_SCRAPE_CHARS"),
        ("max_file_chars", "MAX_FILE_CHARS"),
        ("max_code_output_chars", "MAX_CODE_OUTPUT_CHARS"),
        ("scrape_timeout_s", "SCRAPE_TIMEOUT_S"),
        ("sandbox_timeout_s", "SANDBOX_TIMEOUT_S"),
        ("search_results", "SEARCH_RESULTS"),
        ("scoring_api_url", "SCORING_API_URL"),
        ("log_level", "LOG_LEVEL"),
    )

    def test_a_clean_environment_yields_the_dataclass_defaults(self, monkeypatch):
        for _, variable in self.TUNABLES:
            monkeypatch.delenv(variable, raising=False)

        loaded, defaults = load_settings(), Settings()

        for field, _ in self.TUNABLES:
            assert getattr(loaded, field) == getattr(defaults, field), field

    def test_an_environment_variable_still_wins(self, monkeypatch):
        """The single source of truth is a fallback, not an override."""
        monkeypatch.setenv("PER_QUESTION_TIMEOUT_S", "42")

        assert load_settings().per_question_timeout_s == 42.0

    def test_the_budgets_can_accommodate_the_step_budget(self):
        """A task must be able to finish inside its own timeout.

        4 supervisor steps x 3 specialist iterations is ~12 LLM calls; at the
        latencies measured against a throttled provider that exceeded 180s, so
        tasks were killed mid-progress - one of them 95s before it produced the
        correct answer.
        """
        settings = Settings()
        calls = settings.max_supervisor_steps * settings.max_web_iterations

        assert settings.per_question_timeout_s >= calls * 20.0
        assert settings.total_budget_s >= settings.per_question_timeout_s


class TestEffort:
    def test_the_router_runs_cheap_by_default(self):
        """It picks one name and writes a sentence; depth buys nothing."""
        assert Settings().router_effort == "low"

    def test_the_specialist_effort_is_unset_by_default(self):
        """Empty means 'provider default', which is what an A/B starts from."""
        assert Settings().specialist_effort == ""

    def test_effort_is_overridable_for_experiments(self, monkeypatch):
        monkeypatch.setenv("SPECIALIST_EFFORT", "low")

        assert load_settings().specialist_effort == "low"
