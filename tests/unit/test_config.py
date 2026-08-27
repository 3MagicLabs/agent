"""Configuration resolution and provider selection."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent.config import PROVIDER_DEFAULTS, Settings, get_settings, load_settings, reset_settings
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
        calls = settings.max_supervisor_steps * max(
            settings.max_web_iterations, settings.max_code_iterations
        )

        # 8s per call, measured: five real tasks averaged 2-4s per LLM call
        # against Anthropic. The original 20s came from Groq, where every
        # call carried throttling - and like the token limits and the pacer,
        # it outlived the provider it was measured on.
        assert settings.per_question_timeout_s >= calls * 8.0
        assert settings.total_budget_s >= settings.per_question_timeout_s


class TestEffort:
    def test_the_router_does_not_run_at_the_lowest_effort(self):
        """At "low" it returned an empty object - 0 output tokens, no fields -
        and the pydantic validation failure ended a task that had succeeded on
        every previous run. A component that must emit valid structured output
        has to earn the right to be cheap."""
        assert Settings().router_effort != "low"

    def test_the_specialist_runs_at_medium(self):
        """Level-1 tasks are lookups and small computations, not deep
        reasoning. Set explicitly so a metric records a configuration under
        test rather than "whatever the provider chose"."""
        assert Settings().specialist_effort == "medium"

    def test_every_role_has_a_valid_effort(self):
        """An unknown value is dropped with a warning, so a typo here would
        silently run at the provider default instead of the intended one."""
        from agent.core.llm import EFFORTS

        settings = Settings()
        for role in ("router_effort", "specialist_effort", "finalizer_effort"):
            assert getattr(settings, role) in EFFORTS, role

    def test_effort_is_overridable_for_experiments(self, monkeypatch):
        monkeypatch.setenv("SPECIALIST_EFFORT", "low")

        assert load_settings().specialist_effort == "low"


def _same(documented: str, actual: object) -> bool:
    """Compare numerically where possible - "5.00" and 5.0 are the same default."""
    try:
        return float(documented) == float(actual)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return documented == str(actual)


class TestDocumentedDefaults:
    """Every tunable default must match what the documentation claims.

    Eighteen commits changed eight defaults and touched zero lines of
    documentation, so docs/configuration.md described a configuration that had
    not existed for a day - including a retired model and a pacing value from a
    provider no longer in use. Prose cannot be trusted to track code by
    intention; this makes it fail instead.
    """

    DOC = Path("docs/configuration.md")

    def _documented(self) -> dict[str, str]:
        """Variable -> default, parsed from the markdown tables."""
        rows = re.findall(r"^\|\s*`([A-Z_]+)`\s*\|\s*`([^`]*)`", self.DOC.read_text(), re.M)
        return dict(rows)

    @pytest.mark.parametrize(
        ("variable", "field"),
        [
            ("MAX_SUPERVISOR_STEPS", "max_supervisor_steps"),
            ("MAX_WEB_ITERATIONS", "max_web_iterations"),
            ("MAX_CODE_ITERATIONS", "max_code_iterations"),
            ("HISTORY_WINDOW", "history_window"),
            ("PER_QUESTION_TIMEOUT_S", "per_question_timeout_s"),
            ("TOTAL_BUDGET_S", "total_budget_s"),
            ("MAX_ANSWER_TOKENS", "max_answer_tokens"),
            ("MAX_ROUTER_TOKENS", "max_router_tokens"),
            ("MAX_SPECIALIST_TOKENS", "max_specialist_tokens"),
            ("TOKENS_PER_MINUTE", "tokens_per_minute"),
            ("MAX_TASK_COST_USD", "max_task_cost_usd"),
            ("MAX_RUN_COST_USD", "max_run_cost_usd"),
            ("MAX_SCRAPE_CHARS", "max_scrape_chars"),
            ("MAX_FILE_CHARS", "max_file_chars"),
            ("MAX_CODE_OUTPUT_CHARS", "max_code_output_chars"),
            ("SEARCH_RESULTS", "search_results"),
            ("ROUTER_EFFORT", "router_effort"),
            ("SPECIALIST_EFFORT", "specialist_effort"),
            ("FINALIZER_EFFORT", "finalizer_effort"),
        ],
    )
    def test_the_documented_default_is_the_real_one(self, variable, field):
        documented = self._documented().get(variable)
        actual = getattr(Settings(), field)

        assert documented is not None, f"{variable} is undocumented"
        assert _same(
            documented, actual
        ), f"{variable}: docs say {documented!r}, code says {actual!r}"

    def test_the_configured_model_is_documented(self):
        documented = self._documented().get("ANTHROPIC_MODEL")

        assert documented == PROVIDER_DEFAULTS["anthropic"][0]
