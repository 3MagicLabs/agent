"""Configuration resolution and provider selection."""

from __future__ import annotations

import pytest

from agent.config import Settings, load_settings, reset_settings
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
