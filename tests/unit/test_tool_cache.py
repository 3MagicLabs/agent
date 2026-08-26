"""Memoising tool results within a task."""

from __future__ import annotations

import pytest
from langchain_core.tools import tool

from agent.tools import load_builtin_tools
from agent.tools.cache import ToolCache, looks_like_failure, memoized
from agent.tools.registry import ToolSpec, registered

load_builtin_tools()

pytestmark = pytest.mark.unit


@pytest.fixture
def counter():
    """A tool that records how many times it really ran."""
    calls: list[str] = []

    @tool
    def fetch(url: str) -> str:
        """Fetch a page."""
        calls.append(url)
        return f"contents of {url}"

    return fetch, calls


class TestMemoized:
    def test_a_repeat_call_does_not_reach_the_tool(self, counter):
        fetch, calls = counter
        cached = memoized(fetch, ToolCache())

        first = cached.invoke({"url": "http://a"})
        second = cached.invoke({"url": "http://a"})

        assert "contents of http://a" in first
        assert "contents of http://a" in second
        assert calls == ["http://a"]

    def test_a_hit_is_marked_so_the_model_can_see_it_is_looping(self, counter):
        fetch, _ = counter
        cached = memoized(fetch, ToolCache())

        cached.invoke({"url": "http://a"})
        second = cached.invoke({"url": "http://a"})

        assert "already retrieved earlier in this task" in second

    def test_different_arguments_are_different_entries(self, counter):
        fetch, calls = counter
        cached = memoized(fetch, ToolCache())

        cached.invoke({"url": "http://a"})
        cached.invoke({"url": "http://b"})

        assert calls == ["http://a", "http://b"]

    def test_a_new_generation_makes_old_entries_unreachable(self, counter):
        """Tools are built once and outlive a task; the cache must not."""
        fetch, calls = counter
        cache = ToolCache()
        cached = memoized(fetch, cache)

        cached.invoke({"url": "http://a"})
        cache.new_generation()
        cached.invoke({"url": "http://a"})

        assert calls == ["http://a", "http://a"]

    def test_a_failure_is_not_cached(self):
        """A memoised failure disables the tool for the rest of the task."""
        calls: list[str] = []

        @tool
        def flaky(url: str) -> str:
            """Fetch a page."""
            calls.append(url)
            return "Failed to scrape URL http://a. Error: boom" if len(calls) == 1 else "ok"

        cached = memoized(flaky, ToolCache())

        assert "Failed" in cached.invoke({"url": "http://a"})
        assert cached.invoke({"url": "http://a"}) == "ok"
        assert len(calls) == 2

    def test_the_original_tool_is_left_alone(self, counter):
        """A new tool, not a mutated one."""
        fetch, calls = counter

        memoized(fetch, ToolCache())
        fetch.invoke({"url": "http://a"})
        fetch.invoke({"url": "http://a"})

        assert calls == ["http://a", "http://a"]

    def test_the_schema_survives_wrapping(self, counter):
        """The model sees the tool through its schema; wrapping must not alter it."""
        fetch, _ = counter
        cached = memoized(fetch, ToolCache())

        assert cached.name == fetch.name
        assert cached.description == fetch.description
        assert cached.args_schema.model_json_schema() == fetch.args_schema.model_json_schema()


class TestFailureDetection:
    @pytest.mark.parametrize(
        "result",
        [
            "Search failed with error: timeout",
            "Failed to scrape URL http://x. Error: 404",
            "web_search is unavailable: TAVILY_API_KEY is not configured.",
            "No file is available for task abc.",
            "No Wikipedia article found for 'xyz'. Try web_search instead.",
            "Refusing to fetch non-HTTP URL: file:///etc/passwd",
            "Execution Error: NameError: x is not defined",
            "Could not parse sales.xlsx: bad zip",
        ],
    )
    def test_a_tools_own_error_message_is_recognised(self, result):
        assert looks_like_failure(result)

    @pytest.mark.parametrize(
        "result",
        [
            "Giganotosaurus was promoted in November 2016, nominated by FunkMonk.",
            "name,amount\nwidget,12\nTOTAL,89706.00",
            "3",
        ],
    )
    def test_a_real_result_is_not(self, result):
        assert not looks_like_failure(result)

    def test_a_page_discussing_a_failure_is_still_cacheable(self):
        """Only the opening is inspected, so page content does not trip it.

        A false positive costs a refetch; a false negative caches a failure and
        disables the tool for the task. The bias is deliberate.
        """
        article = "Apollo 13 mission summary. " + "x" * 300 + " the oxygen tank failed."

        assert not looks_like_failure(article)


class TestRegistryPolicy:
    """Which tools may be cached is declared, not remembered."""

    def test_code_execution_is_never_cached(self):
        """Code can be nondeterministic, and rerunning it can be intentional."""
        specs = {spec.name: spec for spec in registered()}

        assert specs["python_repl"].cacheable is False

    def test_read_only_lookups_are_cached(self):
        specs = {spec.name: spec for spec in registered()}

        for name in ("web_search", "scrape_webpage", "wikipedia_lookup", "read_file"):
            assert specs[name].cacheable is True, name

    def test_the_live_listing_is_never_cached(self):
        """Its whole purpose is reflecting what has changed since."""
        specs = {spec.name: spec for spec in registered()}

        assert specs["list_downloaded_files"].cacheable is False

    def test_caching_is_opt_in(self):
        """A new tool is safe until someone has thought about it."""
        assert ToolSpec(name="x", capability="c", factory=lambda: None).cacheable is False
