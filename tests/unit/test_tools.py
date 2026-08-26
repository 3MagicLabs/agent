"""Tool availability, degradation and the registry.

A missing credential must produce a tool *message*, never an exception — an
import-time credential check is what crashed the deployed app.
"""

from __future__ import annotations

import pytest

from agent.config import Settings
from agent.tools import capability_report, get_tools, registered
from agent.tools.code import python_repl
from agent.tools.files import download_task_file, list_downloaded_files, read_file
from agent.tools.web import scrape_webpage, web_search, wikipedia_lookup

pytestmark = pytest.mark.unit


class TestDegradation:
    """No credentials configured: every tool answers, none raises."""

    def test_web_search_reports_unavailable(self):
        from agent.tools.web import _tavily_client

        _tavily_client.cache_clear()
        assert "unavailable" in web_search.invoke({"query": "anything"}).lower()

    def test_python_repl_reports_unavailable(self):
        assert "unavailable" in python_repl.invoke({"code": "print(1)"}).lower()

    def test_python_repl_tells_the_model_to_stop_retrying(self):
        assert "instead of retrying" in python_repl.invoke({"code": "print(1)"})


class TestSecurity:
    def test_scrape_rejects_non_http_schemes(self):
        assert "Refusing" in scrape_webpage.invoke({"url": "file:///etc/passwd"})

    def test_scrape_rejects_ftp(self):
        assert "Refusing" in scrape_webpage.invoke({"url": "ftp://example.com/x"})

    def test_read_file_refuses_path_traversal(self):
        result = read_file.invoke({"path": "../../../../etc/passwd"})
        assert "No such downloaded file" in result


class TestFiles:
    def test_list_is_empty_before_any_download(self):
        assert list_downloaded_files.invoke({}) == "No files downloaded yet."

    def test_read_file_lists_alternatives_when_missing(self, settings):
        settings.download_dir.mkdir(parents=True, exist_ok=True)
        (settings.download_dir / "present.txt").write_text("hello")

        result = read_file.invoke({"path": "absent.txt"})

        assert "present.txt" in result

    def test_reads_a_downloaded_text_file(self, settings):
        settings.download_dir.mkdir(parents=True, exist_ok=True)
        (settings.download_dir / "notes.py").write_text("print('hi')")

        assert "print('hi')" in read_file.invoke({"path": "notes.py"})

    def test_download_failure_is_reported_not_raised(self, settings, monkeypatch):
        """Every source unreachable must return a message, never raise."""
        import agent.tools.files as files_module

        def boom(*_args, **_kwargs):
            raise ConnectionError("network down")

        monkeypatch.setattr(files_module.requests, "get", boom)
        result = download_task_file.invoke({"task_id": "abc"})

        assert "No file is available" in result
        assert "gaia-benchmark/GAIA" in result

    def test_falls_back_to_the_dataset_when_the_scoring_api_has_no_file(
        self, settings, monkeypatch, tmp_path
    ):
        """The scoring API returns 404 for all five attachment tasks."""
        import agent.tools.files as files_module

        monkeypatch.setattr(files_module, "_from_scoring_api", lambda _t: None)
        monkeypatch.setattr(files_module, "_from_dataset", lambda _t: (b"col\n1\n", ".csv"))

        result = download_task_file.invoke({"task_id": "abc"})

        assert "Downloaded to" in result
        assert (settings.download_dir / "abc.csv").read_bytes() == b"col\n1\n"

    def test_the_dataset_is_skipped_without_a_token(self, settings, monkeypatch):
        """No HF_TOKEN must degrade quietly rather than hitting a 401 per task."""
        import agent.tools.files as files_module

        files_module._dataset_index.cache_clear()
        assert files_module._dataset_index() == {}


class TestRegistry:
    def test_builtin_tools_are_registered(self):
        names = {spec.name for spec in registered()}
        assert {"web_search", "scrape_webpage", "python_repl", "download_task_file"} <= names

    def test_get_tools_filters_by_capability(self):
        names = {tool.name for tool in get_tools("code")}
        assert names == {"python_repl"}

    def test_unavailable_tools_are_still_offered_by_default(self):
        """The model learns faster from an explicit 'unavailable' message."""
        assert any(tool.name == "python_repl" for tool in get_tools("code"))

    def test_unavailable_tools_can_be_excluded(self):
        tools = get_tools("code", include_unavailable=False)
        assert tools == ()

    def test_credentials_flip_availability(self):
        configured = Settings(tavily_api_key="tvly-x", e2b_api_key="e2b-x")
        report = capability_report(configured)

        assert report["web_search"] is True
        assert report["python_repl"] is True

    def test_report_marks_missing_credentials(self, settings):
        report = capability_report(settings)

        assert report["web_search"] is False
        assert report["scrape_webpage"] is True  # needs no credential


class TestWikipedia:
    def test_failure_is_reported_not_raised(self, monkeypatch):
        import agent.tools.web as web_module

        def boom(*_args, **_kwargs):
            raise TimeoutError("slow")

        monkeypatch.setattr(web_module.requests, "get", boom)

        assert "failed" in wikipedia_lookup.invoke({"title": "Mars"}).lower()


class TestDownloadMemoisation:
    """One task fetched the same spreadsheet four times: 83s and a rate limit."""

    def test_an_existing_file_is_not_refetched(self, settings, monkeypatch):
        import agent.tools.files as files_module

        settings.download_dir.mkdir(parents=True, exist_ok=True)
        (settings.download_dir / "abc.xlsx").write_bytes(b"already here")

        def explode(*_a, **_kw):
            raise AssertionError("network hit despite a cached file")

        monkeypatch.setattr(files_module.requests, "get", explode)
        result = download_task_file.invoke({"task_id": "abc"})

        assert "Already downloaded" in result
        assert "abc.xlsx" in result

    def test_inventory_is_empty_before_any_download(self, settings):
        from agent.tools.files import downloaded_inventory

        assert downloaded_inventory() == ""

    def test_inventory_lists_files_with_sizes(self, settings):
        from agent.tools.files import downloaded_inventory

        settings.download_dir.mkdir(parents=True, exist_ok=True)
        (settings.download_dir / "data.xlsx").write_bytes(b"12345")

        line = downloaded_inventory()

        assert "data.xlsx" in line
        assert "5 bytes" in line
        assert "read_file" in line
