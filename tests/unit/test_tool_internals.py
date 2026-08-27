"""Tool bodies exercised against mocked transports.

These paths are where the deployed agent actually spent its time, so they are
tested without touching the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from agent.tools import code as code_module
from agent.tools import files as files_module
from agent.tools import web as web_module
from agent.tools.code import _execute, _render, python_repl
from agent.tools.files import download_task_file, read_file
from agent.tools.web import _format_results, scrape_webpage, wikipedia_lookup

pytestmark = pytest.mark.unit


class FakeResponse:
    def __init__(self, text="", json_data=None, headers=None, content=b""):
        self.text = text
        self._json = json_data or {}
        self.headers = headers or {"content-type": "text/html"}
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._json


@pytest.fixture
def fake_get(monkeypatch):
    def _install(module, response):
        monkeypatch.setattr(module.requests, "get", lambda *_a, **_k: response)
        return response

    return _install


class TestScrape:
    HTML = """
    <html><head><style>.x{}</style></head>
    <body><nav>menu</nav><p>Real content here.</p><footer>legal</footer></body></html>
    """

    def test_strips_boilerplate_tags(self, fake_get):
        fake_get(web_module, FakeResponse(text=self.HTML))

        result = scrape_webpage.invoke({"url": "https://example.com"})

        assert "Real content here." in result
        assert "menu" not in result
        assert "legal" not in result

    def test_truncates_long_pages(self, fake_get, settings):
        body = "<html><body>" + ("word " * 20_000) + "</body></html>"
        fake_get(web_module, FakeResponse(text=body))

        result = scrape_webpage.invoke({"url": "https://example.com"})

        assert "content elided" in result
        assert len(result) <= settings.max_scrape_chars + 50

    def test_non_html_is_returned_raw(self, fake_get):
        fake_get(
            web_module,
            FakeResponse(text="col_a,col_b\n1,2", headers={"content-type": "text/csv"}),
        )

        assert scrape_webpage.invoke({"url": "https://example.com/x.csv"}) == "col_a,col_b\n1,2"


class TestSearchFormatting:
    def test_includes_the_direct_answer_first(self):
        payload = {"answer": "42", "results": [{"title": "T", "url": "u", "content": "c"}]}

        output = _format_results(payload, limit=3)

        assert output.startswith("Direct answer: 42")
        assert "### T" in output

    def test_respects_the_result_limit(self):
        payload = {"results": [{"title": f"T{i}", "url": "", "content": ""} for i in range(10)]}

        assert _format_results(payload, limit=2).count("###") == 2

    def test_empty_payload(self):
        assert _format_results({}, limit=3) == "No results found."


class TestWikipedia:
    def test_returns_the_article_extract(self, fake_get):
        fake_get(
            web_module,
            FakeResponse(json_data={"query": {"pages": {"1": {"extract": "Mars is a planet."}}}}),
        )

        assert "Mars is a planet." in wikipedia_lookup.invoke({"title": "Mars"})

    def test_missing_article_suggests_search(self, fake_get):
        fake_get(web_module, FakeResponse(json_data={"query": {"pages": {"-1": {}}}}))

        assert "Try web_search" in wikipedia_lookup.invoke({"title": "Nonexistent"})


class TestDownload:
    def test_saves_with_the_suffix_from_the_header(self, fake_get, settings):
        fake_get(
            files_module,
            FakeResponse(
                content=b"a,b\n1,2",
                headers={"content-disposition": 'attachment; filename="data.csv"'},
            ),
        )

        result = download_task_file.invoke({"task_id": "task9"})

        assert "task9.csv" in result
        assert (settings.download_dir / "task9.csv").read_bytes() == b"a,b\n1,2"

    def test_downloaded_csv_is_readable_as_a_table(self, fake_get, settings):
        fake_get(
            files_module,
            FakeResponse(
                content=b"name,score\nalice,10\nbob,20",
                headers={"content-disposition": 'attachment; filename="s.csv"'},
            ),
        )
        download_task_file.invoke({"task_id": "t"})

        table = read_file.invoke({"path": "t.csv"})

        assert "2 rows x 2 columns" in table
        assert "alice" in table

    def test_unsupported_binary_points_at_the_right_tool(self, settings):
        settings.download_dir.mkdir(parents=True, exist_ok=True)
        (settings.download_dir / "clip.mp3").write_bytes(b"\x00\x01")

        assert "transcribe_audio" in read_file.invoke({"path": "clip.mp3"})


@dataclass
class FakeLogs:
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)


@dataclass
class FakeError:
    name: str
    value: str
    traceback: str = ""


@dataclass
class FakeExecution:
    logs: FakeLogs = field(default_factory=FakeLogs)
    error: FakeError | None = None
    results: list = field(default_factory=list)


class TestSandboxRendering:
    def test_stdout_is_returned(self):
        assert _render(FakeExecution(logs=FakeLogs(stdout=["42"])), limit=100) == "42"

    def test_errors_include_the_traceback(self):
        execution = FakeExecution(error=FakeError("ValueError", "bad", "line 1"))

        output = _render(execution, limit=100)

        assert "ValueError: bad" in output
        assert "line 1" in output

    def test_silent_success_nudges_toward_print(self):
        assert "forget to print" in _render(FakeExecution(), limit=100)

    def test_output_is_truncated(self):
        execution = FakeExecution(logs=FakeLogs(stdout=["x" * 500]))

        assert "output elided" in _render(execution, limit=50)

    def test_stderr_is_labelled(self):
        execution = FakeExecution(logs=FakeLogs(stdout=["ok"], stderr=["warning"]))

        assert "[stderr]" in _render(execution, limit=200)


class TestSandboxApiCompatibility:
    """The deployed bug: calling the pre-1.0 API on a 2.x SDK."""

    def test_prefers_run_code_on_modern_sdks(self):
        class Modern:
            def run_code(self, code, timeout=None):
                return f"ran {code}"

        assert _execute(Modern(), "1+1") == "ran 1+1"

    def test_falls_back_to_the_legacy_notebook_api(self):
        class Legacy:
            class notebook:  # noqa: N801
                @staticmethod
                def exec_cell(code):
                    return f"cell {code}"

        assert _execute(Legacy(), "1+1") == "cell 1+1"

    def test_unknown_api_raises_a_clear_error(self):
        with pytest.raises(AttributeError, match="Unsupported E2B sandbox API"):
            _execute(object(), "1+1")


class TestSandboxExecution:
    def test_runs_code_when_configured(self, monkeypatch, settings):
        from dataclasses import replace

        from agent.config import set_settings

        set_settings(replace(settings, e2b_api_key="e2b-key"))

        class FakeSandbox:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def run_code(self, _code, timeout=None):
                return FakeExecution(logs=FakeLogs(stdout=["7"]))

        monkeypatch.setattr(code_module, "_load_sandbox_class", lambda: FakeSandbox)

        assert python_repl.invoke({"code": "print(7)"}) == "7"

    def test_sdk_absence_is_reported_not_raised(self, monkeypatch, settings):
        from dataclasses import replace

        from agent.config import set_settings

        set_settings(replace(settings, e2b_api_key="e2b-key"))

        def missing():
            raise ImportError("no module named e2b_code_interpreter")

        monkeypatch.setattr(code_module, "_load_sandbox_class", missing)

        assert "unavailable" in python_repl.invoke({"code": "print(1)"})


class TestDatasetIndex:
    """The GAIA listing must retry after a failure, not memoise it."""

    @pytest.fixture(autouse=True)
    def _clear_index(self):
        files_module._INDEX.clear()
        yield
        files_module._INDEX.clear()

    def test_a_failed_listing_is_retried(self, monkeypatch, settings):
        """One transient error must not disable attachments for the process.

        Measured before this fix: six consecutive tasks failed against an empty
        index while the same request succeeded a minute later.
        """
        monkeypatch.setattr(files_module, "get_settings", lambda: replace(settings, hf_token="t"))
        attempts: list[int] = []

        class Response:
            def raise_for_status(self) -> None:
                if len(attempts) == 1:
                    raise OSError("transient")

            def json(self) -> list[dict[str, str]]:
                return [{"path": "2023/validation/abc.xlsx"}]

        def fetch(*_args, **_kwargs):
            attempts.append(1)
            return Response()

        monkeypatch.setattr(files_module.requests, "get", fetch)

        assert files_module._dataset_index() == {}
        assert files_module._dataset_index() == {"abc": "2023/validation/abc.xlsx"}
        assert len(attempts) == 2

    def test_a_successful_listing_is_cached(self, monkeypatch, settings):
        monkeypatch.setattr(files_module, "get_settings", lambda: replace(settings, hf_token="t"))
        attempts: list[int] = []

        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> list[dict[str, str]]:
                return [{"path": "2023/validation/abc.xlsx"}]

        def fetch(*_args, **_kwargs):
            attempts.append(1)
            return Response()

        monkeypatch.setattr(files_module.requests, "get", fetch)

        files_module._dataset_index()
        files_module._dataset_index()

        assert len(attempts) == 1

    def test_no_token_means_no_listing_attempt(self, monkeypatch, settings):
        monkeypatch.setattr(files_module, "get_settings", lambda: replace(settings, hf_token=""))
        monkeypatch.setattr(
            files_module.requests, "get", lambda *a, **k: pytest.fail("should not fetch")
        )

        assert files_module._dataset_index() == {}


class TestInventoryScoping:
    """The download directory outlives a task; the inventory must not."""

    def test_only_the_current_task_is_listed(self, settings, monkeypatch):
        """An unscoped listing offered the Excel task a Python file and a chess
        image left by earlier tasks, and it read both."""
        monkeypatch.setattr(files_module, "get_settings", lambda: settings)
        root = settings.download_dir
        root.mkdir(parents=True, exist_ok=True)
        (root / "aaaa1111.xlsx").write_bytes(b"x")
        (root / "bbbb2222.py").write_bytes(b"y")

        listing = files_module.downloaded_inventory("aaaa1111")

        assert "aaaa1111.xlsx" in listing
        assert "bbbb2222.py" not in listing

    def test_no_task_id_lists_everything(self, settings, monkeypatch):
        """list_downloaded_files wants the whole directory."""
        monkeypatch.setattr(files_module, "get_settings", lambda: settings)
        root = settings.download_dir
        root.mkdir(parents=True, exist_ok=True)
        (root / "aaaa1111.xlsx").write_bytes(b"x")
        (root / "bbbb2222.py").write_bytes(b"y")

        listing = files_module.downloaded_inventory()

        assert "aaaa1111.xlsx" in listing
        assert "bbbb2222.py" in listing

    def test_a_task_with_no_attachment_gets_nothing(self, settings, monkeypatch):
        monkeypatch.setattr(files_module, "get_settings", lambda: settings)
        (settings.download_dir).mkdir(parents=True, exist_ok=True)
        (settings.download_dir / "aaaa1111.xlsx").write_bytes(b"x")

        assert files_module.downloaded_inventory("cccc3333") == ""
