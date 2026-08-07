"""Attachment handling.

Many GAIA tasks ship a file (spreadsheet, source code, audio, image) that the
question is meaningless without. The benchmark serves it from
``GET {scoring_api}/files/{task_id}``; these tools fetch it and turn it into
text the model can reason over.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import requests
from langchain_core.tools import BaseTool, tool

from agent.config import get_settings
from agent.obs.logging import get_logger
from agent.tools.registry import ToolSpec, register

log = get_logger("tools.files")

TEXT_SUFFIXES = frozenset(
    {".txt", ".md", ".py", ".json", ".jsonl", ".xml", ".html", ".csv", ".tsv"}
)
TABULAR_SUFFIXES = frozenset({".csv", ".tsv", ".xlsx", ".xls"})
BINARY_HINTS = {
    ".mp3": "audio file - use transcribe_audio",
    ".wav": "audio file - use transcribe_audio",
    ".m4a": "audio file - use transcribe_audio",
    ".png": "image file - use describe_image",
    ".jpg": "image file - use describe_image",
    ".jpeg": "image file - use describe_image",
    ".pdf": "PDF - text extraction not yet installed",
}


def _download_dir() -> Path:
    target = get_settings().download_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resolve(path: str) -> Path | None:
    """Resolve a model-supplied path, refusing anything outside the download dir."""
    root = _download_dir().resolve()
    candidate = (root / Path(path).name).resolve()
    if candidate.parent != root or not candidate.exists():
        return None
    return candidate


@tool
def download_task_file(task_id: str) -> str:
    """
    Download the file attached to a benchmark task and save it locally.
    Call this first whenever the question mentions an attached file.
    Returns the local path, which you then pass to read_file.
    """
    settings = get_settings()
    url = f"{settings.scoring_api_url}/files/{task_id}"
    log.info("download_task_file: %s", url)

    try:
        response = requests.get(url, timeout=settings.scrape_timeout_s)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a message
        log.error("download failed for %s: %s", task_id, exc)
        return f"Could not download the file for task {task_id}: {exc}"

    suffix = ""
    disposition = response.headers.get("content-disposition", "")
    if "filename=" in disposition:
        suffix = Path(disposition.split("filename=")[-1].strip('"; ')).suffix

    destination = _download_dir() / f"{task_id}{suffix}"
    destination.write_bytes(response.content)
    log.info("saved %d bytes -> %s", len(response.content), destination)
    return f"Downloaded to {destination} ({len(response.content)} bytes). Now call read_file on it."


def _read_tabular(path: Path, limit: int) -> str:
    try:
        import pandas as pd  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - pandas is an app extra
        return f"Cannot parse {path.name}: pandas is not installed."

    try:
        frame = (
            pd.read_excel(path)
            if path.suffix in {".xlsx", ".xls"}
            else pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a message
        return f"Could not parse {path.name}: {exc}"

    header = (
        f"{path.name}: {len(frame)} rows x {len(frame.columns)} columns\n"
        f"Columns: {list(frame.columns)}\n\n"
    )
    return str(header + frame.to_string(max_rows=200))[:limit]


def _read_text(path: Path, limit: int) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        return text[:limit] + "\n...[content truncated]"
    return text


@tool
def read_file(path: str) -> str:
    """
    Read a downloaded file and return its contents as text.
    Spreadsheets and CSVs come back as a table summary; source code and text
    come back verbatim. Use download_task_file first to obtain the path.
    """
    log.info("read_file: %s", path)
    settings = get_settings()

    resolved = _resolve(path)
    if resolved is None:
        available = [p.name for p in _download_dir().iterdir()] or ["(none)"]
        return f"No such downloaded file: {path}. Available: {available}"

    suffix = resolved.suffix.lower()
    if suffix in TABULAR_SUFFIXES:
        return _read_tabular(resolved, settings.max_file_chars)
    if suffix in TEXT_SUFFIXES or not suffix:
        return _read_text(resolved, settings.max_file_chars)
    if suffix in BINARY_HINTS:
        return f"{resolved.name} is a {BINARY_HINTS[suffix]}."
    return f"Unsupported file type {suffix!r} for {resolved.name}."


@tool
def list_downloaded_files() -> str:
    """List files already downloaded during this run, with their sizes."""
    entries = [
        {"name": p.name, "bytes": p.stat().st_size} for p in sorted(_download_dir().iterdir())
    ]
    return json.dumps(entries) if entries else "No files downloaded yet."


def _spec(name: str, tool_obj: BaseTool, capability: str) -> ToolSpec:
    factory: Callable[[], BaseTool] = lambda: tool_obj  # noqa: E731
    return ToolSpec(name=name, capability=capability, factory=factory)


register(_spec("download_task_file", download_task_file, "files"))
register(_spec("read_file", read_file, "files"))
register(_spec("list_downloaded_files", list_downloaded_files, "files"))
