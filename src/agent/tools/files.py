"""Attachment handling.

Many GAIA tasks ship a file (spreadsheet, source code, audio, image) that the
question is meaningless without. The benchmark serves it from
``GET {scoring_api}/files/{task_id}``; these tools fetch it and turn it into
text the model can reason over.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import requests
from langchain_core.tools import BaseTool, tool

from agent.config import get_settings
from agent.obs.logging import get_logger
from agent.tools.registry import ToolSpec, register

log = get_logger("tools.files")

#: The benchmark's 20 tasks are drawn from this gated dataset; its files are the
#: only working source while the scoring API's /files route 404s.
GAIA_DATASET = "gaia-benchmark/GAIA"
GAIA_SPLIT = "2023/validation"

#: Named in the pushed inventory line so the model knows what to do with it.
READ_TOOL_NAME = "read_file"

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


def _existing_download(task_id: str) -> Path | None:
    """A previously fetched attachment for this task, if any."""
    try:
        matches = sorted(p for p in _download_dir().glob(f"{task_id}*") if p.is_file())
    except OSError:
        return None
    return matches[0] if matches else None


def downloaded_inventory() -> str:
    """Attachments already fetched, as a line to push into a specialist's context.

    Pushed rather than left to ``list_downloaded_files``: that tool has been
    bound to every file-capable specialist from the start and called zero times
    across 92 downloads. A tool the model must choose to call cannot fix a
    failure caused by the model not choosing to call things.
    """
    try:
        entries = sorted(p for p in _download_dir().iterdir() if p.is_file())
    except OSError:
        return ""
    if not entries:
        return ""
    listing = ", ".join(f"{p.name} ({p.stat().st_size} bytes)" for p in entries)
    return f"Files already downloaded and ready for {READ_TOOL_NAME}: {listing}"


def _from_scoring_api(task_id: str) -> tuple[bytes, str] | None:
    """Attachment bytes and suffix from the benchmark's own endpoint."""
    settings = get_settings()
    url = f"{settings.scoring_api_url}/files/{task_id}"
    try:
        response = requests.get(url, timeout=settings.scrape_timeout_s)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - a miss here is expected; we fall back
        log.info("scoring API has no file for %s (%s)", task_id, exc)
        return None

    suffix = ""
    disposition = response.headers.get("content-disposition", "")
    if "filename=" in disposition:
        suffix = Path(disposition.split("filename=")[-1].strip('"; ')).suffix
    return response.content, suffix


@lru_cache(maxsize=1)
def _dataset_index() -> dict[str, str]:
    """task_id -> path within the GAIA dataset, or empty when unreachable.

    Cached: one listing serves every task in a run.
    """
    settings = get_settings()
    if not settings.hf_token:
        return {}

    url = f"https://huggingface.co/api/datasets/{GAIA_DATASET}/tree/main/{GAIA_SPLIT}"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            timeout=settings.scrape_timeout_s,
        )
        response.raise_for_status()
        entries = response.json()
    except Exception as exc:  # noqa: BLE001 - degrade to "no attachments available"
        log.warning("GAIA dataset listing unavailable: %s", exc)
        return {}

    index = {Path(str(e.get("path", ""))).stem: str(e.get("path", "")) for e in entries}
    log.info("GAIA dataset index: %d attachments", len(index))
    return index


def _from_dataset(task_id: str) -> tuple[bytes, str] | None:
    """Attachment bytes and suffix from the gated GAIA dataset on the Hub."""
    path = _dataset_index().get(task_id)
    if not path:
        return None

    settings = get_settings()
    url = f"https://huggingface.co/datasets/{GAIA_DATASET}/resolve/main/{path}"
    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {settings.hf_token}"},
            timeout=settings.scrape_timeout_s,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a message
        log.error("dataset fetch failed for %s: %s", task_id, exc)
        return None
    return response.content, Path(path).suffix


@tool
def download_task_file(task_id: str) -> str:
    """
    Download the file attached to a benchmark task and save it locally.
    Call this first whenever the question mentions an attached file.
    Returns the local path, which you then pass to read_file.
    """
    log.info("download_task_file: %s", task_id)

    # Memoised on disk. The supervisor re-delegates freely and each delegation
    # gives the specialist a fresh state, so one task fetched the same
    # spreadsheet four times - 83 seconds and a rate limit for one file.
    cached = _existing_download(task_id)
    if cached is not None:
        size = cached.stat().st_size
        log.info("already downloaded: %s", cached)
        return f"Already downloaded to {cached} ({size} bytes). Now call read_file on it."

    # The scoring API is authoritative but currently returns 404 for every
    # attachment task ("No file path associated with task_id"), so the gated
    # GAIA dataset is the working source. Order kept in case it is restored.
    payload = _from_scoring_api(task_id) or _from_dataset(task_id)
    if payload is None:
        return (
            f"No file is available for task {task_id}. The scoring API has no mapping "
            f"for it, and the GAIA dataset is unreachable - that needs HF_TOKEN set and "
            f"the dataset terms accepted at huggingface.co/datasets/gaia-benchmark/GAIA."
        )

    content, suffix = payload
    destination = _download_dir() / f"{task_id}{suffix}"
    destination.write_bytes(content)
    log.info("saved %d bytes -> %s", len(content), destination)
    return f"Downloaded to {destination} ({len(content)} bytes). Now call read_file on it."


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
