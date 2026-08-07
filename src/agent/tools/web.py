"""Web search and scraping.

Third-party clients are built lazily inside the tool. Constructing them at
import time made a missing credential crash the whole application at startup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool

from agent.config import get_settings
from agent.obs.logging import get_logger
from agent.tools.registry import ToolSpec, register

log = get_logger("tools.web")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: Strip page furniture before extracting text.
BOILERPLATE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form")


@lru_cache(maxsize=1)
def _tavily_client() -> Any | None:
    """Build the search client once, or return None when unconfigured."""
    settings = get_settings()
    if not settings.tavily_api_key:
        return None
    try:
        from tavily import TavilyClient

        return TavilyClient(api_key=settings.tavily_api_key)
    except Exception as exc:  # noqa: BLE001 - missing package or bad key
        log.error("Could not initialise Tavily client: %s", exc)
        return None


def _format_results(payload: dict[str, Any], limit: int) -> str:
    answer = (payload.get("answer") or "").strip()
    blocks = [f"Direct answer: {answer}"] if answer else []
    for item in payload.get("results", [])[:limit]:
        blocks.append(
            f"### {item.get('title', 'untitled')}\n"
            f"{item.get('url', '')}\n"
            f"{(item.get('content') or '')[:1200]}"
        )
    return "\n\n".join(blocks) if blocks else "No results found."


@tool
def web_search(query: str) -> str:
    """
    Search the web for facts, current events, or specific data.
    Input should be a concise, targeted search query.
    Returns a markdown summary of the top results.
    """
    log.info("web_search: %s", query)
    settings = get_settings()
    client = _tavily_client()
    if client is None:
        return (
            "web_search is unavailable: TAVILY_API_KEY is not configured. "
            "Use scrape_webpage on a known URL, or answer from context."
        )
    try:
        payload = client.search(
            query=query,
            max_results=settings.search_results,
            include_answer=True,
            search_depth="basic",
        )
        return _format_results(payload, settings.search_results)
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a message
        log.error("web_search failed for %r: %s", query, exc)
        return f"Search failed with error: {exc}"


@tool
def scrape_webpage(url: str) -> str:
    """
    Read the text content of a specific webpage URL.
    Use this when you need an exact article, document, or dataset behind a link.
    """
    log.info("scrape_webpage: %s", url)
    settings = get_settings()

    if not url.lower().startswith(("http://", "https://")):
        return f"Refusing to fetch non-HTTP URL: {url}"

    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=settings.scrape_timeout_s
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "xml" not in content_type:
            return str(response.text)[: settings.max_scrape_chars]

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(list(BOILERPLATE_TAGS)):
            element.extract()

        text = str(soup.get_text(separator="\n", strip=True))
        if len(text) > settings.max_scrape_chars:
            return text[: settings.max_scrape_chars] + "\n...[content truncated]"
        return text
    except requests.exceptions.Timeout:
        return f"Failed to scrape {url}: timed out after {settings.scrape_timeout_s}s."
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a message
        log.error("scrape_webpage failed for %s: %s", url, exc)
        return f"Failed to scrape URL {url}. Error: {exc}"


@tool
def wikipedia_lookup(title: str) -> str:
    """
    Fetch the plain-text extract of a Wikipedia article by title.
    Prefer this over web_search for encyclopedic facts: it is exact and citable.
    """
    log.info("wikipedia_lookup: %s", title)
    settings = get_settings()
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "redirects": "1",
                "format": "json",
                "titles": title,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=settings.scrape_timeout_s,
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
        extracts = [page["extract"] for page in pages.values() if page.get("extract", "").strip()]
        if not extracts:
            return f"No Wikipedia article found for {title!r}. Try web_search instead."
        text = "\n\n".join(extracts)
        return text[: settings.max_scrape_chars]
    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a message
        log.error("wikipedia_lookup failed for %r: %s", title, exc)
        return f"Wikipedia lookup failed: {exc}"


def _spec(name: str, tool_obj: BaseTool, capability: str, requires: tuple[str, ...]) -> ToolSpec:
    return ToolSpec(name=name, capability=capability, factory=lambda: tool_obj, requires=requires)


register(_spec("web_search", web_search, "search", ("has_search",)))
register(_spec("scrape_webpage", scrape_webpage, "scrape", ()))
register(_spec("wikipedia_lookup", wikipedia_lookup, "search", ()))
