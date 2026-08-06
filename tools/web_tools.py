# tools/web_tools.py
import os
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


# Init built-in LangChain Tavily tool
tavily_search_tool = TavilySearchResults(max_results=3)

@tool
def web_search(query: str) -> str:
    """
    Searches the web for facts, current events, or specific data.
    Input should be a concise, targeted search query.
    Returns a markdown-formatted summary of the top search results.
    """
    print(f"\n[Web Tool]: Searching for -> {query}")
    try:
        results = tavily_search_tool.invoke({"query": query})
        return str(results)
    except Exception as e:
        return f"Search failed with error: {str(e)}"


@tool
def scrape_webpage(url: str) -> str:
    """
    Scrapes the text content of a specific webpage URL.
    Use this when you need to read an exact article, document, or dataset link.
    """
    print(f"\n[Web Tool]: Scraping URL -> {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML and extract clean text
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator="\n", strip=True)
        
        # Truncate to avoid context window explosion (adjust length as needed)
        max_chars = 15000
        if len(text) > max_chars:
            return text[:max_chars] + "\n...[Content truncated due to length]"
            
        return text
    except Exception as e:
        return f"Failed to scrape URL {url}. Error: {str(e)}"

# Export a convenient list of tools to pass directly to your agent
web_tools_list = [web_search, scrape_webpage]
