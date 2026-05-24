"""
MCP Server 1: News Fetcher
Provides tools and resources to fetch real news articles via GNews API.
Primitives: Resource (news://topic), Tool (search_news), Prompt (summarize_for_rag)
"""

import os
import requests
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("news-fetcher-mcp")

GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
GNEWS_URL = "https://gnews.io/api/v4/search"

# ---------- TOOL 1: search_news ----------
@mcp.tool()
def search_news(query: str, max_results: int = 5) -> dict:
    """Search for recent news articles on a given topic using GNews API."""
    try:
        params = {
            "q": query,
            "lang": "en",
            "max": max_results,
            "apikey": GNEWS_API_KEY
        }
        response = requests.get(GNEWS_URL, params=params, timeout=10)
        data = response.json()

        if "articles" not in data:
            return {"error": "No articles found", "raw": data}

        articles = []
        for a in data["articles"]:
            articles.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "content": a.get("content", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "Unknown"),
                "publishedAt": a.get("publishedAt", "")
            })

        return {"query": query, "count": len(articles), "articles": articles}

    except Exception as e:
        return {"error": str(e)}


# ---------- TOOL 2: get_top_headlines ----------
@mcp.tool()
def get_top_headlines(category: str = "general", max_results: int = 5) -> dict:
    """Get top headlines by category (general, technology, science, health, etc.)"""
    try:
        params = {
            "q": category,
            "lang": "en",
            "max": max_results,
            "apikey": GNEWS_API_KEY
        }
        response = requests.get(GNEWS_URL, params=params, timeout=10)
        data = response.json()

        if "articles" not in data:
            return {"error": "No headlines found", "raw": data}

        headlines = []
        for a in data["articles"]:
            headlines.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "source": a.get("source", {}).get("name", "Unknown"),
                "url": a.get("url", "")
            })

        return {"category": category, "count": len(headlines), "headlines": headlines}

    except Exception as e:
        return {"error": str(e)}


# ---------- RESOURCE: news://topic/{keyword} ----------
@mcp.resource("news://topic/{keyword}")
def get_news_resource(keyword: str) -> str:
    """Expose news articles as a readable MCP resource."""
    result = search_news(keyword, max_results=3)
    if "error" in result:
        return f"Error fetching news: {result['error']}"

    output = f"# News Articles: {keyword}\n\n"
    for i, article in enumerate(result["articles"], 1):
        output += f"## Article {i}: {article['title']}\n"
        output += f"**Source:** {article['source']}\n"
        output += f"**Published:** {article['publishedAt']}\n"
        output += f"**Summary:** {article['description']}\n"
        output += f"**Content:** {article['content']}\n"
        output += f"**URL:** {article['url']}\n\n"

    return output


# ---------- PROMPT: summarize_for_rag ----------
@mcp.prompt()
def summarize_for_rag(topic: str, article_text: str) -> str:
    """Reusable prompt template for chunking and summarizing articles for RAG."""
    return f"""You are a news research assistant preparing content for a RAG pipeline.

Topic: {topic}

Article Text:
{article_text}

Your task:
1. Extract the 3 most important facts from this article relevant to the topic.
2. Write a 2-sentence neutral summary.
3. List any key people, organizations, or locations mentioned.

Format your response as:
FACTS:
- fact 1
- fact 2  
- fact 3

SUMMARY:
[2 sentence summary]

ENTITIES:
[comma separated list]
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")