"""
MCP Server 2: Publisher
Simulates a CMS. Saves articles to local JSON and generates social media posts.
Primitives: Tool (publish_article), Tool (generate_social_posts), Prompt (editorial_review_prompt)
"""

import json
import os
from datetime import datetime
from fastmcp import FastMCP

mcp = FastMCP("publisher-mcp")

PUBLISHED_FILE = "published_articles.json"


def load_published():
    if os.path.exists(PUBLISHED_FILE):
        with open(PUBLISHED_FILE, "r") as f:
            return json.load(f)
    return []


def save_published(articles):
    with open(PUBLISHED_FILE, "w") as f:
        json.dump(articles, f, indent=2)


# ---------- TOOL 1: publish_article ----------
@mcp.tool()
def publish_article(title: str, body: str, tags: list[str], author: str = "AI News Studio") -> dict:
    """Publish a finished article to the local CMS (JSON file)."""
    articles = load_published()

    article = {
        "id": len(articles) + 1,
        "title": title,
        "body": body,
        "tags": tags,
        "author": author,
        "published_at": datetime.now().isoformat(),
        "status": "published"
    }

    articles.append(article)
    save_published(articles)

    return {
        "success": True,
        "article_id": article["id"],
        "title": title,
        "published_at": article["published_at"],
        "message": f"Article '{title}' published successfully!"
    }


# ---------- TOOL 2: generate_social_posts ----------
@mcp.tool()
def generate_social_posts(title: str, summary: str, tags: list[str]) -> dict:
    """Generate Twitter and LinkedIn versions of a published article."""
    hashtags = " ".join([f"#{tag.replace(' ', '')}" for tag in tags[:3]])

    twitter = f"🗞️ {title[:100]}{'...' if len(title) > 100 else ''}\n\n{summary[:180]}{'...' if len(summary) > 180 else ''}\n\n{hashtags}"

    linkedin = f"""📰 {title}

{summary}

Key topics: {', '.join(tags)}

#News #AIGenerated #JournalismAI {hashtags}"""

    return {
        "twitter": twitter,
        "linkedin": linkedin,
        "character_count": {
            "twitter": len(twitter),
            "linkedin": len(linkedin)
        }
    }


# ---------- RESOURCE: cms://articles ----------
@mcp.resource("cms://articles")
def get_all_articles() -> str:
    """Expose all published articles as a readable resource."""
    articles = load_published()
    if not articles:
        return "No articles published yet."

    output = f"# Published Articles ({len(articles)} total)\n\n"
    for a in articles:
        output += f"## [{a['id']}] {a['title']}\n"
        output += f"Published: {a['published_at']} | Tags: {', '.join(a['tags'])}\n\n"

    return output


# ---------- PROMPT: editorial_review_prompt ----------
@mcp.prompt()
def editorial_review_prompt(article_title: str, article_body: str) -> str:
    """Structured prompt for the human-in-the-loop editorial review step."""
    return f"""You are a senior editor reviewing an AI-generated news article before publication.

ARTICLE TITLE: {article_title}

ARTICLE BODY:
{article_body}

Please evaluate this article on the following criteria:
1. ACCURACY: Does it appear factually grounded and neutral?
2. CLARITY: Is it well-written and easy to understand?
3. COMPLETENESS: Does it cover the topic adequately?
4. QUALITY SCORE: Rate from 1-10.

Provide your review in this format:
ACCURACY: [assessment]
CLARITY: [assessment]  
COMPLETENESS: [assessment]
SCORE: [1-10]
RECOMMENDATION: [APPROVE / REVISE / REJECT]
NOTES: [any specific feedback]
"""


if __name__ == "__main__":
    mcp.run(transport="stdio")