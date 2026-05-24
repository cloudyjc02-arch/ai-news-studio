"""
LangGraph Orchestrator - AI News Studio
Main pipeline: Research → Write → Evaluate → HITL → Publish

Patterns used:
- Orchestrator-Workers (LangGraph → OpenAI Agents crew)
- Evaluator-Optimizer (quality loop, rewrites if score < 7)
- Human-in-the-Loop (HITL checkpoint)
- Parallelization (parallel news fetch)
- A2A (LangGraph delegates to OpenAI Agents SDK crew)
"""

import os
import asyncio
import json
import requests
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI

import rag
from writing_crew import run_writing_crew

load_dotenv()

# Use school's custom API
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


class NewsState(TypedDict):
    topic: str
    raw_articles: list[dict]
    rag_context: str
    draft_article: str
    quality_score: int
    quality_feedback: str
    rewrite_count: int
    human_decision: str
    human_notes: str
    final_article: str
    social_posts: dict
    published: bool
    article_title: str
    article_tags: list[str]


def research_node(state: NewsState) -> NewsState:
    print(f"\n{'='*60}")
    print(f"🔍 [NODE 1] RESEARCH: Fetching news on '{state['topic']}'")
    print(f"{'='*60}")

    topic = state["topic"]
    all_articles = []
    queries = [topic, f"{topic} latest", f"{topic} news 2025"]

    import concurrent.futures

    def fetch_query(q):
        try:
            params = {"q": q, "lang": "en", "max": 3, "apikey": GNEWS_API_KEY}
            r = requests.get("https://gnews.io/api/v4/search", params=params, timeout=10)
            data = r.json()
            return data.get("articles", [])
        except Exception as e:
            print(f"  ⚠️  Fetch error for '{q}': {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(fetch_query, queries))

    seen_titles = set()
    for articles in results:
        for a in articles:
            title = a.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                all_articles.append({
                    "title": title,
                    "description": a.get("description", ""),
                    "content": a.get("content", ""),
                    "source": a.get("source", {}).get("name", "Unknown"),
                    "url": a.get("url", ""),
                    "publishedAt": a.get("publishedAt", "")
                })

    print(f"  ✅ Fetched {len(all_articles)} unique articles")

    rag.clear_store()
    chunk_count = rag.ingest_articles(all_articles)
    print(f"  ✅ Ingested {chunk_count} chunks into RAG vector store")

    context = rag.retrieve_context(topic, top_k=4)
    print(f"  ✅ Retrieved RAG context ({len(context)} chars)")

    return {**state, "raw_articles": all_articles, "rag_context": context, "rewrite_count": 0}


def write_node(state: NewsState) -> NewsState:
    print(f"\n{'='*60}")
    print(f"✍️  [NODE 2] WRITING: Running Researcher → Writer → Editor crew")
    print(f"{'='*60}")

    extra = ""
    if state.get("quality_feedback") and state.get("rewrite_count", 0) > 0:
        extra = f"REWRITE REQUESTED. Previous feedback: {state['quality_feedback']}"
        print(f"  🔄 Rewrite #{state['rewrite_count']} based on feedback")

    draft = asyncio.run(run_writing_crew(
        topic=state["topic"],
        rag_context=state["rag_context"],
        extra_notes=extra
    ))

    print(f"  ✅ Draft article generated ({len(draft)} chars)")
    print(f"\n  📄 DRAFT PREVIEW:\n  {draft[:300]}...\n")

    lines = draft.strip().split("\n")
    title = lines[0].replace("#", "").strip() if lines else state["topic"]

    return {**state, "draft_article": draft, "article_title": title}


def evaluate_node(state: NewsState) -> NewsState:
    print(f"\n{'='*60}")
    print(f"🔎 [NODE 3] EVALUATING article quality...")
    print(f"{'='*60}")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a strict news quality evaluator.
Score articles 1-10 and provide specific feedback.
Respond ONLY in this exact JSON format:
{"score": 7, "feedback": "specific feedback here", "tags": ["tag1", "tag2", "tag3"]}"""
                },
                {
                    "role": "user",
                    "content": f"""Evaluate this news article for quality, accuracy, clarity and completeness.

Topic: {state['topic']}

Article:
{state['draft_article']}

Return JSON with score (1-10), feedback, and 3 relevant tags."""
                }
            ]
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        score = int(result.get("score", 5))
        feedback = result.get("feedback", "")
        tags = result.get("tags", [state["topic"]])
    except Exception as e:
        print(f"  ⚠️  Evaluator error: {e}. Defaulting score to 7.")
        score = 7
        feedback = "Auto-approved due to evaluator error."
        tags = [state["topic"]]

    print(f"  📊 Quality Score: {score}/10")
    print(f"  💬 Feedback: {feedback}")

    return {**state, "quality_score": score, "quality_feedback": feedback, "article_tags": tags}


def should_rewrite(state: NewsState) -> str:
    score = state.get("quality_score", 0)
    rewrite_count = state.get("rewrite_count", 0)

    if score < 7 and rewrite_count < 2:
        print(f"\n  🔄 Score {score}/10 is below 7 — triggering rewrite #{rewrite_count + 1}")
        return "rewrite"
    else:
        print(f"\n  ✅ Score {score}/10 — proceeding to HITL")
        return "proceed"


def increment_rewrite(state: NewsState) -> NewsState:
    return {**state, "rewrite_count": state.get("rewrite_count", 0) + 1}


def hitl_node(state: NewsState) -> NewsState:
    print(f"\n{'='*60}")
    print(f"🧑 [NODE 4] HUMAN-IN-THE-LOOP — Editor Review Required")
    print(f"{'='*60}")
    print(f"\n📰 ARTICLE TITLE: {state.get('article_title', 'Untitled')}")
    print(f"📊 Quality Score: {state.get('quality_score', 'N/A')}/10")
    print(f"🏷️  Tags: {', '.join(state.get('article_tags', []))}")
    print(f"\n{'─'*60}")
    print(state["draft_article"])
    print(f"{'─'*60}")
    print(f"\n💬 Evaluator Feedback: {state.get('quality_feedback', '')}")
    print(f"\n⚠️  HUMAN DECISION REQUIRED")
    print("Options: [approve] [revise] [reject]")

    while True:
        decision = input("\n👉 Your decision: ").strip().lower()
        if decision in ["approve", "revise", "reject"]:
            break
        print("  Please type exactly: approve, revise, or reject")

    notes = ""
    if decision == "revise":
        notes = input("📝 Enter your revision notes: ").strip()
    elif decision == "reject":
        notes = input("📝 Reason for rejection: ").strip()

    print(f"\n  ✅ Decision recorded: {decision.upper()}")
    return {**state, "human_decision": decision, "human_notes": notes}


def hitl_router(state: NewsState) -> str:
    decision = state.get("human_decision", "reject")
    if decision == "approve":
        return "publish"
    elif decision == "revise":
        return "rewrite"
    else:
        return "end"


def apply_human_revision(state: NewsState) -> NewsState:
    return {
        **state,
        "quality_feedback": f"Human editor requested: {state.get('human_notes', '')}",
        "rewrite_count": 0
    }


def publish_node(state: NewsState) -> NewsState:
    print(f"\n{'='*60}")
    print(f"🚀 [NODE 5] PUBLISHING article...")
    print(f"{'='*60}")

    import json as json_module
    from datetime import datetime

    article = {
        "id": 1,
        "title": state.get("article_title", "Untitled"),
        "body": state["draft_article"],
        "tags": state.get("article_tags", []),
        "quality_score": state.get("quality_score", 0),
        "published_at": datetime.now().isoformat(),
        "status": "published"
    }

    published_file = "published_articles.json"
    existing = []
    if os.path.exists(published_file):
        with open(published_file) as f:
            existing = json_module.load(f)

    existing.append(article)
    with open(published_file, "w") as f:
        json_module.dump(existing, f, indent=2)

    title = state.get("article_title", "")
    summary = state["draft_article"][:200]
    tags = state.get("article_tags", [])
    hashtags = " ".join([f"#{t.replace(' ', '')}" for t in tags[:3]])

    twitter = f"🗞️ {title}\n\n{summary}...\n\n{hashtags}"
    linkedin = f"📰 {title}\n\n{summary}...\n\nKey topics: {', '.join(tags)}\n\n#News #AIGenerated"
    social = {"twitter": twitter, "linkedin": linkedin}

    print(f"  ✅ Article published to '{published_file}'")
    print(f"\n  📱 SOCIAL MEDIA POSTS GENERATED:")
    print(f"\n  🐦 Twitter:\n  {twitter[:200]}")
    print(f"\n  💼 LinkedIn:\n  {linkedin[:200]}")

    return {**state, "final_article": state["draft_article"], "social_posts": social, "published": True}


def end_node(state: NewsState) -> NewsState:
    print(f"\n{'='*60}")
    print(f"❌ Article rejected. Pipeline ended.")
    print(f"{'='*60}")
    return {**state, "published": False}


def build_graph():
    memory = MemorySaver()
    graph = StateGraph(NewsState)

    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("increment_rewrite", increment_rewrite)
    graph.add_node("hitl", hitl_node)
    graph.add_node("apply_human_revision", apply_human_revision)
    graph.add_node("publish", publish_node)
    graph.add_node("end", end_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "write")
    graph.add_edge("write", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        should_rewrite,
        {"rewrite": "increment_rewrite", "proceed": "hitl"}
    )
    graph.add_edge("increment_rewrite", "write")

    graph.add_conditional_edges(
        "hitl",
        hitl_router,
        {"publish": "publish", "rewrite": "apply_human_revision", "end": "end"}
    )
    graph.add_edge("apply_human_revision", "write")
    graph.add_edge("publish", END)
    graph.add_edge("end", END)

    return graph.compile(checkpointer=memory)


def run_pipeline(topic: str):
    print(f"\n{'*'*60}")
    print(f"  🎬 AI NEWS STUDIO — Multi-Agent Pipeline")
    print(f"  Topic: {topic}")
    print(f"{'*'*60}")

    app = build_graph()

    initial_state: NewsState = {
        "topic": topic,
        "raw_articles": [],
        "rag_context": "",
        "draft_article": "",
        "quality_score": 0,
        "quality_feedback": "",
        "rewrite_count": 0,
        "human_decision": "",
        "human_notes": "",
        "final_article": "",
        "social_posts": {},
        "published": False,
        "article_title": "",
        "article_tags": []
    }

    config = {"configurable": {"thread_id": "news-session-1"}}
    final = app.invoke(initial_state, config=config)

    print(f"\n{'*'*60}")
    if final.get("published"):
        print(f"  ✅ PIPELINE COMPLETE — Article Published!")
        print(f"  Title: {final.get('article_title')}")
        print(f"  Score: {final.get('quality_score')}/10")
    else:
        print(f"  ❌ PIPELINE ENDED — Article not published")
    print(f"{'*'*60}\n")

    return final


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "artificial intelligence"
    run_pipeline(topic)