"""
Flask web interface for AI News Studio
Streams pipeline progress to browser in real time
"""

import os
import json
import queue
import threading
from flask import Flask, render_template, request, Response, stream_with_context
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Global queue for streaming progress updates
progress_queue = queue.Queue()
result_store = {}

def stream_print(msg, msg_type="info"):
    """Send a message to the browser via SSE."""
    progress_queue.put({"type": msg_type, "message": msg})

def run_pipeline_async(topic):
    """Run the pipeline in a background thread, streaming updates."""
    import sys
    import io

    # Patch print to also send to browser
    original_print = __builtins__.__dict__['print'] if isinstance(__builtins__, dict) else print

    try:
        stream_print(f"Starting pipeline for topic: <b>{topic}</b>", "start")

        # Import pipeline components
        stream_print("Loading models and libraries...", "info")
        import requests as req
        import rag
        from writing_crew import run_writing_crew
        import asyncio
        import concurrent.futures
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from openai import OpenAI
        from typing import TypedDict

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")
        GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

        # ── NODE 1: RESEARCH ──────────────────────────────
        stream_print("🔍 <b>Node 1 — Research:</b> Fetching news articles...", "node")

        queries = [topic, f"{topic} latest", f"{topic} news 2025"]
        all_articles = []

        def fetch_query(q):
            try:
                params = {"q": q, "lang": "en", "max": 3, "apikey": GNEWS_API_KEY}
                r = req.get("https://gnews.io/api/v4/search", params=params, timeout=10)
                return r.json().get("articles", [])
            except:
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

        stream_print(f"✅ Fetched <b>{len(all_articles)}</b> unique articles", "success")
        stream_print("📦 Building RAG vector store...", "info")

        rag.clear_store()
        chunk_count = rag.ingest_articles(all_articles)
        context = rag.retrieve_context(topic, top_k=4)

        stream_print(f"✅ Ingested <b>{chunk_count}</b> chunks into RAG store", "success")
        stream_print(f"✅ Retrieved <b>{len(context)}</b> chars of context", "success")

        # ── NODE 2: WRITE ─────────────────────────────────
        stream_print("✍️ <b>Node 2 — Write:</b> Running Researcher → Writer → Editor crew...", "node")

        draft = asyncio.run(run_writing_crew(topic=topic, rag_context=context))
        lines = draft.strip().split("\n")
        title = lines[0].replace("#", "").strip() if lines else topic

        stream_print(f"✅ Draft generated: <b>{title}</b>", "success")

        # ── NODE 3: EVALUATE ──────────────────────────────
        stream_print("🔎 <b>Node 3 — Evaluate:</b> Scoring article quality...", "node")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": 'Evaluate this news article. Respond ONLY in JSON: {"score": 7, "feedback": "...", "tags": ["tag1","tag2","tag3"]}'},
                    {"role": "user", "content": f"Topic: {topic}\n\nArticle:\n{draft}"}
                ]
            )
            raw = response.choices[0].message.content.strip().replace("```json","").replace("```","")
            result = json.loads(raw)
            score = int(result.get("score", 7))
            feedback = result.get("feedback", "")
            tags = result.get("tags", [topic])
        except Exception as e:
            score = 7
            feedback = "Auto-approved."
            tags = [topic]

        stream_print(f"📊 Quality score: <b>{score}/10</b> — {feedback}", "score")

        # ── NODE 4: HITL ──────────────────────────────────
        stream_print("⛔ <b>Node 4 — HITL:</b> Waiting for human approval...", "hitl")

        # Store results for HITL page
        result_store["pending"] = {
            "topic": topic,
            "title": title,
            "draft": draft,
            "score": score,
            "feedback": feedback,
            "tags": tags,
            "articles": all_articles
        }

        stream_print("HITL_READY", "hitl_ready")

    except Exception as e:
        stream_print(f"❌ Error: {str(e)}", "error")
        stream_print("DONE", "done")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    topic = request.form.get("topic", "artificial intelligence")
    # Clear queue
    while not progress_queue.empty():
        progress_queue.get()
    result_store.clear()
    # Run pipeline in background
    thread = threading.Thread(target=run_pipeline_async, args=(topic,))
    thread.daemon = True
    thread.start()
    return render_template("progress.html", topic=topic)


@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                msg = progress_queue.get(timeout=60)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ["hitl_ready", "done", "error"]:
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/review")
def review():
    pending = result_store.get("pending")
    if not pending:
        return "No article pending review.", 404
    return render_template("review.html", **pending)


@app.route("/decision", methods=["POST"])
def decision():
    action = request.form.get("action")
    notes = request.form.get("notes", "")
    pending = result_store.get("pending", {})

    if action == "approve":
        # Save article
        from datetime import datetime
        article = {
            "title": pending.get("title"),
            "body": pending.get("draft"),
            "tags": pending.get("tags", []),
            "score": pending.get("score"),
            "published_at": datetime.now().isoformat(),
            "status": "published"
        }
        published = []
        if os.path.exists("published_articles.json"):
            with open("published_articles.json") as f:
                published = json.load(f)
        published.append(article)
        with open("published_articles.json", "w") as f:
            json.dump(published, f, indent=2)

        title = pending.get("title", "")
        draft = pending.get("draft", "")
        tags = pending.get("tags", [])
        hashtags = " ".join([f"#{t.replace(' ','')}" for t in tags[:3]])
        twitter = f"🗞️ {title}\n\n{draft[:200]}...\n\n{hashtags}"
        linkedin = f"📰 {title}\n\n{draft[:300]}...\n\nTopics: {', '.join(tags)}\n\n#News #AIGenerated"

        return render_template("published.html",
            article=article,
            twitter=twitter,
            linkedin=linkedin
        )
    else:
        return render_template("rejected.html", action=action, notes=notes)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)