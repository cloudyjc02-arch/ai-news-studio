"""
Custom RAG Module - uses sentence-transformers for free local embeddings
No OpenAI embeddings needed - works with any LLM backend
"""

import os
from dotenv import load_dotenv

load_dotenv()

# In-memory vector store
vector_store = []

# Lazy-load the embedding model
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("  📦 Loading local embedding model (first time only)...")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("  ✅ Embedding model loaded!")
    return _embedder


def chunk_text(text: str, chunk_size: int = 300) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - 50):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def embed_text(text: str) -> list[float]:
    """Get embedding vector using local sentence-transformers model."""
    embedder = get_embedder()
    embedding = embedder.encode(text[:2000], convert_to_numpy=True)
    return embedding.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def ingest_articles(articles: list[dict]) -> int:
    """Chunk and embed a list of articles into the vector store."""
    global vector_store
    count = 0

    for article in articles:
        full_text = f"Title: {article.get('title', '')}\n"
        full_text += f"Source: {article.get('source', '')}\n"
        full_text += f"Description: {article.get('description', '')}\n"
        full_text += f"Content: {article.get('content', '')}"

        chunks = chunk_text(full_text)
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            embedding = embed_text(chunk)
            vector_store.append({
                "text": chunk,
                "embedding": embedding,
                "source": article.get("source", "Unknown"),
                "title": article.get("title", "")
            })
            count += 1

    return count


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve the top-k most relevant chunks for a query."""
    if not vector_store:
        return "No context available."

    query_embedding = embed_text(query)

    scored = []
    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    context = "=== RETRIEVED CONTEXT FROM NEWS SOURCES ===\n\n"
    for i, (score, item) in enumerate(top, 1):
        context += f"[Chunk {i} | Source: {item['source']} | Relevance: {score:.2f}]\n"
        context += item["text"] + "\n\n"

    return context


def clear_store():
    """Clear the in-memory vector store."""
    global vector_store
    vector_store = []