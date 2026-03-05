#!/usr/bin/env python3
"""
ask_brain.py

RAG Q&A over your local knowledge base using:
- ./data/chunks.jsonl
- ./data/embeddings_cache.jsonl
"""

import os
import json
import math
from typing import Any, Dict, List, Tuple
from openai import OpenAI

DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
EMBED_CACHE_PATH = os.path.join(DATA_DIR, "embeddings_cache.jsonl")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def load_chunks() -> List[Dict[str, Any]]:
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(f"Missing {CHUNKS_PATH}. Run: python chunk_docs.py")
    chunks: List[Dict[str, Any]] = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("text"):
                continue
            chunks.append(d)
    if not chunks:
        raise RuntimeError("No chunks loaded.")
    return chunks


def load_embed_cache() -> Dict[str, List[float]]:
    cache: Dict[str, List[float]] = {}
    if not os.path.exists(EMBED_CACHE_PATH):
        return cache
    with open(EMBED_CACHE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cache[d["id"]] = d["embedding"]
    return cache


def embed_query(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def retrieve_top_k(client: OpenAI, query: str, chunks: List[Dict[str, Any]], cache: Dict[str, List[float]], k: int = 5):
    q = embed_query(client, query)
    scored = []
    for c in chunks:
        emb = cache.get(c["id"])
        if not emb:
            continue
        scored.append(( _cosine(q, emb), c))
    scored.sort(key=lambda x: x[0], reverse=True)
    contexts = []
    for score, c in scored[:k]:
        contexts.append({
            "id": c["id"],
            "text": c["text"],
            "source": c.get("source", "unknown"),
            "meta": c.get("meta", {}),
            "score": score
        })
    return contexts


def answer_with_citations(client: OpenAI, query: str, contexts: List[Dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        [f"[{i+1}] Source: {c['source']} | {c['text']}" for i, c in enumerate(contexts)]
    )

    system = (
        "You are a Customer Success expert. Answer using only the provided sources when possible. "
        "If you must assume, label it clearly as an assumption. "
        "Cite sources by bracket number like [1], [2]."
    )

    user = f"""Question:
{query}

Sources:
{context_text}

Instructions:
- Give a helpful, practical answer.
- Use citations like [1] when referencing sources.
"""

    resp = client.responses.create(
        model=CHAT_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.output_text.strip()


def main():
    import sys
    if len(sys.argv) < 2:
        print('Usage: python ask_brain.py "your question"')
        raise SystemExit(1)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set.")

    q = sys.argv[1]
    client = OpenAI()
    chunks = load_chunks()
    cache = load_embed_cache()
    ctx = retrieve_top_k(client, q, chunks, cache, k=5)
    ans = answer_with_citations(client, q, ctx)
    print(ans)


if __name__ == "__main__":
    main()
