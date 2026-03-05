#!/usr/bin/env python3

import os
import json
from openai import OpenAI

DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.jsonl")
EMBED_CACHE_PATH = os.path.join(DATA_DIR, "embeddings_cache.jsonl")

EMBED_MODEL = "text-embedding-3-small"


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def load_cache():
    cache = {}

    if not os.path.exists(EMBED_CACHE_PATH):
        return cache

    with open(EMBED_CACHE_PATH, "r") as f:
        for line in f:
            d = json.loads(line)
            cache[d["id"]] = d["embedding"]

    return cache


def append_cache(rows):

    with open(EMBED_CACHE_PATH, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    client = OpenAI()

    chunks = load_chunks()

    cache = load_cache()

    to_embed = [c for c in chunks if c["id"] not in cache]

    print("Chunks to embed:", len(to_embed))

    batch_size = 64

    for i in range(0, len(to_embed), batch_size):

        batch = to_embed[i:i+batch_size]

        texts = [b["text"] for b in batch]

        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )

        rows = []

        for b, emb in zip(batch, resp.data):

            rows.append({
                "id": b["id"],
                "embedding": emb.embedding
            })

        append_cache(rows)

        print("Embedded", i + len(rows))


if __name__ == "__main__":
    main()
