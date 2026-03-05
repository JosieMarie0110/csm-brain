#!/usr/bin/env python3
"""
chunk_docs.py

Reads PDFs from ./pdfs, extracts text, and writes chunks to:
  ./data/chunks.jsonl

Chunking is simple and reliable:
- split by paragraphs
- pack into ~1200-1600 char chunks
"""

import os
import json
from typing import List, Dict
from pypdf import PdfReader

PDF_DIR = "pdfs"
DATA_DIR = "data"
OUT_PATH = os.path.join(DATA_DIR, "chunks.jsonl")

TARGET_CHARS = 1400
MIN_CHARS = 400


def extract_pdf_text(pdf_path: str) -> List[Dict]:
    reader = PdfReader(pdf_path)
    items = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.replace("\r", "\n").strip()
        if text:
            items.append({"page": i, "text": text})
    return items


def split_paragraphs(text: str) -> List[str]:
    # normalize newlines
    text = "\n".join([ln.rstrip() for ln in text.split("\n")])
    # split on blank lines
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts


def pack_chunks(paragraphs: List[str]) -> List[str]:
    chunks = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
            continue
        # +2 for spacing
        if len(buf) + 2 + len(p) <= TARGET_CHARS:
            buf = buf + "\n\n" + p
        else:
            if len(buf) >= MIN_CHARS:
                chunks.append(buf)
            buf = p

    if buf and len(buf) >= MIN_CHARS:
        chunks.append(buf)

    # If everything was tiny, still return something
    if not chunks and paragraphs:
        chunks = ["\n\n".join(paragraphs)]
    return chunks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        raise SystemExit(f"No PDFs found in ./{PDF_DIR}")

    out_count = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for pdf_name in sorted(pdfs):
            pdf_path = os.path.join(PDF_DIR, pdf_name)
            pages = extract_pdf_text(pdf_path)
            for page_item in pages:
                paras = split_paragraphs(page_item["text"])
                chunks = pack_chunks(paras)
                for j, chunk in enumerate(chunks):
                    rec = {
                        "id": f"{pdf_name}::p{page_item['page']}::c{j}",
                        "text": chunk,
                        "source": pdf_name,
                        "meta": {"page": page_item["page"]},
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_count += 1

    print(f"Wrote {out_count} chunks → {OUT_PATH}")


if __name__ == "__main__":
    main()
