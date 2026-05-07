"""
PHAI - ingest curated knowledge-base snippets into ChromaDB.

Run from the project root:
    python kb/ingest.py

Reads kb/snippets.json, embeds each snippet with sentence-transformers
(all-MiniLM-L6-v2, ~80 MB, free, runs on CPU), stores in a persistent
ChromaDB collection at chroma_db/.

Idempotent: deletes the collection first if it exists. Re-run any time
you edit snippets.json.

Includes a smoke test at the end - runs sample queries and prints the
top retrievals so you can eyeball that the embeddings make sense.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "kb"
SNIPPETS_PATH = KB_DIR / "snippets.json"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "phai_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Sample queries used to smoke-test retrieval at the end.
SMOKE_TESTS = [
    "I want to feel more energetic during the week",
    "my caffeine intake might be affecting my sleep",
    "what should I eat to lose weight and feel full",
    "how do I improve my sleep quality",
    "I have a slow caffeine metabolism, what should I do",
    "tips for managing stress and anxiety",
    "what exercise is best for my body type",
]


def main() -> None:
    if not SNIPPETS_PATH.exists():
        raise FileNotFoundError(f"snippets.json not found at {SNIPPETS_PATH}")

    print(f"Loading snippets from {SNIPPETS_PATH.name}...")
    snippets = json.loads(SNIPPETS_PATH.read_text(encoding="utf-8"))
    print(f"  {len(snippets)} snippets")

    # Sanity: counts by category
    cats: dict[str, int] = {}
    for s in snippets:
        cats[s["category"]] = cats.get(s["category"], 0) + 1
    print("  by category: " + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))

    print(f"\nInitialising ChromaDB at {CHROMA_DIR}...")
    CHROMA_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop existing collection for a clean re-index.
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  dropped existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    print(f"\nLoading embedder: {EMBEDDING_MODEL}")
    print("  (first run downloads ~80 MB; subsequent runs load from cache)")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    print(f"\nEmbedding {len(snippets)} snippets...")
    collection.add(
        ids=[s["id"] for s in snippets],
        documents=[s["text"] for s in snippets],
        metadatas=[
            {
                "category": s["category"],
                "topic": s["topic"],
                "source": s.get("source", ""),
                "url": s.get("url", ""),
                "tags": ",".join(s.get("tags", [])),
            }
            for s in snippets
        ],
    )
    n = collection.count()
    print(f"  done. Collection now has {n} items.")

    # ----- Smoke test -----
    print("\n=== Smoke test (top-3 retrieval per query) ===")
    for q in SMOKE_TESTS:
        print(f"\nQ: {q}")
        results = collection.query(query_texts=[q], n_results=3)
        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            preview = doc[:90].replace("\n", " ") + ("..." if len(doc) > 90 else "")
            print(f"  {i + 1}. [{meta['topic']}, dist={dist:.3f}]  {preview}")

    print("\nKnowledge base ingestion complete.")


if __name__ == "__main__":
    main()
