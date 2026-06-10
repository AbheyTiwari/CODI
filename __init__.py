# core/__init__.py


"""
scripts/ingest_data.py
Run this once (or whenever the data changes) to populate ChromaDB.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --data ./data/glossary_data.json
"""
import sys
import os
import json
import argparse

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import get_settings
from app.core.embedder import get_embedder
from app.core.vectorstore import get_vectorstore
from app.core.logging import setup_logging, logger


def main():
    setup_logging()
    cfg = get_settings()

    parser = argparse.ArgumentParser(description="Ingest glossary JSON into ChromaDB")
    parser.add_argument("--data", default=cfg.data_file, help="Path to JSON file")
    parser.add_argument("--batch", type=int, default=cfg.embed_batch_size, help="Embedding batch size")
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        logger.error("File not found: {}", data_path)
        sys.exit(1)

    logger.info("Loading data from: {}", data_path)
    with open(data_path, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    logger.info("Loaded {} raw records", len(raw))

    embedder = get_embedder()
    store    = get_vectorstore()

    ids, embeddings_list, documents, metadatas = [], [], [], []
    skipped = 0

    for i, item in enumerate(raw):
        title       = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        link        = str(item.get("link", "")).strip()

        if not title:
            logger.warning("Record {} has no title — skipping", i)
            skipped += 1
            continue

        text = title
        if description:
            text += f"\n{description}"

        ids.append(f"doc_{i}")
        documents.append(text)
        metadatas.append({"title": title, "description": description, "link": link})

    logger.info("Embedding {} documents (skipped {})…", len(ids), skipped)

    vecs = embedder.embed(documents, batch_size=args.batch)

    logger.info("Upserting into ChromaDB…")
    store.upsert(
        ids=ids,
        embeddings=vecs,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info("✅ Done — {} documents indexed, {} skipped", len(ids), skipped)
    logger.info("Collection '{}' now has {} documents", store.collection_name(), store.count())


if __name__ == "__main__":
    main()
