# core/__init__.py


"""
app/api/ingest.py
POST /ingest — (re)index the glossary JSON into ChromaDB.
Idempotent: safe to run multiple times (upsert).
"""
import json
import os
from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.core.config import get_settings
from app.core.embedder import get_embedder
from app.core.vectorstore import get_vectorstore
from app.core.logging import logger
from app.models.schemas import IngestResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])

# Simple flag so we don't run two ingests simultaneously
_ingesting = False


def _do_ingest() -> IngestResponse:
    global _ingesting
    cfg = get_settings()

    data_path = cfg.data_file
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("JSON must be an array of objects at the top level.")

    embedder = get_embedder()
    store    = get_vectorstore()

    ids, embeddings, documents, metadatas = [], [], [], []
    skipped = 0

    for i, item in enumerate(raw):
        title       = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        link        = str(item.get("link", "")).strip()

        if not title:
            skipped += 1
            continue

        # Build the text we embed — title + description
        text = title
        if description:
            text += f"\n{description}"

        ids.append(f"doc_{i}")
        documents.append(text)
        metadatas.append({
            "title":       title,
            "description": description,
            "link":        link,
        })

    logger.info("Embedding {} documents (skipped {})…", len(ids), skipped)

    vecs = embedder.embed(documents, batch_size=cfg.embed_batch_size)
    embeddings = vecs

    store.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return IngestResponse(
        status="ok",
        documents_indexed=len(ids),
        skipped=skipped,
        collection=store.collection_name(),
    )


@router.post("", response_model=IngestResponse)
async def ingest(background_tasks: BackgroundTasks) -> IngestResponse:
    """
    Reads glossary_data.json, embeds all terms, upserts into ChromaDB.
    Idempotent — safe to re-run.
    """
    global _ingesting
    if _ingesting:
        raise HTTPException(status_code=409, detail="Ingest already in progress.")

    _ingesting = True
    try:
        logger.info("Ingest started")
        result = _do_ingest()
        logger.info("Ingest done — indexed={} skipped={}", result.documents_indexed, result.skipped)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Ingest failed: {}", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}")
    finally:
        _ingesting = False
