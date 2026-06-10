# core/__init__.py



"""
app/core/vectorstore.py
All ChromaDB operations — one place, no leaking chroma details elsewhere.
"""
from __future__ import annotations
import chromadb
from chromadb.config import Settings as ChromaSettings
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import SourceDoc


class VectorStore:
    def __init__(self) -> None:
        cfg = get_settings()
        self._client = chromadb.PersistentClient(
            path=cfg.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=cfg.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready — collection='{}' docs={}",
            cfg.chroma_collection,
            self._collection.count(),
        )

    # ── Write ──────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
        batch_size: int = 500,
    ) -> None:
        """Upsert in batches so memory stays bounded."""
        total = len(ids)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )
            logger.debug("Upserted batch {}-{} / {}", start, end, total)
        logger.info("Upsert complete — {} documents", total)

    # ── Read ───────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[SourceDoc]:
        """Return top-k results above score_threshold."""
        if self._collection.count() == 0:
            logger.warning("Collection is empty — run ingest first")
            return []

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        sources: list[SourceDoc] = []
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # Chroma cosine distance → similarity score (0–1)
            score = float(1 - dist)
            if score < score_threshold:
                continue
            sources.append(
                SourceDoc(
                    title=meta.get("title", "Unknown"),
                    snippet=doc[:300],
                    link=meta.get("link", ""),
                    score=round(score, 4),
                )
            )

        logger.debug("Query returned {} sources above threshold", len(sources))
        return sources

    # ── Meta ───────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def collection_name(self) -> str:
        return self._collection.name


@lru_cache(maxsize=1)
def get_vectorstore() -> VectorStore:
    """Cached singleton."""
    return VectorStore()
