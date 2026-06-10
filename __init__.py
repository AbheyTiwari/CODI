# core/__init__.py



"""
app/core/embedder.py
Wraps sentence-transformers.
Singleton so the model is loaded once at startup.
Swap to Ollama embeddings later by replacing _encode().
"""
from __future__ import annotations
from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import logger


class Embedder:
    """Thread-safe embedding wrapper."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: {}", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model ready — dim={}", self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed a list of strings. Returns list of float vectors."""
        cfg = get_settings()
        bs = batch_size or cfg.embed_batch_size
        vecs = self._model.encode(
            list(texts),
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via dot product
        )
        return vecs.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Cached singleton — safe to call from anywhere."""
    return Embedder(get_settings().embed_model)
