# core/__init__.py

"""
app/api/health.py
GET /health — liveness + dependency status.
"""
from fastapi import APIRouter

from app.core.vectorstore import get_vectorstore
from app.core.embedder import get_embedder
from app.core.llm import get_llm
from app.core.logging import logger
from app.models.schemas import HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Returns the status of all subsystems."""
    # ChromaDB
    try:
        store = get_vectorstore()
        doc_count = store.count()
        chroma_status = "ok"
    except Exception as e:
        logger.error("ChromaDB health check failed: {}", e)
        chroma_status = f"error: {e}"
        doc_count = -1

    # Embedder
    try:
        get_embedder()
        embed_status = "ok"
    except Exception as e:
        logger.error("Embedder health check failed: {}", e)
        embed_status = f"error: {e}"

    # LLM
    try:
        llm = get_llm()
        llm_status = "ok" if llm.is_available() else "unavailable (stub mode)"
    except Exception as e:
        llm_status = f"error: {e}"

    overall = "ok" if all(
        s in ("ok", "unavailable (stub mode)")
        for s in [chroma_status, embed_status, llm_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        chroma=chroma_status,
        embedder=embed_status,
        llm=llm_status,
        documents_in_db=doc_count,
    )
