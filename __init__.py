# core/__init__.py


"""
app/core/pipeline.py
The RAG pipeline — orchestrates embed → retrieve → generate.
This is the only place that touches all three subsystems together.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.embedder import get_embedder
from app.core.vectorstore import get_vectorstore
from app.core.llm import get_llm
from app.core.logging import logger
from app.models.schemas import ChatRequest, ChatResponse


def run_rag(request: ChatRequest) -> ChatResponse:
    """
    Full RAG pipeline:
      1. Embed the query
      2. Retrieve top-k docs from ChromaDB
      3. Generate answer via LLM (or stub)
      4. Return structured ChatResponse
    """
    cfg  = get_settings()
    query = request.query

    logger.info("RAG pipeline start — query='{}'", query[:80])

    # ── 1. Embed ──────────────────────────────────────────────────────────
    embedder = get_embedder()
    q_vec    = embedder.embed_one(query)

    # ── 2. Retrieve ───────────────────────────────────────────────────────
    store   = get_vectorstore()
    sources = store.query(
        embedding=q_vec,
        top_k=cfg.top_k,
        score_threshold=cfg.score_threshold,
    )
    logger.info("Retrieved {} sources", len(sources))

    # ── 3. Generate ───────────────────────────────────────────────────────
    llm                = get_llm()
    history_dicts      = [m.model_dump() for m in request.history]
    answer, llm_used   = llm.generate(query, sources, history_dicts)

    # ── 4. Return ─────────────────────────────────────────────────────────
    response = ChatResponse(
        answer=answer,
        sources=sources,
        retrieved_count=len(sources),
        llm_available=llm_used,
    )
    logger.info(
        "Pipeline done — sources={} llm_used={} answer_len={}",
        len(sources), llm_used, len(answer),
    )
    return response
