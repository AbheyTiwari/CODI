# core/__init__.py


"""
app/api/chat.py
POST /chat — the main RAG endpoint.
"""
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.pipeline import run_rag
from app.core.logging import logger
from app.models.schemas import ChatRequest, ChatResponse
from app.utils.sanitise import is_prompt_injection, clean_query

router  = APIRouter(prefix="/chat", tags=["chat"])
limiter = Limiter(key_func=get_remote_address)


@router.post("", response_model=ChatResponse)
async def chat(raw_request: Request, request: ChatRequest) -> ChatResponse:
    """
    Main RAG endpoint.
    - Validates & sanitises input
    - Runs embed → retrieve → generate pipeline
    - Returns answer + sources
    """
    # Extra sanitisation on top of Pydantic
    clean = clean_query(request.query)

    if is_prompt_injection(clean):
        logger.warning("Prompt injection attempt blocked — ip={}", get_remote_address(raw_request))
        raise HTTPException(status_code=400, detail="Query contains disallowed patterns.")

    # Rebuild with cleaned query
    request = ChatRequest(query=clean, history=request.history)

    try:
        return run_rag(request)
    except Exception as exc:
        logger.error("Pipeline error: {}", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal pipeline error.")
