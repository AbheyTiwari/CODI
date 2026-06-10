# core/__init__.py


"""
app/models/schemas.py
All Pydantic v2 request & response models.
Strict validation — bad input never reaches the pipeline.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ── Shared ────────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        return v.strip()


# ── /chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question.",
    )
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Conversation history, oldest first.",
    )

    @field_validator("query")
    @classmethod
    def sanitise_query(cls, v: str) -> str:
        v = v.strip()
        # Strip null bytes and control chars that could break downstream
        v = "".join(ch for ch in v if ch >= " " or ch in "\t\n")
        if not v:
            raise ValueError("query must contain visible characters")
        return v

    @field_validator("history")
    @classmethod
    def validate_history_alternates(cls, v: list[HistoryMessage]) -> list[HistoryMessage]:
        """History should alternate user/assistant. Enforce loosely."""
        if len(v) > 1:
            for i in range(1, len(v)):
                if v[i].role == v[i - 1].role:
                    raise ValueError(
                        f"history[{i}] has role '{v[i].role}' same as previous — "
                        "messages must alternate user/assistant"
                    )
        return v


class SourceDoc(BaseModel):
    title: str
    snippet: str
    link: str = ""
    score: float = Field(ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc] = []
    retrieved_count: int = 0
    llm_available: bool = False


# ── /ingest ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    skipped: int
    collection: str


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    chroma: str
    embedder: str
    llm: str
    documents_in_db: int
