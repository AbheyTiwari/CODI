# core/__init__.py


"""
app/core/llm.py
Local in-process LLM using Hugging Face Transformers `pipeline`.
No external server needed — model loads once into memory at startup
(lazily, on first request) and runs on CPU.

If the model fails to load (e.g. not enough RAM), falls back to a
stub answer so the rest of the pipeline keeps working.
"""
from __future__ import annotations
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import SourceDoc


_STUB_PREFIX = "[LLM unavailable — showing retrieved context only]\n\n"


class LLMClient:
    """
    Wraps a Transformers text-generation pipeline.
    Lazily loaded on first call to generate(), so the API starts up fast
    even if the model takes a while / fails to load.
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._model_name = cfg.llm_model
        self._max_new_tokens = cfg.llm_max_new_tokens
        self._pipe = None          # lazy
        self._load_failed = False

    # ── Lazy model loading ──────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Load the pipeline on first use. Returns True if ready."""
        if self._pipe is not None:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import pipeline

            logger.info("Loading local LLM '{}' (this can take a while)…", self._model_name)
            self._pipe = pipeline(
                "text-generation",
                model=self._model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,   # CPU-friendly
                device_map="cpu",
            )
            logger.info("Local LLM loaded successfully")
            return True

        except Exception as exc:
            logger.error("Failed to load local LLM '{}': {}", self._model_name, exc)
            self._load_failed = True
            return False

    # ── Availability ─────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._ensure_loaded()

    # ── Prompt builder ──────────────────────────────────────────────────────

    @staticmethod
    def build_messages(
        query: str,
        sources: list[SourceDoc],
        history: list[dict],
    ) -> list[dict]:
        ctx_parts = []
        for i, src in enumerate(sources, 1):
            ctx_parts.append(
                f"[{i}] {src.title}\n{src.snippet}"
                + (f"\nSource: {src.link}" if src.link else "")
            )
        context = "\n\n".join(ctx_parts) if ctx_parts else "No relevant context found."

        system_prompt = (
            "You are Cyientist AI, a helpful assistant for a technical glossary. "
            "Answer ONLY from the provided context. "
            "If the answer is not in the context, say so honestly.\n\n"
            f"### Context\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history[-6:]:  # last 3 turns
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})
        return messages

    # ── Generate ───────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        sources: list[SourceDoc],
        history: list[dict],
    ) -> tuple[str, bool]:
        """
        Returns (answer_text, llm_was_used).
        Never raises — falls back to stub on any error.
        """
        if not self._ensure_loaded():
            logger.warning("Local LLM not available — returning stub answer")
            return self._stub_answer(sources), False

        messages = self.build_messages(query, sources, history)

        try:
            tokenizer = self._pipe.tokenizer

            # Build a chat-formatted prompt using the model's chat template
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                # Fallback: simple concatenation
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
                prompt += "\nassistant:"

            outputs = self._pipe(
                prompt,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            answer = outputs[0]["generated_text"].strip()
            if not answer:
                raise ValueError("Empty response from local model")

            logger.info("LLM generated answer ({} chars)", len(answer))
            return answer, True

        except Exception as exc:
            logger.error("Local LLM generation error: {}", exc)
            return self._stub_answer(sources), False

    # ── Stub ───────────────────────────────────────────────────────────────

    @staticmethod
    def _stub_answer(sources: list[SourceDoc]) -> str:
        if not sources:
            return (
                _STUB_PREFIX
                + "No matching glossary terms were found for your query. "
                "Try rephrasing or using different keywords."
            )
        lines = [_STUB_PREFIX + "Here are the most relevant glossary terms I found:\n"]
        for src in sources:
            lines.append(f"**{src.title}**")
            if src.snippet:
                lines.append(src.snippet)
            if src.link:
                lines.append(f"→ {src.link}")
            lines.append("")
        return "\n".join(lines).strip()


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    return LLMClient()



"""
app/core/config.py
Single source of truth for all configuration.
Values are read from .env (or environment variables).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:5500,null"

    # ChromaDB
    chroma_path: str = "./data/chroma"
    chroma_collection: str = "glossary"

    # Embeddings
    embed_model: str = "all-MiniLM-L6-v2"
    embed_batch_size: int = 64

    # Retrieval
    top_k: int = 5
    score_threshold: float = 0.3

    # LLM — local Transformers pipeline (in-process, CPU)
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_max_new_tokens: int = 512

    # Rate limiting
    rate_limit: str = "30/minute"

    # Data
    data_file: str = "./data/glossary_data.json"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached singleton — call get_settings() everywhere."""
    return Settings()


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

    # LLM — report status without forcing a full model load on every health check.
    # First /chat call will trigger loading; here we just report current state.
    try:
        llm = get_llm()
        if llm._pipe is not None:
            llm_status = "ok (model loaded)"
        elif llm._load_failed:
            llm_status = "unavailable (stub mode — load failed)"
        else:
            llm_status = "not loaded yet (loads on first /chat request)"
    except Exception as e:
        llm_status = f"error: {e}"

    overall = "ok" if all(
        s in ("ok",) or "ok" in s or "not loaded" in s or "unavailable" in s
        for s in [chroma_status, embed_status, llm_status]
    ) else "degraded"

    return HealthResponse(
        status=overall,
        chroma=chroma_status,
        embedder=embed_status,
        llm=llm_status,
        documents_in_db=doc_count,
    )




<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Cyientist AI</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-dark: #0a0a0c;
    --surface: rgba(255,255,255,0.04);
    --surface-hover: rgba(255,255,255,0.08);
    --border: rgba(255,255,255,0.10);
    --border-accent: rgba(255,255,255,0.18);
    --text-primary: #f1f1f3;
    --text-muted: #9ca3af;
    --text-dim: #6b7280;
    --accent-blue: #d4d4d8;
    --accent-purple: #a1a1aa;
    --accent-pink: #e4e4e7;
    --glow-purple: rgba(212,212,216,0.18);
    --glow-blue: rgba(228,228,231,0.15);
    --source-bg: rgba(255,255,255,0.05);
    --source-border: rgba(255,255,255,0.12);
  }

  html, body { height: 100%; overflow: hidden; font-family: 'Inter', system-ui, sans-serif; background: var(--bg-dark); color: var(--text-primary); }

  /* ── Hex canvas ── */
  #hex-bg { position: fixed; inset: 0; z-index: 0; pointer-events: none; }

  /* ── App shell ── */
  #app { position: relative; z-index: 1; height: 100dvh; display: flex; flex-direction: column; }

  /* ── Top bar ── */
  #topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px 10px;
    border-bottom: 1px solid var(--border);
    background: rgba(6,9,18,0.6);
    backdrop-filter: blur(12px);
    flex-shrink: 0;
  }
  #topbar .brand {
    display: flex; align-items: center; gap: 9px;
  }
  #topbar .brand-ring {
    width: 28px; height: 28px; border-radius: 50%;
    border: 1.5px solid rgba(168,85,247,0.5);
    box-shadow: 0 0 12px var(--glow-purple);
    display: flex; align-items: center; justify-content: center;
  }
  #topbar .brand-name { font-size: 13px; font-weight: 500; letter-spacing: 0.04em; color: var(--text-primary); }
  #topbar .brand-name span { color: var(--accent-purple); }

  #clear-btn {
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text-muted); font-family: inherit; font-size: 12px;
    padding: 5px 12px; border-radius: 8px; cursor: pointer;
    display: flex; align-items: center; gap: 5px;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
  }
  #clear-btn:hover { background: var(--surface-hover); color: var(--accent-pink); border-color: rgba(232,121,160,0.3); }
  #clear-btn svg { width: 12px; height: 12px; }

  /* ── Messages ── */
  #messages {
    flex: 1; overflow-y: auto;
    padding: 20px 16px 8px;
    display: flex; flex-direction: column; gap: 20px;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent;
  }
  #messages::-webkit-scrollbar { width: 4px; }
  #messages::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }

  /* ── Empty state ── */
  #empty-state {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 12px;
    pointer-events: none; user-select: none;
  }
  #empty-state .ring {
    width: 60px; height: 60px; border-radius: 50%;
    border: 1.5px solid rgba(168,85,247,0.4);
    box-shadow: 0 0 30px var(--glow-purple), inset 0 0 20px rgba(168,85,247,0.08);
    display: flex; align-items: center; justify-content: center;
  }
  #empty-state .hint { font-size: 13px; color: var(--text-muted); letter-spacing: 0.03em; }
  #empty-state .sub  { font-size: 11px; color: var(--text-dim); letter-spacing: 0.02em; }

  /* ── Message row ── */
  .msg {
    display: flex; gap: 10px;
    max-width: 760px; width: 100%; align-self: center;
    animation: fadeUp 0.22s ease both;
  }
  @keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

  .msg.user { flex-direction: row-reverse; align-self: flex-end; max-width: 75%; }

  .avatar {
    width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; margin-top: 2px;
  }
  .msg.user .avatar     { background: linear-gradient(135deg, #e4e4e7, #a1a1aa); color: #18181b; }
  .msg.assistant .avatar { background: linear-gradient(135deg, #52525b, #27272a); color: #f1f1f3; box-shadow: 0 0 10px var(--glow-purple); }

  .msg-body { display: flex; flex-direction: column; gap: 8px; flex: 1; min-width: 0; }

  .bubble {
    padding: 12px 16px; border-radius: 14px;
    font-size: 14px; line-height: 1.75; letter-spacing: 0.01em;
    word-break: break-word;
  }
  .msg.user .bubble {
    background: linear-gradient(135deg, rgba(255,255,255,0.14), rgba(255,255,255,0.06));
    border: 1px solid rgba(255,255,255,0.18);
    border-bottom-right-radius: 4px;
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    color: #fafafa;
  }
  .msg.assistant .bubble {
    background: rgba(20, 20, 23, 0.93);
    border: 1px solid rgba(255,255,255,0.1);
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 12px rgba(0,0,0,0.4);
  }

  /* ── Sources panel ── */
  .sources {
    display: flex; flex-direction: column; gap: 8px;
    margin-top: 2px;
  }
  .sources-toggle {
    display: flex; align-items: center; gap: 6px;
    align-self: flex-start;
    font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--text-muted);
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px; padding: 5px 12px;
    cursor: pointer; user-select: none;
    transition: background 0.15s, color 0.15s, border-color 0.15s;
  }
  .sources-toggle:hover {
    background: rgba(255,255,255,0.08); color: var(--text-primary);
    border-color: rgba(255,255,255,0.2);
  }
  .sources-toggle .chevron {
    width: 9px; height: 9px;
    transition: transform 0.2s ease;
  }
  .sources-toggle.open .chevron { transform: rotate(180deg); }

  .sources-row {
    display: flex; gap: 10px; overflow-x: auto;
    padding: 2px 2px 8px 2px;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.12) transparent;
    max-height: 0; opacity: 0;
    transition: max-height 0.3s ease, opacity 0.25s ease;
  }
  .sources-row.open {
    max-height: 260px; opacity: 1;
  }
  .sources-row::-webkit-scrollbar { height: 4px; }
  .sources-row::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 2px; }

  .source-card {
    flex: 0 0 220px;
    background: linear-gradient(160deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px; padding: 12px 14px;
    display: flex; flex-direction: column; gap: 7px;
    transition: transform 0.15s, border-color 0.15s, background 0.15s;
  }
  .source-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255,255,255,0.28);
    background: linear-gradient(160deg, rgba(255,255,255,0.1), rgba(255,255,255,0.03));
  }
  .source-card-top {
    display: flex; align-items: flex-start; justify-content: space-between; gap: 8px;
  }
  .source-num {
    flex-shrink: 0;
    width: 20px; height: 20px; border-radius: 50%;
    background: rgba(255,255,255,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700; color: var(--text-primary);
  }
  .source-title {
    font-size: 12.5px; font-weight: 600; color: #f4f4f5;
    line-height: 1.4;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }
  .source-snippet {
    font-size: 11.5px; color: #a1a1aa; line-height: 1.5;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
  }
  .source-card-bottom {
    display: flex; align-items: center; justify-content: space-between; gap: 8px;
    margin-top: auto;
  }
  .score-bar {
    flex: 1; height: 3px; border-radius: 2px;
    background: rgba(255,255,255,0.08); overflow: hidden;
  }
  .score-fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, #71717a, #f4f4f5);
    transition: width 0.4s ease;
  }
  .source-score {
    font-size: 10px; color: var(--text-dim); flex-shrink: 0;
  }
  .source-link {
    flex-shrink: 0;
    font-size: 11px; font-weight: 600;
    color: #18181b;
    text-decoration: none;
    background: #f4f4f5;
    border-radius: 6px; padding: 4px 10px;
    transition: background 0.15s, transform 0.1s;
    white-space: nowrap;
  }
  .source-link:hover { background: #ffffff; transform: scale(1.04); }

  /* ── Typing dots ── */
  .typing-dots span {
    display: inline-block; width: 5px; height: 5px; border-radius: 50%;
    background: var(--accent-purple); margin: 0 2px;
    animation: blink 1.1s infinite both;
  }
  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink { 0%,80%,100%{opacity:.2;transform:scale(.8);} 40%{opacity:1;transform:scale(1);} }

  /* ── Input bar ── */
  #input-area { padding: 10px 16px 20px; display: flex; justify-content: center; flex-shrink: 0; }

  #input-wrap {
    width: 100%; max-width: 760px;
    display: flex; align-items: flex-end; gap: 0;
    background: rgba(10,13,26,0.8);
    backdrop-filter: blur(20px) saturate(1.4);
    -webkit-backdrop-filter: blur(20px) saturate(1.4);
    border: 1px solid var(--border-accent);
    border-radius: 22px; padding: 10px 10px 10px 18px;
    box-shadow: 0 0 0 1px rgba(79,163,224,0.06), 0 8px 32px rgba(0,0,0,0.5), 0 0 40px rgba(168,85,247,0.07);
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  #input-wrap:focus-within {
    border-color: rgba(168,85,247,0.5);
    box-shadow: 0 0 0 1px rgba(79,163,224,0.12), 0 8px 32px rgba(0,0,0,0.5), 0 0 52px rgba(168,85,247,0.16);
  }

  #prompt-input {
    flex: 1; background: transparent; border: none; outline: none; resize: none;
    color: var(--text-primary); font-family: inherit; font-size: 14px; line-height: 1.6;
    max-height: 160px; min-height: 24px; overflow-y: auto; scrollbar-width: none;
    caret-color: var(--accent-purple);
  }
  #prompt-input::placeholder { color: var(--text-muted); }
  #prompt-input::-webkit-scrollbar { display: none; }

  #send-btn {
    flex-shrink: 0; width: 36px; height: 36px; border-radius: 50%;
    border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    box-shadow: 0 0 14px var(--glow-purple);
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s;
    color: #fff;
  }
  #send-btn:hover  { transform: scale(1.08); box-shadow: 0 0 22px var(--glow-purple); }
  #send-btn:active { transform: scale(0.95); }
  #send-btn:disabled { opacity: 0.3; cursor: default; transform: none; }

  /* ── Divider ── */
  .day-divider {
    display: flex; align-items: center; gap: 10px;
    font-size: 10px; color: var(--text-dim); letter-spacing: 0.06em; text-transform: uppercase;
  }
  .day-divider::before, .day-divider::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
  }
</style>
</head>
<body>

<canvas id="hex-bg"></canvas>

<div id="app">

  <!-- Top bar -->
  <div id="topbar">
    <div class="brand">
      <div class="brand-ring">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M12 3L20 7.5V16.5L12 21L4 16.5V7.5L12 3Z" stroke="url(#tg)" stroke-width="1.5" fill="none"/>
          <defs><linearGradient id="tg" x1="4" y1="3" x2="20" y2="21" gradientUnits="userSpaceOnUse">
            <stop stop-color="#4fa3e0"/><stop offset="1" stop-color="#a855f7"/>
          </linearGradient></defs>
        </svg>
      </div>
      <span class="brand-name">Cyientist<span> AI</span></span>
    </div>
    <button id="clear-btn">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6"/>
      </svg>
      Clear chat
    </button>
  </div>

  <!-- Messages -->
  <div id="messages">
    <div id="empty-state">
      <div class="ring">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
          <path d="M12 3L20 7.5V16.5L12 21L4 16.5V7.5L12 3Z" stroke="url(#eg)" stroke-width="1.4" fill="none"/>
          <defs><linearGradient id="eg" x1="4" y1="3" x2="20" y2="21" gradientUnits="userSpaceOnUse">
            <stop stop-color="#4fa3e0"/><stop offset="1" stop-color="#a855f7"/>
          </linearGradient></defs>
        </svg>
      </div>
      <p class="hint">Ask Cyientist AI anything</p>
      <p class="sub">Powered by your glossary — sources cited with every answer</p>
    </div>
  </div>

  <!-- Input -->
  <div id="input-area">
    <div id="input-wrap">
      <textarea id="prompt-input" rows="1" placeholder="Ask me anything…"></textarea>
      <button id="send-btn" title="Send (Enter)">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
          <path d="M22 2L11 13" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>
  </div>

</div>

<script>
/* ─────────────────────────────────────────
   HEX CANVAS BACKGROUND
───────────────────────────────────────── */
(function () {
  const canvas = document.getElementById('hex-bg');
  const ctx = canvas.getContext('2d');
  const R = 13, GAP = 1.4;
  const SX = R * Math.sqrt(3) + GAP;
  const SY = R * 1.5 + GAP * 0.866;

  function resize() { canvas.width = innerWidth; canvas.height = innerHeight; draw(); }

  function hexPath(cx, cy) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 3) * i - Math.PI / 6;
      i === 0 ? ctx.moveTo(cx + R * Math.cos(a), cy + R * Math.sin(a))
              : ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a));
    }
    ctx.closePath();
  }

  function lerp(a, b, t) { return a + (b - a) * t; }

  function cellColor(nx, ny) {
    const dB = Math.hypot(nx - 0.33, ny - 0.76);
    const dP = Math.hypot(nx - 0.63, ny - 0.71);
    const iB = Math.max(0, 1 - dB / 0.40);
    const iP = Math.max(0, 1 - dP / 0.32);
    const tot = Math.max(iB, iP);
    if (tot < 0.015) return null;
    const t = iP / (iB + iP + 1e-4);
    const v = lerp(90, 235, t) | 0;   // greyscale value
    return {
      r: v, g: v, b: v,
      a: Math.min(1, tot * 1.3)
    };
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cols = Math.ceil(canvas.width  / SX) + 2;
    const rows = Math.ceil(canvas.height / SY) + 2;
    for (let row = -1; row < rows; row++) {
      for (let col = -1; col < cols; col++) {
        const cx = col * SX + (row % 2 === 0 ? 0 : SX / 2);
        const cy = row * SY;
        const c  = cellColor(cx / canvas.width, cy / canvas.height);
        hexPath(cx, cy);
        if (c) {
          ctx.fillStyle   = `rgba(${c.r},${c.g},${c.b},${(c.a * 0.30).toFixed(3)})`;
          ctx.fill();
          ctx.strokeStyle = `rgba(${c.r},${c.g},${c.b},${c.a.toFixed(3)})`;
          ctx.lineWidth   = 0.8;
        } else {
          ctx.strokeStyle = 'rgba(255,255,255,0.035)';
          ctx.lineWidth   = 0.5;
        }
        ctx.stroke();
      }
    }
  }

  window.addEventListener('resize', resize);
  resize();
})();

/* ─────────────────────────────────────────
   CONFIG — swap these when your API is ready
───────────────────────────────────────── */
const API_BASE  = 'http://localhost:8000';   // your FastAPI base URL
const CHAT_PATH = '/chat';                   // POST endpoint

/*
  Expected request body:
  {
    "query":   "user message",
    "history": [{ "role": "user"|"assistant", "content": "..." }, ...]
  }

  Expected response body (non-streaming JSON):
  {
    "answer": "...",
    "sources": [
      { "title": "doc.pdf", "snippet": "...", "score": 0.91 },
      ...
    ]
  }

  To switch to streaming, see the streamChat() function below.
*/

/* ─────────────────────────────────────────
   STATE
───────────────────────────────────────── */
let history = [];   // [{ role, content }]
let busy    = false;

/* ─────────────────────────────────────────
   DOM REFS
───────────────────────────────────────── */
const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('prompt-input');
const sendBtn    = document.getElementById('send-btn');
const clearBtn   = document.getElementById('clear-btn');

/* ─────────────────────────────────────────
   INPUT HELPERS
───────────────────────────────────────── */
function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}
inputEl.addEventListener('input', autoResize);
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener('click', send);

/* ─────────────────────────────────────────
   CLEAR
───────────────────────────────────────── */
clearBtn.addEventListener('click', () => {
  history = [];
  messagesEl.innerHTML = `
    <div id="empty-state">
      <div class="ring">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
          <path d="M12 3L20 7.5V16.5L12 21L4 16.5V7.5L12 3Z" stroke="url(#eg2)" stroke-width="1.4" fill="none"/>
          <defs><linearGradient id="eg2" x1="4" y1="3" x2="20" y2="21" gradientUnits="userSpaceOnUse">
            <stop stop-color="#4fa3e0"/><stop offset="1" stop-color="#a855f7"/>
          </linearGradient></defs>
        </svg>
      </div>
      <p class="hint">Ask Cyientist AI anything</p>
      <p class="sub">Sources will appear alongside each answer</p>
    </div>`;
});

/* ─────────────────────────────────────────
   RENDER HELPERS
───────────────────────────────────────── */
function removeEmpty() {
  const e = document.getElementById('empty-state');
  if (e) e.remove();
}

function addUserMessage(text) {
  removeEmpty();
  const msg = document.createElement('div');
  msg.className = 'msg user';
  msg.innerHTML = `
    <div class="avatar">U</div>
    <div class="msg-body">
      <div class="bubble">${escHtml(text)}</div>
    </div>`;
  messagesEl.appendChild(msg);
  scrollBottom();
}

function addAssistantPlaceholder() {
  removeEmpty();
  const msg = document.createElement('div');
  msg.className = 'msg assistant';
  msg.innerHTML = `
    <div class="avatar">AI</div>
    <div class="msg-body">
      <div class="bubble" id="live-bubble">
        <span class="typing-dots"><span></span><span></span><span></span></span>
      </div>
    </div>`;
  messagesEl.appendChild(msg);
  scrollBottom();
  return msg;
}

function finalizeAssistantMsg(msgEl, answer, sources) {
  const body   = msgEl.querySelector('.msg-body');
  const bubble = msgEl.querySelector('.bubble');
  bubble.removeAttribute('id');
  bubble.textContent = answer;

  if (sources && sources.length > 0) {
    // De-duplicate sources by link (or title if no link)
    const seen = new Set();
    const unique = [];
    for (const s of sources) {
      const key = (s.link && s.link.trim()) || s.title;
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push(s);
    }

    const sourcesEl = document.createElement('div');
    sourcesEl.className = 'sources';

    const toggleId = 'src-' + Math.random().toString(36).slice(2, 9);

    const cardsHtml = unique.map((s, i) => {
      const scoreVal = s.score != null ? Math.round(s.score * 100) : null;
      const linkHtml = s.link
        ? `<a class="source-link" href="${escHtml(s.link)}" target="_blank" rel="noopener">Open ↗</a>`
        : '';
      const scoreHtml = scoreVal != null ? `
        <div class="score-bar"><div class="score-fill" style="width:${scoreVal}%"></div></div>
        <span class="source-score">${scoreVal}%</span>` : '';
      return `
      <div class="source-card">
        <div class="source-card-top">
          <div class="source-num">${i + 1}</div>
          ${linkHtml}
        </div>
        <div class="source-title">${escHtml(s.title || 'Document')}</div>
        ${s.snippet ? `<div class="source-snippet">${escHtml(s.snippet)}</div>` : ''}
        <div class="source-card-bottom">${scoreHtml}</div>
      </div>`;
    }).join('');

    sourcesEl.innerHTML = `
      <div class="sources-toggle" data-target="${toggleId}">
        <span>${unique.length} source${unique.length > 1 ? 's' : ''}</span>
        <svg class="chevron" viewBox="0 0 12 8" fill="none">
          <path d="M1 1.5L6 6.5L11 1.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>
      <div class="sources-row" id="${toggleId}">${cardsHtml}</div>
    `;

    body.appendChild(sourcesEl);

    const toggle = sourcesEl.querySelector('.sources-toggle');
    const row    = sourcesEl.querySelector('.sources-row');
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('open');
      row.classList.toggle('open');
      scrollBottom();
    });
  }
  scrollBottom();
}

function showError(msgEl, err) {
  const bubble = msgEl.querySelector('.bubble');
  bubble.innerHTML = `<span style="color:#e879a0">⚠ ${escHtml(String(err))}</span>`;
  scrollBottom();
}

function scrollBottom() { messagesEl.scrollTop = messagesEl.scrollHeight; }
function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ─────────────────────────────────────────
   SEND
───────────────────────────────────────── */
async function send() {
  const query = inputEl.value.trim();
  if (!query || busy) return;

  busy = true;
  sendBtn.disabled = true;
  inputEl.value = '';
  autoResize();

  addUserMessage(query);
  history.push({ role: 'user', content: query });

  const msgEl = addAssistantPlaceholder();

  try {
    const res = await fetch(API_BASE + CHAT_PATH, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, history })
    });

    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }

    const data = await res.json();
    /*
      Adjust these keys to match your actual FastAPI response shape:
        data.answer  → the LLM answer string
        data.sources → array of { title, snippet, score }
    */
    const answer  = data.answer  ?? data.response ?? data.text ?? JSON.stringify(data);
    const sources = data.sources ?? data.citations ?? [];

    history.push({ role: 'assistant', content: answer });
    finalizeAssistantMsg(msgEl, answer, sources);

  } catch (err) {
    // Remove the failed turn from history
    history.pop();
    showError(msgEl, err.message || err);
  }

  busy = false;
  sendBtn.disabled = false;
  inputEl.focus();
}
</script>
</body>
</html>
