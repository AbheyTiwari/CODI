# core/__init__.py


"""
app/core/llm.py
Ollama LLM client.
When Ollama is not running, returns a graceful stub answer
so the rest of the pipeline (retrieval, frontend) keeps working.
"""
from __future__ import annotations
import httpx
from functools import lru_cache

from app.core.config import get_settings
from app.core.logging import logger
from app.models.schemas import SourceDoc


_STUB_PREFIX = "[LLM unavailable — showing retrieved context only]\n\n"


class LLMClient:
    def __init__(self) -> None:
        cfg = get_settings()
        self._base_url = cfg.ollama_base_url.rstrip("/")
        self._model    = cfg.ollama_model
        self._timeout  = cfg.ollama_timeout
        self._available: bool | None = None  # None = not yet checked

    # ── Availability probe ─────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=3) as client:
                r = client.get(f"{self._base_url}/api/tags")
                self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    # ── Prompt builder ─────────────────────────────────────────────────────

    @staticmethod
    def build_prompt(
        query: str,
        sources: list[SourceDoc],
        history: list[dict],
    ) -> str:
        ctx_parts = []
        for i, src in enumerate(sources, 1):
            ctx_parts.append(
                f"[{i}] {src.title}\n{src.snippet}"
                + (f"\nSource: {src.link}" if src.link else "")
            )
        context = "\n\n".join(ctx_parts) if ctx_parts else "No relevant context found."

        history_text = ""
        for msg in history[-6:]:  # last 3 turns
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

        prompt = (
            "You are a helpful assistant for a technical glossary. "
            "Answer ONLY from the provided context. "
            "If the answer is not in the context, say so honestly.\n\n"
            f"### Context\n{context}\n\n"
            + (f"### Conversation so far\n{history_text}\n" if history_text else "")
            + f"### Question\n{query}\n\n"
            "### Answer"
        )
        return prompt

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
        if not self.is_available():
            logger.warning("Ollama not available — returning stub answer")
            return self._stub_answer(sources), False

        prompt = self.build_prompt(query, sources, history)

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model":  self._model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2, "num_predict": 512},
                    },
                )
                response.raise_for_status()
                data   = response.json()
                answer = data.get("response", "").strip()
                if not answer:
                    raise ValueError("Empty response from Ollama")
                logger.info("LLM generated answer ({} chars)", len(answer))
                return answer, True

        except Exception as exc:
            logger.error("Ollama error: {}", exc)
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
