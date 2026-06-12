# core/__init__.py


"""
app/core/llm.py
vLLM LLM client (OpenAI-compatible API).
When vLLM is not running, returns a graceful stub answer
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
        self._base_url = cfg.vllm_base_url.rstrip("/")
        self._model    = cfg.vllm_model
        self._timeout  = cfg.vllm_timeout
        self._api_key  = cfg.vllm_api_key
        self._available: bool | None = None  # None = not yet checked

    # ── Availability probe ─────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=3) as client:
                r = client.get(
                    f"{self._base_url}/v1/models",
                    headers=self._headers(),
                )
                self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    # ── Prompt / messages builder ──────────────────────────────────────────

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
            "You are a helpful assistant for a technical glossary. "
            "Answer ONLY from the provided context. "
            "If the answer is not in the context, say so honestly.\n\n"
            f"### Context\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Last 3 turns of history
        for msg in history[-6:]:
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
        if not self.is_available():
            logger.warning("vLLM not available — returning stub answer")
            return self._stub_answer(sources), False

        messages = self.build_messages(query, sources, history)

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    f"{self._base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json={
                        "model":       self._model,
                        "messages":    messages,
                        "temperature": 0.2,
                        "max_tokens":  512,
                        "stream":      False,
                    },
                )
                response.raise_for_status()
                data   = response.json()
                answer = data["choices"][0]["message"]["content"].strip()
                if not answer:
                    raise ValueError("Empty response from vLLM")
                logger.info("LLM generated answer ({} chars)", len(answer))
                return answer, True

        except Exception as exc:
            logger.error("vLLM error: {}", exc)
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
