# llm_factory.py
# Returns the right LLM instance based on MODE in config.py
# Supports: local (Ollama) | cloud (Groq/Anthropic/OpenAI/Gemini) | air (Air LLM) | hybrid

import copy
import re

import requests
from config import (
    MODE, CLOUD_PROVIDER,
    REFINER_MODEL_LOCAL, CODER_MODEL_LOCAL,
    REFINER_MODEL_CLOUD, CODER_MODEL_CLOUD,
    AIR_LLM_URL, AIR_LLM_REFINER_MODEL, AIR_LLM_CODER_MODEL, AIR_LLM_TIMEOUT,
    OLLAMA_BASE_URL, OLLAMA_THINK,
)
from config_loader import get_api_key
from status_stream import emit_status


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


class _FallbackLLM:
    def __init__(self, error: str):
        self._error = error

    def invoke(self, *_args, **_kwargs):
        raise RuntimeError(self._error)


def _strip_thinking_blocks(text: str) -> tuple[str, bool]:
    """Remove Qwen-style raw reasoning blocks before parsing or display."""
    cleaned, count = _THINK_BLOCK_RE.subn("", text or "")
    return cleaned.strip(), count > 0


class _ReasoningFilteredLLM:
    """Adapter that keeps raw model reasoning out of CODI's UI and parsers."""

    def __init__(self, llm):
        self._llm = llm

    def invoke(self, *args, **kwargs):
        response = self._llm.invoke(*args, **kwargs)
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            return response

        visible_content, had_reasoning = _strip_thinking_blocks(content)
        if not had_reasoning:
            return response

        emit_status("model", "Reasoning complete; using concise final output.")
        try:
            filtered = copy.copy(response)
            filtered.content = visible_content
            return filtered
        except Exception:
            response.content = visible_content
            return response


# ── Ollama health check ───────────────────────────────────────────────────────
def _ollama_is_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Air LLM health check ──────────────────────────────────────────────────────
def _air_llm_is_running() -> bool:
    try:
        r = requests.get(AIR_LLM_URL, timeout=5)
        return r.status_code < 500
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def get_refiner_llm():
    """Fast, cheap model for planning / refining / summarising."""
    return _resolve("refiner")


def get_coder_llm():
    """Stronger model for code generation and tool use."""
    return _resolve("coder")


def _resolve(role: str):
    """
    Route to the right LLM based on MODE.
    role: "refiner" | "coder"
    """
    if MODE == "local":
        return _local_llm(role)

    if MODE == "air":
        return _air_llm(role)

    if MODE == "cloud":
        return _cloud_llm(role)

    if MODE == "hybrid":
        # Try local first; fall back to Air LLM, then cloud
        if _ollama_is_running():
            return _local_llm(role)
        if _air_llm_is_running():
            print(f"  [LLM] Ollama offline — falling back to Air LLM ({AIR_LLM_URL})")
            return _air_llm(role)
        print(f"  [LLM] Ollama + Air LLM offline — escalating to cloud ({CLOUD_PROVIDER})")
        return _cloud_llm(role)

    raise ValueError(f"Unknown MODE: {MODE}. Use local | hybrid | cloud | air")


# ── Local (Ollama) ────────────────────────────────────────────────────────────

def _local_llm(role: str):
    if not _ollama_is_running():
        return _FallbackLLM(f"Ollama is not reachable on {OLLAMA_BASE_URL}")

    try:
        from langchain_ollama import ChatOllama
    except Exception:
        return _FallbackLLM("langchain_ollama unavailable")

    model   = REFINER_MODEL_LOCAL if role == "refiner" else CODER_MODEL_LOCAL
    num_ctx = 4096 if role == "refiner" else 8192
    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2 if role == "refiner" else 0.1,
        num_ctx=num_ctx,
        timeout=300,
        options={"think": OLLAMA_THINK},
    )
    return _ReasoningFilteredLLM(llm)


# ── Air LLM (llama.cpp-compatible HTTP server on your phone) ─────────────────

def _air_llm(role: str):
    """
    Air LLM exposes an OpenAI-compatible /v1/chat/completions endpoint.
    We use langchain_openai with a custom base_url pointing to the phone.
    Make sure the model is loaded in the Air LLM app before calling this.
    """
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return _FallbackLLM("langchain_openai unavailable")

    model = AIR_LLM_REFINER_MODEL if role == "refiner" else AIR_LLM_CODER_MODEL
    return ChatOpenAI(
        model=model,
        base_url=f"{AIR_LLM_URL.rstrip('/')}/v1",
        api_key="not-needed",                   # Air LLM doesn't need a key
        temperature=0.2 if role == "refiner" else 0.1,
        timeout=AIR_LLM_TIMEOUT,
        max_retries=1,
    )


# ── Cloud ─────────────────────────────────────────────────────────────────────

def _cloud_llm(role: str):
    model = REFINER_MODEL_CLOUD if role == "refiner" else CODER_MODEL_CLOUD
    temp  = 0.2 if role == "refiner" else 0.1

    if CLOUD_PROVIDER == "groq":
        try:
            from langchain_groq import ChatGroq
        except Exception:
            return _FallbackLLM("langchain_groq unavailable")
        return ChatGroq(
            model=model,
            temperature=temp,
            api_key=get_api_key("groq"),
        )

    if CLOUD_PROVIDER == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception:
            return _FallbackLLM("langchain_anthropic unavailable")
        return ChatAnthropic(
            model=model,
            temperature=temp,
            api_key=get_api_key("anthropic"),
        )

    if CLOUD_PROVIDER == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except Exception:
            return _FallbackLLM("langchain_openai unavailable")
        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=get_api_key("openai"),
        )

    if CLOUD_PROVIDER == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception:
            return _FallbackLLM("langchain_google_genai unavailable")
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            google_api_key=get_api_key("gemini"),
        )

    raise ValueError(f"Unknown CLOUD_PROVIDER: {CLOUD_PROVIDER}")
