# llm_factory.py
# Returns the right LLM instance based on MODE in config.py
# Supports: local (Ollama) | cloud (Groq/Anthropic/OpenAI/Gemini) | air (Air LLM) | hybrid

import requests
from config import (
    MODE, CLOUD_PROVIDER,
    REFINER_MODEL_LOCAL, CODER_MODEL_LOCAL,
    REFINER_MODEL_CLOUD, CODER_MODEL_CLOUD,
    AIR_LLM_URL, AIR_LLM_REFINER_MODEL, AIR_LLM_CODER_MODEL, AIR_LLM_TIMEOUT,
)
from config_loader import get_api_key


# ── Ollama health check ───────────────────────────────────────────────────────
_health_cache = {}  # {"ollama": (timestamp, bool), "air": (timestamp, bool)}
_CACHE_TTL = 30     # seconds

def _ollama_is_running() -> bool:
    import time
    cached = _health_cache.get("ollama")
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return cached[1]
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        result = r.status_code == 200
    except Exception:
        result = False
    _health_cache["ollama"] = (time.time(), result)
    return result


# ── Air LLM health check ──────────────────────────────────────────────────────
def _air_llm_is_running() -> bool:
    import time
    cached = _health_cache.get("air")
    if cached and (time.time() - cached[0]) < _CACHE_TTL:
        return cached[1]
    try:
        r = requests.get(AIR_LLM_URL, timeout=3)
        result = r.status_code < 500
    except Exception:
        result = False
    _health_cache["air"] = (time.time(), result)
    return result


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
        # Health checks are cached for 30s so refiner+coder don't both wait
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
    from langchain_ollama import ChatOllama
    model   = REFINER_MODEL_LOCAL if role == "refiner" else CODER_MODEL_LOCAL
    num_ctx = 4096 if role == "refiner" else 8192
    return ChatOllama(
        model=model,
        temperature=0.2 if role == "refiner" else 0.1,
        num_ctx=num_ctx,
        timeout=300,
    )


# ── Air LLM (llama.cpp-compatible HTTP server on your phone) ─────────────────

def _air_llm(role: str):
    """
    Air LLM exposes an OpenAI-compatible /v1/chat/completions endpoint.
    We use langchain_openai with a custom base_url pointing to the phone.
    Make sure the model is loaded in the Air LLM app before calling this.
    """
    from langchain_openai import ChatOpenAI
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
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temp,
            api_key=get_api_key("groq"),
        )

    if CLOUD_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temp,
            api_key=get_api_key("anthropic"),
        )

    if CLOUD_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=get_api_key("openai"),
        )

    if CLOUD_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temp,
            google_api_key=get_api_key("gemini"),
        )

    raise ValueError(f"Unknown CLOUD_PROVIDER: {CLOUD_PROVIDER}")