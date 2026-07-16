# llm_factory.py
# Returns the right LLM instance based on MODE in config.py
# Supports: local (Ollama) | cloud (Groq/Anthropic/OpenAI/Gemini) | air (Air LLM) | hybrid

import copy
import re
import time
import inspect

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
from core.profiler import get_active_profiler


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)

# Rough chars-per-token used ONLY for profiler estimation when a provider
# doesn't return real usage stats (Ollama/llama.cpp don't; cloud SDKs often
# do via response.usage_metadata but we keep one consistent estimate so
# numbers are comparable across backends rather than mixing exact+estimated).
_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> int:
    return int(len(text or "") / _CHARS_PER_TOKEN)


def _caller_label() -> str:
    """Best-effort 'module.function' label for the profiler, walking past
    this file's own frames to find the actual calling code (improver.py,
    executor.py, validator.py, planner.py, mission_analyzer.py, etc)."""
    try:
        frame = inspect.currentframe()
        frame = frame.f_back  # caller of _caller_label
        while frame:
            module = inspect.getmodule(frame)
            name = module.__name__ if module else ""
            if name and name not in ("llm_factory",):
                return f"{name}.{frame.f_code.co_name}"
            frame = frame.f_back
    except Exception:
        pass
    return "unknown"


class _FallbackLLM:
    def __init__(self, error: str):
        self._error = error

    def invoke(self, *_args, **_kwargs):
        raise RuntimeError(self._error)


def _strip_thinking_blocks(text: str) -> tuple[str, bool]:
    """Remove Qwen-style raw reasoning blocks before parsing or display."""
    cleaned, count = _THINK_BLOCK_RE.subn("", text or "")
    return cleaned.strip(), count > 0


class _ProfiledLLM:
    """
    Wraps any LangChain chat model so every .invoke() call is timed and
    token-estimated into the active AgentProfiler (if one is bound for the
    current run — see core/profiler.py's thread-local active_profiler()).

    This is the single chokepoint for LLM call telemetry: it wraps whatever
    the mode-specific constructor below returns, so local/llamacpp/air/cloud
    are all covered identically without duplicating timing code four times.
    """

    def __init__(self, llm, role: str):
        self._llm = llm
        self._role = role

    def invoke(self, messages, *args, **kwargs):
        profiler = get_active_profiler()
        caller = _caller_label() if profiler else ""

        prompt_text = ""
        try:
            prompt_text = "\n".join(
                getattr(m, "content", "") if not isinstance(m, str) else m
                for m in (messages if isinstance(messages, list) else [messages])
            )
        except Exception:
            prompt_text = ""

        started = time.monotonic()
        error = None
        try:
            response = self._llm.invoke(messages, *args, **kwargs)
        except Exception as e:
            error = str(e)
            duration = time.monotonic() - started
            if profiler:
                profiler.record_llm_call(
                    role=self._role, caller=caller,
                    prompt_tokens_est=_estimate_tokens(prompt_text),
                    completion_tokens_est=0,
                    duration_s=duration, error=error,
                )
            raise
        duration = time.monotonic() - started

        if profiler:
            completion_text = getattr(response, "content", "") or ""
            profiler.record_llm_call(
                role=self._role, caller=caller,
                prompt_tokens_est=_estimate_tokens(prompt_text),
                completion_tokens_est=_estimate_tokens(completion_text),
                duration_s=duration,
            )
        return response


def _profiled(llm, role: str):
    """Attach profiling without disturbing any existing wrapper (e.g.
    _ReasoningFilteredLLM for local Ollama) — profiling wraps OUTERMOST so
    it measures true end-to-end latency including reasoning-strip overhead."""
    return _ProfiledLLM(llm, role)


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
    

# ── llama.cpp health check ────────────────────────────────────────────────────
def _llamacpp_is_running() -> bool:
    from config import LLAMACPP_URL
    try:
        r = requests.get(f"{LLAMACPP_URL.rstrip('/')}/health", timeout=3)
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


def get_validator_llm():
    """Independent validator invocation using the refiner model configuration."""
    return _resolve("validator")


def _resolve(role: str):
    """
    Route to the right LLM based on MODE.
    role: "refiner" | "coder" | "validator"
    """
    if MODE == "local":
        return _profiled(_local_llm(role), role)

    if MODE == "llamacpp":
        return _profiled(_llamacpp_llm(role), role)

    if MODE == "air":
        return _profiled(_air_llm(role), role)

    if MODE == "cloud":
        return _profiled(_cloud_llm(role), role)

    if MODE == "hybrid":
        if _ollama_is_running():
            return _profiled(_local_llm(role), role)
        if _llamacpp_is_running():
            print(f"  [LLM] Ollama offline — falling back to llama.cpp (localhost)")
            return _profiled(_llamacpp_llm(role), role)
        if _air_llm_is_running():
            print(f"  [LLM] Ollama + llama.cpp offline — falling back to Air LLM ({AIR_LLM_URL})")
            return _profiled(_air_llm(role), role)
        print(f"  [LLM] All local backends offline — escalating to cloud ({CLOUD_PROVIDER})")
        return _profiled(_cloud_llm(role), role)

    raise ValueError(f"Unknown MODE: {MODE}. Use local | hybrid | cloud | air")

# ── Local (llama.cpp) ────────────────────────────────────────────────────────────

def _llamacpp_llm(role: str):
    from langchain_openai import ChatOpenAI
    from config import LLAMACPP_URL, LLAMACPP_REFINER_MODEL, LLAMACPP_CODER_MODEL, LLAMACPP_TIMEOUT
    model = LLAMACPP_CODER_MODEL if role == "coder" else LLAMACPP_REFINER_MODEL
    return ChatOpenAI(
        model=model,
        base_url=f"{LLAMACPP_URL.rstrip('/')}/v1",
        api_key="not-needed",
        temperature=0.1 if role == "coder" else 0.2,
        # Bound local generation so one stalled request does not consume two
        # full timeout windows and leave the agent appearing to only reason.
        timeout=LLAMACPP_TIMEOUT,
        max_retries=0,
        max_tokens=1600 if role == "coder" else 1200,
    )

# ── Local (Ollama) ────────────────────────────────────────────────────────────

def _local_llm(role: str):
    if not _ollama_is_running():
        return _FallbackLLM(f"Ollama is not reachable on {OLLAMA_BASE_URL}")

    try:
        from langchain_ollama import ChatOllama
    except Exception:
        return _FallbackLLM("langchain_ollama unavailable")

    from config import CODI_CONTEXT_WINDOW
    model   = CODER_MODEL_LOCAL if role == "coder" else REFINER_MODEL_LOCAL
    num_ctx = CODI_CONTEXT_WINDOW if role in {"coder", "validator"} else min(CODI_CONTEXT_WINDOW, 8192)
    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1 if role == "coder" else 0.2,
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

    model = AIR_LLM_CODER_MODEL if role == "coder" else AIR_LLM_REFINER_MODEL
    return ChatOpenAI(
        model=model,
        base_url=f"{AIR_LLM_URL.rstrip('/')}/v1",
        api_key="not-needed",                   # Air LLM doesn't need a key
        temperature=0.1 if role == "coder" else 0.2,
        timeout=AIR_LLM_TIMEOUT,
        max_retries=1,
    )


# ── Cloud ─────────────────────────────────────────────────────────────────────

def _cloud_llm(role: str):
    model = CODER_MODEL_CLOUD if role == "coder" else REFINER_MODEL_CLOUD
    temp  = 0.1 if role == "coder" else 0.2

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