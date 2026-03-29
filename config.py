# config.py  —  Codi mode & model configuration
# ─────────────────────────────────────────────────────────────────────────────
#
#  MODE OPTIONS
#  ────────────
#  "local"   — 100% offline. Ollama only. Zero API spend. No internet needed.
#  "hybrid"  — Ollama handles most tasks. Cloud escalates when local overflows.
#  "cloud"   — Every LLM call goes to cloud. Max capability, costs money.
#  "air"     — Use Air LLM server on your phone over local Wi-Fi.
#
# ─────────────────────────────────────────────────────────────────────────────
MODE = "hybrid"          # local | hybrid | cloud | air

# ── Cloud provider (used when MODE="cloud" or hybrid escalates) ───────────────
# Options: "groq" | "anthropic" | "openai" | "gemini"
CLOUD_PROVIDER = "groq"

# ── Hybrid escalation settings ────────────────────────────────────────────────
HYBRID_TOKEN_LIMIT     = 3500   # trimmed context tokens before escalating to cloud
HYBRID_REQUIRE_CONFIRM = False  # True = prompt user before every cloud call
HYBRID_ESCALATE_ON     = [
    "token_overflow",       # context too large for local model
    "reasoning_required",   # task classifier flagged deep reasoning
    "local_unavailable",    # ollama not reachable / timed out
]

# ── Air LLM (local Wi-Fi inference — Android phone / tablet) ─────────────────
#
#  Air LLM app: https://play.google.com/store/apps/details?id=com.airlm.app
#  1. Install Air LLM on your Android phone
#  2. Download a GGUF model inside the app (phi-3-mini-4k-instruct.Q4_K_M is good)
#  3. Start the server — the app shows your phone's LAN IP + port
#  4. Set AIR_LLM_URL below to that address
#  5. Set MODE = "air"  (or it auto-activates as hybrid fallback if ollama dies)
#
AIR_LLM_URL           = "http://192.168.1.XXX:8080"  # replace with phone's LAN IP
AIR_LLM_REFINER_MODEL = "phi-3-mini"                 # model loaded in Air LLM
AIR_LLM_CODER_MODEL   = "phi-3-mini"
AIR_LLM_TIMEOUT       = 180                          # phones are slower — be patient

# ── Local models (Ollama) ─────────────────────────────────────────────────────
#
#  RECOMMENDED MODELS — run these to get ready:
#  ─────────────────────────────────────────────
#
#  TIER 0 · Refiner / Planner  (fast, tiny, instruction-following)
#    ollama pull phi3:mini              ~2.2 GB   Best CPU-only pick. Very fast.
#    ollama pull qwen2.5:3b             ~2.0 GB   Good instruction following.
#
#  TIER 1 · Code Generation  (the workhorse — you need at least one of these)
#    ollama pull qwen2.5-coder:7b       ~4.7 GB   RECOMMENDED. Best 7B coder.
#    ollama pull deepseek-coder:6.7b    ~4.2 GB   Strong alternative to above.
#    ollama pull codellama:7b           ~3.8 GB   Older but reliable.
#
#  TIER 2 · Deep Reasoning  (needs decent RAM / VRAM — optional)
#    ollama pull qwen2.5-coder:14b      ~9.0 GB   Best local coder if you have RAM.
#    ollama pull deepseek-r1:8b         ~5.0 GB   Reasoning model. Good for plans.
#    ollama pull llama3.1:8b            ~5.0 GB   General purpose fallback.
#
#  MINIMUM SETUP (low-end machine):
#    ollama pull phi3:mini              (refiner)
#    ollama pull qwen2.5-coder:7b       (coder)
#
#  HIGH-END SETUP (16 GB+ RAM or GPU):
#    ollama pull phi3:mini              (refiner — still keep it fast)
#    ollama pull qwen2.5-coder:14b      (coder — full quality)
#    ollama pull deepseek-r1:8b         (deep reasoning tasks)
#
REFINER_MODEL_LOCAL = "phi3:mini"           # fast planner / refiner
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"   # main code executor

# ── Cloud models ──────────────────────────────────────────────────────────────
#  Groq (free tier, very fast):
REFINER_MODEL_CLOUD = "llama-3.1-8b-instant"     # fast, cheap — good for planning
CODER_MODEL_CLOUD   = "llama-3.3-70b-versatile"  # strong — use for actual codegen

#  Anthropic (uncomment + set CLOUD_PROVIDER = "anthropic"):
#  REFINER_MODEL_CLOUD = "claude-haiku-4-5-20251001"
#  CODER_MODEL_CLOUD   = "claude-sonnet-4-6"

#  Gemini (uncomment + set CLOUD_PROVIDER = "gemini"):
#  REFINER_MODEL_CLOUD = "gemini-2.0-flash"
#  CODER_MODEL_CLOUD   = "gemini-2.5-pro"

#  OpenAI (uncomment + set CLOUD_PROVIDER = "openai"):
#  REFINER_MODEL_CLOUD = "gpt-4o-mini"
#  CODER_MODEL_CLOUD   = "gpt-4o"

# ── API keys ──────────────────────────────────────────────────────────────────
# Best practice: keep these blank here and set in .env or shell env instead:
#   CODI_GROQ_API_KEY=gsk_...
#   CODI_ANTHROPIC_API_KEY=sk-ant-...
#   CODI_OPENAI_API_KEY=sk-...
#   CODI_GEMINI_API_KEY=AIza...
GROQ_API_KEY      = ""
OPENAI_API_KEY    = ""
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY    = ""