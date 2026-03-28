# config.py
# Change these to switch modes

# "local" or "cloud"
MODE = "cloud"

# Cloud provider: "anthropic", "openai", "groq", or "gemini"
CLOUD_PROVIDER = "groq"  # groq is free and fast

# Model routing — local
REFINER_MODEL_LOCAL = "qwen2.5-coder:7b"
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"

# Model routing — cloud
REFINER_MODEL_CLOUD = "llama-3.1-8b-instant"
CODER_MODEL_CLOUD   = "llama-3.1-8b-instant"

# Gemini model options (used when CLOUD_PROVIDER = "gemini")
# Refiner: gemini-2.0-flash  (fast, cheap, good at instruction following)
# Coder:   gemini-2.5-pro    (best reasoning, use for complex codegen)

# claude model options (used when CLOUD_PROVIDER = "anthropic")
#CODER_MODEL_CLOUD   = "claude-opus-4-64k"  # best for codegen, use for coder
#REFINER_MODEL_CLOUD = "claude-opus-4-64k"  # best for reasoning, use for refiner

# API keys (only needed when MODE = "cloud")
GROQ_API_KEY      = ""
OPENAI_API_KEY    = ""
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY    = ""