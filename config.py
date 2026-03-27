# config.py
# Change these to switch modes

# "local" or "cloud"
MODE = "cloud"

# Cloud provider: "anthropic", "openai", "groq", or "gemini"
CLOUD_PROVIDER = "gemini"  # groq is free and fast

# Model routing — local
REFINER_MODEL_LOCAL = "qwen2.5-coder:7b"
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"

# Model routing — cloud
REFINER_MODEL_CLOUD = "llama-3.1-8b-instant"
CODER_MODEL_CLOUD   = "llama-3.3-70b-versatile"

# Gemini model options (used when CLOUD_PROVIDER = "gemini")
# Refiner: gemini-2.0-flash  (fast, cheap, good at instruction following)
# Coder:   gemini-2.5-pro    (best reasoning, use for complex codegen)
REFINER_MODEL_GEMINI = "gemini-2.5-pro"
CODER_MODEL_GEMINI   = "gemini-2.5-flash-lite"  # free experimental tier

# API keys (only needed when MODE = "cloud")
GROQ_API_KEY      = ""
OPENAI_API_KEY    = ""
ANTHROPIC_API_KEY = ""
GEMINI_API_KEY    = ""