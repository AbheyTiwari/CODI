# config.py
# Change these to switch modes

# "local" or "cloud"
MODE = "local"

# Cloud provider: "anthropic", "openai", or "groq"
CLOUD_PROVIDER = "groq"  # groq is free and fast

# Model routing
REFINER_MODEL_LOCAL = "llama3:8b"
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"

REFINER_MODEL_CLOUD = "llama-3.1-8b-instant"   # groq
CODER_MODEL_CLOUD   = "llama-3.1-70b-versatile" # groq — much better at CSS/JS

# API keys (only needed when MODE = "cloud")
GROQ_API_KEY     = "your-groq-api-key-here"
OPENAI_API_KEY   = "your-openai-api-key-here"
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
