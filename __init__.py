# core/__init__.py

# ── Server ──────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:5500,null

# ── ChromaDB ─────────────────────────────────────────────
CHROMA_PATH=./data/chroma
CHROMA_COLLECTION=glossary

# ── Embeddings ───────────────────────────────────────────
EMBED_MODEL=all-MiniLM-L6-v2
EMBED_BATCH_SIZE=64

# ── Retrieval ────────────────────────────────────────────
TOP_K=5
SCORE_THRESHOLD=0.3

# ── LLM — Ollama ─────────────────────────────────────────
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a model:  ollama pull qwen2.5:3b
# 3. Ollama runs automatically in the background
#
# Good models for your i7 + 32GB RAM:
#   qwen2.5:3b   → fast, ~2GB, good for POC
#   qwen2.5:7b   → better quality, ~5GB
#   llama3.2:3b  → alternative option
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TIMEOUT=60

# ── Rate limiting ────────────────────────────────────────
RATE_LIMIT=30/minute

# ── Data ─────────────────────────────────────────────────
DATA_FILE=./data/rag_chunks.jsonl