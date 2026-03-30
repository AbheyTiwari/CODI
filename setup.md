# Codi — Setup Guide

Complete installation guide for Windows, macOS, and Linux.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1 · Clone the repo](#1--clone-the-repo)
- [2 · Create a virtual environment](#2--create-a-virtual-environment)
- [3 · Run the setup script](#3--run-the-setup-script)
- [4 · Configure your mode](#4--configure-your-mode)
- [5 · Add API keys](#5--add-api-keys)
- [6 · Install local models (optional)](#6--install-local-models-optional)
- [7 · Register the global `codi` command](#7--register-the-global-codi-command)
- [8 · Launch](#8--launch)
- [Modes reference](#modes-reference)
- [MCP servers](#mcp-servers)
- [AirLLM — phone inference](#airllm--phone-inference)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Tool | Why | Install |
|------|-----|---------|
| **Python 3.10+** | Required | [python.org](https://python.org) |
| **Git** | Clone the repo | [git-scm.com](https://git-scm.com) |
| **Node.js 18+** | npm MCP servers (`npx`) | [nodejs.org](https://nodejs.org) |
| **uv / uvx** | uvx MCP servers | `pip install uv` |
| **Ollama** *(optional)* | Local model inference | [ollama.com](https://ollama.com) |

Check what you have:

```bash
python --version   # need 3.10+
git --version
node --version     # need 18+
npx --version
uvx --version      # or: pip install uv
ollama --version   # optional
```

---

## 1 · Clone the repo

```bash
git clone https://github.com/AbheyTiwari/CODI.git
cd CODI/coding-agent
```

---

## 2 · Create a virtual environment

A virtual environment keeps Codi's dependencies isolated from your system Python.

**Windows (PowerShell)**
```powershell
python -m venv codi
.\codi\Scripts\Activate.ps1
```

> If you see a script execution error, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux**
```bash
python3 -m venv codi
source codi/bin/activate
```

Your prompt should now show `(codi)` at the start. Keep this active for all following steps.

---

## 3 · Run the setup script

The setup script auto-detects your system, patches `mcp_servers.json` with your real paths, creates a `.env` template, installs CPU-only PyTorch (avoids a 3 GB CUDA download), and installs all Python dependencies.

```bash
python setup.py
```

It will print progress for each of its 5 steps:

```
1/5  Resolving tool paths
2/5  Configuring MCP servers
3/5  Setting up .env
4/5  Installing Python dependencies
5/5  Creating launcher
```

If any step fails, read the warning — it will tell you exactly what's missing and how to fix it.

---

## 4 · Configure your mode

Open `config.py` and set `MODE` to match your setup:

```python
# config.py
MODE = "hybrid"   # ← change this
```

| Mode | What it does | Needs |
|------|-------------|-------|
| `local` | 100% offline, Ollama only, zero API cost | Ollama + models |
| `hybrid` | Ollama first, falls back to cloud if unavailable | Ollama + API key |
| `cloud` | Every call goes to cloud, max capability | API key |
| `air` | Inference on a phone/device over Wi-Fi | AirLLM server |

**Recommended starting point:** `hybrid` — works offline when Ollama is running, falls back to Groq (free) when it's not.

Also set your cloud provider if using `hybrid` or `cloud`:

```python
CLOUD_PROVIDER = "groq"   # groq | anthropic | openai | gemini
```

---

## 5 · Add API keys

Open `.env` (created by the setup script) and fill in the keys for the providers you want to use. You only need one.

```env
# .env

# Groq — free tier, very fast, recommended
CODI_GROQ_API_KEY=gsk_...

# Anthropic — best quality codegen
CODI_ANTHROPIC_API_KEY=sk-ant-...

# OpenAI
CODI_OPENAI_API_KEY=sk-...

# Gemini
CODI_GEMINI_API_KEY=AIza...

# GitHub MCP server (optional — needed for GitHub operations)
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
```

**Where to get keys:**
- Groq (free): [console.groq.com](https://console.groq.com) → API Keys
- Anthropic: [console.anthropic.com](https://console.anthropic.com)
- OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Gemini: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
- GitHub token: Settings → Developer settings → Personal access tokens

> **Never commit `.env` to git.** It's already in `.gitignore`.

---

## 6 · Install local models (optional)

Required for `local` and `hybrid` modes. Skip if using `cloud` only.

```bash
# Minimum — works on any machine with 8 GB RAM
ollama pull phi3:mini            # 2.2 GB  fast refiner / planner
ollama pull qwen2.5-coder:7b     # 4.7 GB  main code model (recommended)

# Upgrade — 16 GB RAM or a GPU
ollama pull qwen2.5-coder:14b    # 9.0 GB  noticeably better output
ollama pull deepseek-r1:8b       # 5.0 GB  better reasoning / planning
```

After pulling, update `config.py` if you want to use a different model:

```python
REFINER_MODEL_LOCAL = "phi3:mini"         # fast planner
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"  # code generation
```

Ollama must be running when you use `local` or `hybrid` mode:

**Windows / macOS:** Ollama runs as a background service after install — nothing extra needed.

**Linux:**
```bash
ollama serve &   # start in background
```

---

## 7 · Register the global `codi` command

This is what lets you type `codi` in any project directory and have it launch there automatically.

```bash
# From inside coding-agent/, with venv active:
pip install -e .
```

This registers `codi` as a console script in your venv's `Scripts/` (Windows) or `bin/` (Unix) folder.

### Making `codi` available in every terminal

The `codi` command only works in terminals where the venv is active. To make it permanent:

**Windows — add to system PATH (run once in admin PowerShell):**
```powershell
# Replace the path with your actual venv location
$p = "C:\path\to\CODI\coding-agent\codi\Scripts"
[Environment]::SetEnvironmentVariable(
    "PATH",
    [Environment]::GetEnvironmentVariable("PATH", "User") + ";$p",
    "User"
)
# Restart all terminals after running this
```

Or add this to your PowerShell profile (`$PROFILE`) to auto-activate the venv:
```powershell
& "C:\path\to\CODI\coding-agent\codi\Scripts\Activate.ps1"
```

**macOS / Linux — the setup script already added an alias** to your `~/.bashrc` or `~/.zshrc`. Activate it:
```bash
source ~/.bashrc    # bash
source ~/.zshrc     # zsh
```

Or add it manually if needed:
```bash
echo 'alias codi="python /path/to/CODI/coding-agent/main.py"' >> ~/.bashrc
source ~/.bashrc
```

---

## 8 · Launch

Navigate to any project directory and run:

```bash
cd /path/to/your/project
codi
```

Codi will:
1. Auto-index the directory (incremental — only changed files on subsequent runs)
2. Load MCP servers
3. Start the interactive CLI

From inside Codi, type `/help` to see all commands.

---

## Modes reference

Switch modes at any time inside Codi without restarting:

```
❯ /mode hybrid
❯ /mode cloud
❯ /mode local
❯ /mode air
```

Or set permanently in `config.py`.

### Local mode

```python
MODE = "local"
REFINER_MODEL_LOCAL = "phi3:mini"
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"
```

No internet required. All inference runs on your machine via Ollama. MCP tool list is automatically filtered to a small essential set to keep the context window usable with 7B models.

### Hybrid mode (recommended)

```python
MODE = "hybrid"
CLOUD_PROVIDER = "groq"
```

Tries Ollama first. If Ollama is offline, falls back to AirLLM (if configured), then to your cloud provider. Best of both worlds — free and offline most of the time, cloud when you need it.

### Cloud mode

```python
MODE = "cloud"
CLOUD_PROVIDER = "groq"   # or anthropic, openai, gemini
CODER_MODEL_CLOUD   = "llama-3.3-70b-versatile"
REFINER_MODEL_CLOUD = "llama-3.1-8b-instant"
```

Every LLM call goes to your chosen cloud provider. Best capability, uses API credits.

**Recommended cloud models by provider:**

| Provider | Refiner | Coder |
|----------|---------|-------|
| Groq | `llama-3.1-8b-instant` | `llama-3.3-70b-versatile` |
| Anthropic | `claude-haiku-4-5-20251001` | `claude-sonnet-4-6` |
| OpenAI | `gpt-4o-mini` | `gpt-4o` |
| Gemini | `gemini-2.0-flash` | `gemini-2.5-pro` |

### Air mode

See [AirLLM — phone inference](#airllm--phone-inference) below.

---

## MCP servers

MCP (Model Context Protocol) servers extend what Codi can do. They're managed in `mcp_servers.json`.

### Enabled by default

| Server | What it does |
|--------|-------------|
| `filesystem` | Read/write any file on disk |
| `memory` | Persistent cross-session knowledge graph |
| `sequential-thinking` | Forces step-by-step reasoning |
| `github` | Read/write GitHub repos, issues, PRs |
| `fetch` | Fetch any URL |
| `git` | Git operations on local repos |
| `playwright` | Browser automation |
| `stitch` | Google Stitch UI/UX code generation |

### Toggle servers at runtime

```
❯ /mcp              # list all servers and status
❯ /mcp on brave-search
❯ /mcp off playwright
```

Changes take effect on restart.

### Enable optional servers

**Brave Search** (free web search, 2000 req/month):
1. Get a key at [brave.com/search/api](https://search.brave.com/search/api)
2. Add `BRAVE_API_KEY=your_key` to `.env`
3. Run `/mcp on brave-search` and restart

**PostgreSQL:**
Edit `mcp_servers.json`, update the connection string in the `postgres` args, then `/mcp on postgres`.

**Docker / Kubernetes:**
Just `/mcp on docker` or `/mcp on kubernetes` — no extra config needed if Docker/kubectl is installed.

### Requirements

- **npx servers** need Node.js 18+ installed
- **uvx servers** need `pip install uv` (setup script handles this)
- **HTTP servers** (stitch) need the relevant API key in `.env`

---

## AirLLM — phone inference

[AirLLM](https://github.com/lyogavin/airllm) runs quantized LLMs with extreme memory efficiency, loading one layer at a time. You can run it on a device with as little as 3 GB of RAM — a phone via Termux, a Raspberry Pi, or a spare laptop.

### Setup

**1. Install AirLLM on the inference device**

```bash
pip install airllm
pip install fastapi uvicorn   # for the HTTP server
```

**2. Create a server script** — save as `llm_server.py` on the inference device:

```python
from airllm import AutoModel
from fastapi import FastAPI
import uvicorn

model = AutoModel.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
app   = FastAPI()

@app.post("/v1/chat/completions")
async def chat(req: dict):
    msgs   = req.get("messages", [])
    prompt = "\n".join(m["content"] for m in msgs)
    tokens = model.tokenizer(prompt, return_tensors="pt")
    out    = model.generate(**tokens, max_new_tokens=512)
    text   = model.tokenizer.decode(out[0], skip_special_tokens=True)
    return {"choices": [{"message": {"content": text}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

**3. Run the server**

```bash
python llm_server.py
```

Note the LAN IP of the device (e.g. `192.168.1.42`).

**4. Configure Codi**

In `.env`:
```env
CODI_AIR_LLM_URL=http://192.168.1.42:8080
```

In `config.py`:
```python
MODE = "air"
AIR_LLM_REFINER_MODEL = "phi-3-mini"
AIR_LLM_CODER_MODEL   = "phi-3-mini"
AIR_LLM_TIMEOUT       = 180   # phones are slower — be patient
```

Or leave `MODE = "hybrid"` and Codi will automatically fall back to AirLLM when Ollama is offline.

**Recommended models for AirLLM:**
- `microsoft/Phi-3-mini-4k-instruct` — ~2.2 GB, fast, good quality
- `meta-llama/Llama-3.2-3B-Instruct` — ~1.8 GB, lower RAM requirement

Both run on devices with 3–4 GB of available RAM.

---

## Troubleshooting

### `codi` not recognised after `pip install -e .`

The venv isn't active or its `Scripts/` folder isn't in PATH.

**Windows:**
```powershell
# Activate venv first
.\codi\Scripts\Activate.ps1
# Then try again
codi
```
Or add the Scripts folder to your system PATH permanently (see [step 7](#7--register-the-global-codi-command)).

**macOS / Linux:**
```bash
source codi/bin/activate
codi
```

### `ModuleNotFoundError: No module named 'codi_agent'`

Old install with the wrong package name. Reinstall cleanly:
```bash
pip uninstall codi-agent -y
pip install -e .
```

### `pip install -e .` fails with `BackendUnavailable`

Wrong build backend in `pyproject.toml`. Make sure it says:
```toml
[build-system]
build-backend = "setuptools.build_meta"
```
Not `setuptools.backends.legacy` (that doesn't exist).

### Token limit / 413 errors

Context window overflowed. Inside Codi:
```
❯ /clear          # wipe session memory
❯ /mode cloud     # switch to a model with larger context
```

### MCP shows `[Tools] 5 native + 0 MCP`

MCP servers failed to load. Common causes:

- **npx not found** — install Node.js
- **uvx not found** — run `pip install uv`
- **Server crashed on startup** — check `codi.log` with `/logs`
- **mcp_servers.json has wrong paths** — re-run `python setup.py`

### Ollama connection refused in hybrid mode

Ollama isn't running. Start it:

**Windows / macOS:** Open the Ollama app or check system tray.

**Linux:**
```bash
ollama serve &
```

Codi will automatically fall back to cloud in hybrid mode if Ollama can't be reached.

### `UnicodeDecodeError: charmap codec can't decode` (Windows)

Fixed in the latest `tools.py` — all subprocess calls now use `encoding="utf-8", errors="replace"`. Make sure you have the latest version.

### HuggingFace rate limit warnings

```
Warning: You are sending unauthenticated requests to the HF Hub
```

This is harmless but slows down the first index. To silence it, add a free HF token:
1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings → Access Tokens → New token (read permissions is enough)
3. Add to `.env`:
   ```env
   HF_TOKEN=hf_...
   ```

### `setup.py` fails on Python dependencies

Try installing without the build cache:
```bash
pip install -r requirements.txt --no-cache-dir
```

If torch fails specifically:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## File structure

```
coding-agent/
├── main.py            # CLI entry point — launched by `codi` command
├── agent.py           # LangGraph plan → execute → verify → synthesize loop
├── tools.py           # Native tools + MCP loader
├── llm_factory.py     # Routes to local / air / cloud LLM
├── config.py          # All settings (MODE, models, providers)
├── config_loader.py   # Reads API keys from .env
├── context_trimmer.py # Keeps token usage within budget
├── indexer.py         # Per-project ChromaDB indexing
├── mcp_manager.py     # Persistent MCP server sessions
├── mcp_servers.json   # Which MCP servers are enabled
├── memory.py          # Session memory with compression
├── refiner.py         # Prompt rewriting
├── logger.py          # Structured event log → codi.log
├── log_viewer.py      # Live TUI telemetry dashboard
├── setup.py           # One-time setup script
├── pyproject.toml     # Package definition (`pip install -e .`)
├── requirements.txt   # Python dependencies
├── .env               # API keys (never commit)
└── chroma_db/         # Per-project vector indexes (auto-created)
```

---

## Quick reference

```bash
# First time
git clone https://github.com/AbheyTiwari/CODI.git
cd CODI/coding-agent
python -m venv codi && source codi/bin/activate   # or .\codi\Scripts\Activate.ps1
python setup.py
pip install -e .

# Every time you open a new project
cd /your/project
codi

# Inside Codi
❯ /help            # all commands
❯ /mode hybrid     # switch mode
❯ /mcp             # list MCP servers
❯ /tools           # list all tools
❯ /clear           # wipe memory
❯ /logs            # live telemetry
❯ /quit            # exit
```
