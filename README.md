# Codi — AI Coding Agent

Codi is a system-driven coding agent that runs in your terminal. You point it at any project directory and talk to it in plain English. It reads your code, makes a plan, executes tools in parallel, validates the result, and corrects itself if something goes wrong.

It works fully offline (Ollama), on your phone over Wi-Fi (Air LLM), or through cloud providers (Groq, Anthropic, OpenAI, Gemini).

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [Modes](#modes)
   - [Cloud](#cloud-mode)
   - [Local (Ollama)](#local-mode)
   - [Hybrid](#hybrid-mode)
   - [Air LLM](#air-llm-mode)
6. [API keys](#api-keys)
7. [Commands](#commands)
8. [MCP servers](#mcp-servers)
9. [Project structure](#project-structure)
10. [How the agent loop works](#how-the-agent-loop-works)
11. [Adding a custom tool](#adding-a-custom-tool)
12. [Troubleshooting](#troubleshooting)

---

## How it works

Codi uses a two-LLM architecture with a central Dispatcher:

- **Improver LLM** (fast/cheap) — orchestrates. It reads context, creates a plan, decides the next step each iteration, and writes the final output.
- **Coder LLM** (stronger) — executes. It receives one step at a time and translates it into a JSON action bundle.
- **Dispatcher** — routes. It receives the JSON, runs tools in parallel (local Python functions or MCP servers), and returns structured results.
- **Validator** — checks. After each execution round it decides pass or fail. On fail, the Improver corrects and retries.

No LangGraph. No regex parsing. No guessing. Every decision flows through JSON.

---

## Requirements

- Python 3.10 or higher
- `pip`
- Node.js 18+ and `npx` (required for MCP servers)
- `uvx` — install with `pip install uv` (required for some MCP servers)
- For local mode: [Ollama](https://ollama.com) installed and running
- For cloud mode: an API key for at least one provider

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourname/codi.git
cd codi
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install Codi

```bash
pip install -e .
```

This installs the `codi` command globally in your environment. You can now type `codi` from any directory.

### 4. Create your `.env` file

In the repo root, create a file called `.env`:

```env
# Paste whichever keys you have — leave the rest blank
CODI_GROQ_API_KEY=gsk_...
CODI_ANTHROPIC_API_KEY=sk-ant-...
CODI_OPENAI_API_KEY=sk-...
CODI_GEMINI_API_KEY=AIza...
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
STITCH_API_KEY=...
BRAVE_API_KEY=...
```

---

## Quick start

Navigate to any project you want to work on, then run:

```bash
cd ~/my-project
codi
```

Codi will:
1. Auto-index your project into a local vector database
2. Load all enabled MCP servers
3. Show you a prompt

Then just type what you want:

```
❯ create a FastAPI app with a /health endpoint and write it to app.py
❯ find all TODO comments in this codebase
❯ add pytest tests for the User class in models.py
❯ push all uncommitted changes to github with message "fix auth bug"
❯ what does the parse_config function do?
```

Codi will plan, execute, validate, and tell you exactly what it did.

---

## Modes

Set `MODE` at the top of `config.py`, or switch live with `/mode <name>` while Codi is running.

### Cloud mode

```python
# config.py
MODE = "cloud"
CLOUD_PROVIDER = "groq"   # groq | anthropic | openai | gemini
```

Every LLM call goes to the cloud provider. Best quality and capability. Costs money (except Groq free tier).

**Recommended cloud setup for most users:**

```python
MODE           = "cloud"
CLOUD_PROVIDER = "groq"
REFINER_MODEL_CLOUD = "llama-3.1-8b-instant"
CODER_MODEL_CLOUD   = "llama-3.3-70b-versatile"
```

Groq has a free tier with generous rate limits. Sign up at [console.groq.com](https://console.groq.com).

**To use Anthropic:**

```python
CLOUD_PROVIDER      = "anthropic"
REFINER_MODEL_CLOUD = "claude-haiku-4-5-20251001"
CODER_MODEL_CLOUD   = "claude-sonnet-4-6"
```

**To use OpenAI:**

```python
CLOUD_PROVIDER      = "openai"
REFINER_MODEL_CLOUD = "gpt-4o-mini"
CODER_MODEL_CLOUD   = "gpt-4o"
```

**To use Gemini:**

```python
CLOUD_PROVIDER      = "gemini"
REFINER_MODEL_CLOUD = "gemini-2.0-flash"
CODER_MODEL_CLOUD   = "gemini-2.5-pro"
```

---

### Local mode

100% offline. Ollama only. Zero API spend. No internet needed.

**Step 1 — Install Ollama:**

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download installer from https://ollama.com
```

**Step 2 — Pull models:**

Minimum setup (low RAM):
```bash
ollama pull phi3:mini           # ~2.2 GB — fast planner
ollama pull qwen2.5-coder:7b    # ~4.7 GB — code generation
```

High-end setup (16 GB+ RAM or GPU):
```bash
ollama pull phi3:mini
ollama pull qwen2.5-coder:14b   # ~9.0 GB — better code quality
ollama pull deepseek-r1:8b      # ~5.0 GB — deep reasoning tasks
```

**Step 3 — Set config:**

```python
# config.py
MODE = "local"
REFINER_MODEL_LOCAL = "phi3:mini"
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"
```

**Step 4 — Start Ollama and run Codi:**

```bash
ollama serve   # keep this running in a separate terminal
codi
```

**Model recommendations by task:**

| Use case | Refiner | Coder |
|---|---|---|
| Low-end machine (8 GB RAM) | `phi3:mini` | `qwen2.5-coder:7b` |
| Mid-range (16 GB RAM) | `phi3:mini` | `qwen2.5-coder:14b` |
| GPU / high-end | `qwen2.5:3b` | `deepseek-coder:33b` |
| General purpose | `phi3:mini` | `llama3.1:8b` |

---

### Hybrid mode

Ollama handles most tasks. When Ollama is unavailable or the context overflows, it automatically escalates to cloud.

```python
MODE                   = "hybrid"
CLOUD_PROVIDER         = "groq"
HYBRID_TOKEN_LIMIT     = 3500    # escalate when context exceeds this
HYBRID_REQUIRE_CONFIRM = False   # set True to prompt before every cloud call
```

Fallback chain: **Ollama → Air LLM → Cloud**

---

### Air LLM mode

Run LLMs on your Android phone over local Wi-Fi. Useful when you have no internet and your laptop is too slow for Ollama.

**Step 1 — Install Air LLM on Android:**

Download from [Google Play](https://play.google.com/store/apps/details?id=com.airlm.app).

**Step 2 — Load a model in the app:**

`phi-3-mini-4k-instruct.Q4_K_M` is a good pick. Download it inside the app.

**Step 3 — Start the server in the app:**

The app will show you a LAN IP and port, e.g. `http://192.168.1.42:8080`.

**Step 4 — Set config:**

```python
# config.py
MODE              = "air"
AIR_LLM_URL       = "http://192.168.1.42:8080"   # your phone's address
AIR_LLM_REFINER_MODEL = "phi3-mini"
AIR_LLM_CODER_MODEL   = "phi3-mini"
AIR_LLM_TIMEOUT   = 180    # phones are slower — be patient
```

**Step 5:**

```bash
codi
```

Your phone and laptop must be on the same Wi-Fi network.

---

## API keys

The cleanest way to set keys is in the `.env` file in the repo root. Codi loads it automatically on startup.

```env
CODI_GROQ_API_KEY=gsk_...
CODI_ANTHROPIC_API_KEY=sk-ant-...
CODI_OPENAI_API_KEY=sk-...
CODI_GEMINI_API_KEY=AIza...
```

You can also set them as shell environment variables:

```bash
export CODI_GROQ_API_KEY=gsk_...
```

Or hard-code them in `config.py` (not recommended — do not commit keys):

```python
GROQ_API_KEY = "gsk_..."
```

**Where to get keys:**

| Provider | URL | Notes |
|---|---|---|
| Groq | [console.groq.com](https://console.groq.com) | Free tier, very fast |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | Pay per token |
| OpenAI | [platform.openai.com](https://platform.openai.com) | Pay per token |
| Gemini | [aistudio.google.com](https://aistudio.google.com) | Free tier available |

---

## Commands

These are typed directly at the `❯` prompt while Codi is running.

| Command | What it does |
|---|---|
| `/help` | Show all commands |
| `/mode` | Show current mode and all available modes |
| `/mode cloud` | Switch to cloud mode (live, no restart needed) |
| `/mode local` | Switch to local Ollama mode |
| `/mode hybrid` | Switch to hybrid mode |
| `/mode air` | Switch to Air LLM mode |
| `/mcp` | List all MCP servers and their ON/OFF status |
| `/mcp on <name>` | Enable an MCP server (restart to apply) |
| `/mcp off <name>` | Disable an MCP server (restart to apply) |
| `/tools` | List every tool currently loaded (local + MCP) |
| `/index` | Re-index the current project directory |
| `/index /path/to/dir` | Index a specific directory |
| `/clear` | Wipe session memory (helps with token overflow) |
| `/history` | Print the full conversation history for this session |
| `/logs` | Open the live telemetry dashboard |
| `/quit` or `/exit` | Exit Codi |

**Tip:** If Codi starts giving confused responses mid-session, run `/clear` to wipe context and start fresh.

---

## MCP servers

MCP (Model Context Protocol) servers extend Codi with external capabilities. They are configured in `mcp_servers.json`.

### Enabled by default

| Server | What it does |
|---|---|
| `filesystem` | Read and write files anywhere on disk |
| `sqlite` | Query SQLite databases in your project |
| `git` | `git status`, `git diff`, `git commit`, `git log` |
| `memory` | Persistent knowledge graph across sessions |
| `sequential-thinking` | Forces structured step-by-step reasoning |
| `github` | Read/write repos, issues, pull requests |
| `fetch` | Fetch any URL and return its content |
| `playwright` | Full browser automation |
| `stitch` | Google Stitch — generate UI/UX code |

### Disabled by default (enable as needed)

| Server | What it does | Key needed |
|---|---|---|
| `brave-search` | Web search | `BRAVE_API_KEY` |
| `postgres` | Query PostgreSQL databases | Connection string in config |
| `mysql` | Query MySQL databases | Credentials in config |
| `redis` | Redis key inspection | Redis URL in config |
| `docker` | Manage Docker containers | Docker running locally |
| `kubernetes` | kubectl operations | kubeconfig present |
| `gitlab` | GitLab repos and CI/CD | `GITLAB_PERSONAL_ACCESS_TOKEN` |
| `puppeteer` | Browser automation (alternative to playwright) | — |
| `python-sandbox` | Execute Python in a safe sandbox | — |
| `sentry` | Read errors and stack traces | Sentry auth token |
| `prometheus` | Query metrics | Prometheus running locally |
| `openapi` | Call any API from its OpenAPI spec | API spec URL |

### Enable or disable a server

**Option 1 — Live command (restart required):**
```
❯ /mcp on brave-search
❯ /mcp off playwright
```

**Option 2 — Edit `mcp_servers.json` directly:**
```json
"brave-search": {
  "enabled": true,
  "env": {
    "BRAVE_API_KEY": "your-key-here"
  }
}
```

### GitHub MCP setup

The GitHub server is enabled by default but requires a token:

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Generate a Personal Access Token with `repo` scope
3. Add to `.env`:

```env
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...
```

### Google Stitch setup

Stitch generates UI/UX code from descriptions.

1. Get a key from [Google AI Studio](https://aistudio.google.com)
2. Add to `.env`:

```env
STITCH_API_KEY=AIza...
```

### Brave Search setup

2000 free searches/month at [brave.com/search/api](https://brave.com/search/api).

1. Get your key
2. Add to `.env`:

```env
BRAVE_API_KEY=BSA...
```

3. Enable the server:

```
❯ /mcp on brave-search
```

---

## Project structure

```
codi/
├── agent.py              Thin agent shell — explicit loop, no LangGraph
├── dispatcher.py         Central router — receives JSON, runs tools in parallel
│
├── core/
│   ├── improver.py       Orchestrator LLM: plans, drives loop, summarizes
│   ├── planner.py        Routes simple Q&A vs execution, refines input
│   ├── executor.py       Coder LLM: step → JSON action bundle
│   └── validator.py      Explicit pass/fail validation after each step
│
├── tools/
│   ├── registry.py       Unified tool registry — dispatcher calls this
│   ├── local/
│   │   ├── file_tools.py     read_file, write_file, list_files
│   │   ├── shell_tools.py    run_command
│   │   └── search_tools.py   search_codebase
│   └── mcp/
│       └── mcp_tools.py      Wraps MCP servers as plain callables
│
├── state/
│   └── temp_db.py        RunState — owns all data for a single run
│
├── main.py               Terminal UI — renders output, handles commands
├── cli.py                Entry point for the `codi` command
├── config.py             Mode, models, providers — edit this to configure
├── config_loader.py      Reads API keys from env / .env file
├── context_trimmer.py    Keeps token usage under budget
├── indexer.py            Builds and queries the ChromaDB vector index
├── llm_factory.py        Returns the right LLM for local/cloud/air/hybrid
├── logger.py             Appends JSON events to codi.log
├── log_viewer.py         Live telemetry dashboard (/logs command)
├── mcp_manager.py        Connects to MCP servers and loads their tools
├── mcp_servers.json      MCP server configuration — edit to add/remove servers
├── memory.py             Per-session conversation history with compression
├── pyproject.toml        Package definition and dependencies
└── .env                  Your API keys — create this, never commit it
```

---

## How the agent loop works

Understanding this helps you write better prompts and debug when things go wrong.

```
Your input
    │
    ▼
Planner — is this a simple question or does it need tools?
    │
    ├── Simple (what is X, explain Y) ──► Direct answer, no tools
    │
    └── Action (create, build, fix, run...) ──► Execution loop
            │
            ▼
        Improver reads context
        (lists your files, searches codebase)
            │
            ▼
        Improver creates a plan
        (JSON: {plan, steps[]})
            │
            ▼
        ┌─── LOOP ───────────────────────────────┐
        │                                        │
        │  Improver → "what's the next step?"    │
        │                                        │
        │  Executor (Coder LLM) translates step  │
        │  into JSON action bundle:              │
        │    {action: tool_call,                 │
        │     tools: [{name, args}, ...]}        │
        │                                        │
        │  Dispatcher runs tools in parallel     │
        │  (local Python or MCP server)          │
        │                                        │
        │  Results stored in RunState            │
        │                                        │
        │  Validator checks: pass or fail?       │
        │    pass ──► exit loop                  │
        │    fail ──► Improver generates         │
        │             correction ──► retry       │
        │                                        │
        └────────────────────────────────────────┘
            │
            ▼
        Improver writes final summary
            │
            ▼
        Output rendered to terminal
```

**Max iterations:** 8 by default. Change in `state/temp_db.py`:
```python
@dataclass
class RunState:
    max_iterations: int = 8   # ← change this
```

---

## Adding a custom tool

Tools are plain Python functions that take a `dict` and return a `str`. Here's how to add one.

**Step 1 — Write the function:**

Create or edit a file in `tools/local/`:

```python
# tools/local/my_tools.py

def count_lines(args: dict) -> str:
    """Count lines in a file."""
    import os
    path = args.get("path", "")
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    with open(path) as f:
        count = sum(1 for _ in f)
    return f"{count} lines in {path}"


def register_my_tools(registry):
    registry.register_local("count_lines", count_lines)
```

**Step 2 — Register it in the registry loader:**

Edit `tools/registry.py`, inside the `load_all` method:

```python
def load_all(self, mode: str = "cloud") -> "ToolRegistry":
    from tools.local.file_tools   import register_file_tools
    from tools.local.shell_tools  import register_shell_tools
    from tools.local.search_tools import register_search_tools
    from tools.local.my_tools     import register_my_tools   # ← add this
    from tools.mcp.mcp_tools      import register_mcp_tools

    register_file_tools(self)
    register_shell_tools(self)
    register_search_tools(self)
    register_my_tools(self)                                  # ← add this
    register_mcp_tools(self, mode=mode)
    return self
```

**Step 3 — Restart Codi.** The tool is now available. You can verify with `/tools`.

That's it. The Coder LLM will see the tool name and its docstring and use it when relevant.

---

## Troubleshooting

### `codi: command not found`

You installed into a virtual environment that isn't active. Run:

```bash
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

Or reinstall:

```bash
pip install -e .
```

### Ollama errors / connection refused

Make sure Ollama is running:

```bash
ollama serve
```

And the models are actually pulled:

```bash
ollama list
```

If you changed models in `config.py`, make sure to pull them first:

```bash
ollama pull phi3:mini
ollama pull qwen2.5-coder:7b
```

### MCP server fails to connect

Check that `npx` and `uvx` are installed:

```bash
npx --version
uvx --version    # install with: pip install uv
```

Disable a broken server while you debug:

```
❯ /mcp off <server-name>
```

Then restart Codi.

### Token limit / context overflow

If you see a warning about tokens, or responses become confused:

```
❯ /clear
```

This wipes session memory. Then re-state your task.

For consistently large projects, switch to cloud mode which has a bigger context window:

```
❯ /mode cloud
```

### Codi loops without doing anything

This usually means the Coder LLM doesn't support the task well enough. Try switching to a stronger model or cloud mode:

```
❯ /mode cloud
```

Or break your request into smaller steps.

### Check the logs

```
❯ /logs
```

This opens a live log viewer showing every tool call, LLM decision, and error. Press `Ctrl+C` to exit the viewer.

The raw log file is at `codi.log` in the repo root. Each line is a JSON event.

### Re-index the project

If Codi seems unaware of recent changes to your codebase:

```
❯ /index
```

---

## Environment variables reference

| Variable | Description |
|---|---|
| `CODI_GROQ_API_KEY` | Groq API key |
| `CODI_ANTHROPIC_API_KEY` | Anthropic API key |
| `CODI_OPENAI_API_KEY` | OpenAI API key |
| `CODI_GEMINI_API_KEY` | Gemini API key |
| `CODI_WORKING_DIR` | Auto-set to the directory you ran `codi` from |
| `CODI_CHROMA_DIR` | Auto-set to the vector DB path for your project |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub MCP server token |
| `STITCH_API_KEY` | Google Stitch MCP server key |
| `BRAVE_API_KEY` | Brave Search MCP server key |

All of these can be set in `.env` in the repo root or as shell environment variables.

---

## Tips for best results

**Be specific about files and frameworks.** Codi respects your choices — if you say FastAPI, it will use FastAPI, not Flask.

**For multi-file tasks**, break them into steps. One clear request per session works better than one massive request.

**Use `/clear` liberally.** Session memory has limits. If a long session is producing degraded output, clear and restate.

**Local mode works best for focused tasks** — write a function, fix a bug, read a file. For complex multi-step reasoning, cloud mode is more reliable.

**Check `/tools` after startup** to confirm all your MCP servers loaded. If a server failed, it will be absent from the list.
