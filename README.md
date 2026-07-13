# ⚽ Codi — AI Coding Agent

**Codi** is a system-driven, terminal-native coding agent. Point it at any project directory, talk to it in plain English, and it will read your code, form a plan, execute tools (locally and via MCP), validate the result, and correct itself automatically when something goes wrong — all without leaving your terminal.

It runs fully offline (Ollama / llama.cpp), on your phone over Wi-Fi (Air LLM), or through cloud providers (Groq, Anthropic, OpenAI, Gemini) — switchable live, mid-session, with `/mode`.

```
  ██████╗ ██████╗ ██████╗ ██╗
 ██╔════╝██╔═══██╗██╔══██╗██║
 ██║     ██║   ██║██║  ██║██║
 ██║     ██║   ██║██║  ██║██║
 ╚██████╗╚██████╔╝██████╔╝██║
  ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝
```

---

## Table of Contents

1. [Why Codi](#why-codi)
2. [How it works](#how-it-works)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Modes](#modes)
   - [Cloud](#cloud-mode)
   - [Local (Ollama)](#local-mode)
   - [Hybrid](#hybrid-mode)
   - [Air LLM](#air-llm-mode)
   - [llama.cpp](#llamacpp-mode)
7. [API keys](#api-keys)
8. [Commands](#commands)
9. [How the agent loop works](#how-the-agent-loop-works)
10. [Intent routing (qa / read / edit / build)](#intent-routing-qa--read--edit--build)
11. [Surgical file editing](#surgical-file-editing)
12. [Running commands in a separate terminal](#running-commands-in-a-separate-terminal)
13. [MCP servers](#mcp-servers)
14. [Football theming (CLI)](#football-theming-cli)
15. [Project structure](#project-structure)
16. [Adding a custom tool](#adding-a-custom-tool)
17. [Configuration reference](#configuration-reference)
18. [Environment variables reference](#environment-variables-reference)
19. [Troubleshooting](#troubleshooting)
20. [Tips for best results](#tips-for-best-results)
21. [Contributing](#contributing)
22. [License](#license)

---

## Why Codi

Most AI coding tools either:
- **Rewrite entire files** for a one-line change, destroying unrelated work, or
- **Hide their reasoning** behind a black-box agent loop with no visibility into what's actually happening, or
- **Require a cloud subscription** even for simple, local, offline-capable tasks.

Codi is built around three explicit principles instead:

- **Surgical over destructive.** Codi edits the smallest possible region of a file — a text span, a line range, an append — instead of regenerating the whole thing whenever it can avoid it.
- **Transparent over magical.** No LangGraph, no hidden state machines. Every decision — plan, step, tool call, validation, correction — is a JSON object you can see in `/logs`.
- **Yours over rented.** Run it 100% offline against your own Ollama models with zero API spend, or escalate to cloud only when you choose to.

---

## How it works

Codi uses a **two-LLM architecture** with a central **Dispatcher**:

| Component | Role |
|---|---|
| **Improver LLM** (fast/cheap) | Orchestrates. Reads context, creates a plan, decides the next step each iteration, writes the final summary. |
| **Coder LLM** (stronger) | Executes. Receives one step at a time and translates it into a JSON action bundle — or, for file edits, a precise text/line-range operation. |
| **Dispatcher** | Routes. Normalizes the JSON, runs tools in parallel (local Python functions or MCP servers), returns structured results. |
| **Validator** | Checks. After each round, decides pass/fail deterministically first (cheap, fast, no LLM cost) and only escalates to an LLM semantic check if every deterministic gate already passed. |

No regex-parsed prose. No guessing. Every hop is JSON in, JSON out.

```
Your input
    │
    ▼
Planner — qa / read / edit / build?
    │
    ├── qa    ──────────────────────► Direct answer, no tools
    ├── read  ──────────────────────► Read-only context, answer, never writes
    ├── edit  ──────────────────────► One direct targeted call, falls back to build
    │
    └── build ──────────────────────► Full pipeline:
            │
            ▼
        Improver reads context (list_files, search_codebase)
            │
            ▼
        Improver creates a plan → writes plan.md → waits for 'y'
            │
            ▼
        ┌─── LOOP ─────────────────────────────────────────────┐
        │  Improver: "what's the next step?"                   │
        │  Executor (Coder LLM): step → JSON action bundle      │
        │    - text-match edit_file (old/new)                   │
        │    - line-range edit_file (replace/delete/insert)      │
        │    - content-first write_file (new/large files)        │
        │    - additive append (safe fallback)                  │
        │  Dispatcher: runs tools in parallel                   │
        │  Validator: pass? exit loop. fail? Improver corrects.  │
        └────────────────────────────────────────────────────────┘
            │
            ▼
        Improver writes final, deterministic summary
            │
            ▼
        Rendered to terminal (with football-themed live status)
```

**Max iterations:** 8 by default (`state/temp_db.py` → `RunState.max_iterations`).

---

## Requirements

- Python 3.10+
- `pip`
- Node.js 18+ and `npx` (for MCP servers)
- `uvx` — install with `pip install uv` (for some MCP servers)
- For local mode: [Ollama](https://ollama.com) installed and running, **or** a running [llama.cpp](https://github.com/ggerganov/llama.cpp) server
- For cloud mode: an API key for at least one provider
- **Windows only** (for the external-shell feature): PowerShell available on `PATH` (default on Windows 10/11)
- **macOS** (for the external-shell feature): Terminal.app available (default)
- **Linux** (for the external-shell feature): one of `gnome-terminal`, `konsole`, `xterm`, or `x-terminal-emulator` on `PATH`

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

This installs the `codi` command globally in your environment (see `cli.py` / `setup.cfg`). You can now type `codi` from any directory.

### 4. Create your `.env` file

In the repo root:

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

None of these are required to get started — local mode needs zero keys.

---

## Quick start

```bash
cd ~/my-project
codi
```

Codi will:
1. Auto-index your project into a local ChromaDB vector database (skipped automatically for >5000 files).
2. Load every MCP server marked `"enabled": true` in `mcp_servers.json`.
3. Show you a prompt: `❯`

Then just talk to it:

```
❯ create a FastAPI app with a /health endpoint and write it to app.py
❯ find all TODO comments in this codebase
❯ replace lines 40 to 55 in server.py with a proper error handler
❯ run npm install in a separate terminal window
❯ push all uncommitted changes to github with message "fix auth bug"
❯ what does the parse_config function do?
```

For anything that touches files (`build` intent), Codi writes a plan to `plan.md`, shows it to you, and **waits for confirmation**:

```
❯ build a todo app with FastAPI and SQLite
  Plan written to /home/you/my-project/plan.md. Review it, then type 'y' to run it,
  or give me a new instruction to replan.
❯ y
  → resuming confirmed plan
  ...
```

---

## Modes

Set `MODE` at the top of `config.py`, or switch live with `/mode <name>` while Codi is running (no restart needed for most modes).

### Cloud mode

```python
# config.py
MODE = "cloud"
CLOUD_PROVIDER = "groq"   # groq | anthropic | openai | gemini
```

Every LLM call goes to the cloud provider. Best quality and capability. Costs money (except Groq's free tier).

**Recommended cloud setup:**
```python
MODE = "cloud"
CLOUD_PROVIDER = "groq"
REFINER_MODEL_CLOUD = "llama-3.1-8b-instant"
CODER_MODEL_CLOUD   = "llama-3.3-70b-versatile"
```

| Provider | Refiner example | Coder example |
|---|---|---|
| Groq | `llama-3.1-8b-instant` | `llama-3.3-70b-versatile` |
| Anthropic | `claude-haiku-4-5-20251001` | `claude-sonnet-4-6` |
| OpenAI | `gpt-4o-mini` | `gpt-4o` |
| Gemini | `gemini-2.0-flash` | `gemini-2.5-pro` |

### Local mode

100% offline. Ollama only. Zero API spend. No internet needed.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh    # Linux/macOS
# Windows: download from https://ollama.com

# Pull models
ollama pull phi3:mini           # ~2.2 GB — fast planner/refiner
ollama pull qwen2.5-coder:7b    # ~4.7 GB — code generation
```

```python
# config.py
MODE = "local"
REFINER_MODEL_LOCAL = "phi3:mini"
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"
```

```bash
ollama serve   # keep running in a separate terminal
codi
```

| Machine | Refiner | Coder |
|---|---|---|
| Low-end (8 GB RAM) | `phi3:mini` | `qwen2.5-coder:7b` |
| Mid-range (16 GB RAM) | `phi3:mini` | `qwen2.5-coder:14b` |
| GPU / high-end | `qwen2.5:3b` | `deepseek-coder:33b` |

### Hybrid mode

Ollama handles most tasks; escalates to cloud automatically when Ollama is unreachable, or falls back through llama.cpp → Air LLM → cloud in that order.

```python
MODE                   = "hybrid"
CLOUD_PROVIDER         = "groq"
HYBRID_TOKEN_LIMIT     = 3500
HYBRID_REQUIRE_CONFIRM = False
```

Fallback chain: **Ollama → llama.cpp (localhost) → Air LLM → Cloud**

### Air LLM mode

Run models on your Android phone over local Wi-Fi — useful with no internet and a laptop too slow for local inference.

1. Install [Air LLM](https://play.google.com/store/apps/details?id=com.airlm.app) on Android.
2. Load a GGUF model in-app (`phi-3-mini-4k-instruct.Q4_K_M` recommended).
3. Start the server — note the LAN IP:port shown.
4. Configure:
   ```python
   MODE = "air"
   AIR_LLM_URL = "http://192.168.1.42:8080"
   AIR_LLM_REFINER_MODEL = "phi3-mini"
   AIR_LLM_CODER_MODEL   = "phi3-mini"
   AIR_LLM_TIMEOUT = 180
   ```
5. `codi` — phone and laptop must share the same Wi-Fi network.

### llama.cpp mode

Point Codi at a local `llama-server` OpenAI-compatible endpoint:

```python
MODE = "llamacpp"
LLAMACPP_URL = "http://127.0.0.1:8080"
LLAMACPP_REFINER_MODEL = "qwen2.5-coder-7b"
LLAMACPP_CODER_MODEL   = "qwen2.5-coder-7b"
LLAMACPP_TIMEOUT = 120
```

---

## API keys

Preferred: `.env` in the repo root (loaded automatically). Also supported: shell environment variables, or hardcoding in `config.py` (not recommended — never commit real keys).

| Provider | Get a key at |
|---|---|
| Groq | [console.groq.com](https://console.groq.com) — free tier, very fast |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | [platform.openai.com](https://platform.openai.com) |
| Gemini | [aistudio.google.com](https://aistudio.google.com) — free tier available |

---

## Commands

Typed directly at the `❯` prompt:

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/mode` | Show current mode and all available modes |
| `/mcp` | List all MCP servers and ON/OFF status |
| `/mcp on <name>` | Enable an MCP server (restart to apply) |
| `/mcp off <name>` | Disable an MCP server (restart to apply) |
| `/tools` | List every tool currently loaded (local + MCP) |
| `/index [path]` | Index a directory and set it as the working dir |
| `/clear` | Wipe session memory (fixes context overflow / confused responses) |
| `/history` | Print the full conversation history |
| `/logs` | Open the live telemetry dashboard (every tool call, LLM decision, error) |
| `cd <path>` | Change Codi's project directory (re-indexes automatically) |
| `pwd` | Show Codi's current project directory |
| `/quit` / `/exit` | Exit Codi (cleans up session memory, MCP connections, and `plan.md`) |

---

## How the agent loop works

Understanding this helps you write better prompts and debug faster.

1. **Planner** classifies your input into `qa`, `read`, `edit`, or `build` (see [Intent routing](#intent-routing-qa--read--edit--build) below).
2. For `build` tasks, the **Improver** reads context (`list_files`, `search_codebase`, and reads any just-created boilerplate files), then produces a plan of 2–5 concrete steps.
3. The plan is written to `plan.md` and Codi **waits for your confirmation** (`y`) before touching anything.
4. Once confirmed, the loop runs:
   - Improver picks the next step.
   - **Executor** (Coder LLM) turns it into a precise action — see [Surgical file editing](#surgical-file-editing) for how it picks between text-match, line-range, content-first, and additive-append strategies.
   - **Dispatcher** runs the tool(s), normalizing common LLM JSON mistakes (missing `tools[]` wrapper, `content_lines` arrays, truncated JSON, etc.) before execution.
   - **Validator** runs deterministic checks first (file write success, syntax checks, framework-contamination checks, Java compile checks) and only calls an LLM for semantic verification if everything deterministic already passed.
   - On failure, the Improver generates a targeted correction and the loop retries (up to `max_iterations`, default 8).
5. A deterministic (non-LLM) summary is produced from the actual tool results — it only ever reports files that were *successfully* written, never guesses.

---

## Intent routing (qa / read / edit / build)

Every input is classified before anything else runs (`core/planner.py`):

| Intent | Behavior |
|---|---|
| `qa` | Plain question — answered directly, zero tool calls. E.g. *"what is a hash map?"* |
| `read` | Read-only investigation — reads/searches files, answers, **never writes**. E.g. *"explain what auth.py does"* |
| `edit` | Single targeted change to one (or two) existing files — tries one direct Executor call before falling back to the full `build` pipeline. E.g. *"fix the null check in parse_config"* |
| `build` | Multi-file creation, scaffolding, or anything ambiguous — full plan → confirm → execute → validate loop. |

Routing also recognizes typos (fuzzy keyword matching), broad-scope phrases (*"debug the entire website"* → `build`, not a narrow single-file `edit`), and explicit file/path mentions.

---

## Surgical file editing

Codi never regenerates a whole file for a small change unless it has to. `edit_file` supports four strategies, and the Executor picks automatically based on the step's wording:

For exact code navigation, Codi keeps ChromaDB for semantic search and a local SQLite index at `.codi/code_index.sqlite3` for exact file paths, declarations, and identifier references. SQLite ships with Python; no MySQL or separate database installation is needed. Before a variable or symbol rename, the Coder is instructed to use `find_symbol` and `find_references`, then edit only a verified span. Ambiguous text replacements are rejected instead of silently changing the first match.

### 1. Text-match replace (default for small, precise changes)
```json
{"path": "app.py", "old": "def foo():\n    pass", "new": "def foo():\n    return 42"}
```
Falls back through three levels of whitespace normalization (trailing spaces, CRLF vs LF, indentation collapse) before giving up — this is what makes small-model edits reliable even when they don't reproduce whitespace perfectly.

### 2. Line-range surgical edit (for anything referencing lines, functions, classes, or methods)
Uses `read_file_numbered` first to get **real, verified** line numbers — no guessing:
```json
{"path": "app.py", "replace_lines": {"start": 40, "end": 55, "content": "..."}}
{"path": "app.py", "delete_lines":  {"start": 12, "end": 18}}
{"path": "app.py", "insert_at_line": {"line": 9, "content": "..."}}
```
Example prompts that trigger this path:
```
❯ replace lines 40 to 55 in server.py with a proper error handler
❯ delete the old_helper function, it's around lines 20-30
❯ insert a new validate_input method after line 12
```
If a line range turns out to be stale or out-of-bounds (e.g. the file changed since it was last read), Codi automatically re-reads the file and retries once before falling back to the text-match strategy.

### 3. Content-first (new files, or large HTML/CSS/JS generation)
Used for `write_file`/`create_file` on fresh files, or files matching size/complexity heuristics (e.g. any `.html`, or steps containing words like *"responsive"*, *"dashboard"*, *"animated"*).

### 4. Additive append (safe fallback)
If a text-match `old` genuinely can't be found (and it isn't a line-range case), Codi falls back to appending clearly-scoped new code rather than failing the whole step.

All four report back through the same JSON shape (`{"success": bool, "file_modified": path, ...}`), so the Validator, framework-contamination checker, and correction loop work identically regardless of which strategy fired.

---

## Running commands in a separate terminal

Two shell tools are available to the Coder LLM:

| Tool | Behavior |
|---|---|
| `run_command` | Hidden, in-process, 60s timeout. Good for quick, non-interactive commands (`git status`, `ls`, `pytest`). |
| `run_command_external` | Opens a **separate, visible** terminal window (PowerShell on Windows, Terminal.app on macOS, or your Linux terminal emulator), asks you for **explicit permission first**, tees all output to a log file, and reports the exit code + full output back to Codi once it finishes. |

Example:
```
❯ run npm install in a separate terminal window
```
```
  ┌─ CODI wants to run a command in a separate shell window ─
  │  npm install
  └─────────────────────────────────────────────────────────
  Allow? [y/N]:
```

Type `y` and a real terminal window opens so you can watch it live. Once it exits, Codi reads the captured output back and — if the command failed — automatically triggers the Improver's correction loop, the same way a failed file edit does. No architecture changes were needed for this: `run_command_external` returns errors prefixed with `ERROR:`, which the Dispatcher already treats as a failed step.

**Skip the permission prompt** for unattended/CI runs:
```bash
export CODI_AUTO_APPROVE_SHELL=1      # bash
$env:CODI_AUTO_APPROVE_SHELL="1"      # PowerShell
```

**Change the timeout** (default 300s):
```bash
export CODI_EXTERNAL_SHELL_TIMEOUT=600
```

Both dangerous-pattern blocking (`rm -rf`, `DROP TABLE`, fork bombs, etc.) and the same JSON result shape as `run_command` apply identically here.

---

## MCP servers

MCP (Model Context Protocol) servers extend Codi with external capabilities, configured in `mcp_servers.json`. Connections are opened **once and kept alive** for Codi's entire session (via a persistent `AsyncExitStack` in `mcp_manager.py`) rather than being closed and reopened per call — this is what makes tools like `browser_navigate` actually work reliably run after run.

### Enabled by default

| Server | What it does |
|---|---|
| `filesystem` | Read/write files anywhere on disk |
| `memory` | Persistent cross-session knowledge graph |
| `sequential-thinking` | Forces step-by-step structured reasoning |
| `github` | Read/write repos, issues, pull requests |
| `fetch` | Fetch any URL and return its content |
| `playwright` | Full browser automation (navigate, click, fill, screenshot, evaluate JS) |

### Disabled by default

| Server | What it does | Requires |
|---|---|---|
| `brave-search` | Web search (2000 free/month) | `BRAVE_API_KEY` |
| `sqlite` | Query SQLite databases | — |
| `git` | `git status/diff/commit/log` | Valid git repo |
| `postgres` / `mysql` / `redis` | Query databases | Connection details |
| `docker` / `kubernetes` | Container/cluster ops | Docker/kubeconfig locally |
| `gitlab` | Repos and CI/CD | `GITLAB_PERSONAL_ACCESS_TOKEN` |
| `puppeteer` | Browser automation (alt. to Playwright) | — |
| `sentry` | Read errors/stack traces | Sentry auth token |
| `stitch` | Google Stitch UI/UX generation | `STITCH_API_KEY` |

### Toggle a server

```
❯ /mcp on brave-search
❯ /mcp off playwright
```
...or edit `mcp_servers.json` directly, then restart Codi.

### GitHub setup
1. [github.com/settings/tokens](https://github.com/settings/tokens) → generate a token with `repo` scope.
2. Add `GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...` to `.env`.

### Brave Search setup
1. Get a free key at [brave.com/search/api](https://brave.com/search/api).
2. Add `BRAVE_API_KEY=BSA...` to `.env`.
3. `/mcp on brave-search`

---

## Football theming (CLI) ⚽

Because why not. This is purely cosmetic, purely hardcoded — **zero LLM calls**, so it costs no tokens and no latency:

- A bouncing ⚽ animates across the live status panel title as Codi works.
- Every real status line gets a matching football pun appended via plain keyword lookup (e.g. *"Creating an execution plan." → "Creating an execution plan. — Setting up the formation"*).
- Lives entirely in `football_theme.py` (repo root) and is only ever called from `main.py`'s `LiveRenderer._panel()` — the actual agent status strings emitted by `agent.py` / `improver.py` are completely untouched.

Turning it off is as simple as reverting the two lines in `LiveRenderer._panel()` that call `next_frame()` / `themed_status_line()` — no other file depends on it.

---

## Project structure

```
codi/
├── agent.py                  Thin agent shell — explicit loop, no LangGraph
├── dispatcher.py              Central router — JSON in, tools out, parallel execution
├── football_theme.py          Hardcoded CLI football animation + puns (zero LLM cost)
├── mcp_manager.py              Persistent MCP session lifecycle (AsyncExitStack)
│
├── core/
│   ├── improver.py             Orchestrator LLM: plans, drives loop, summarizes
│   ├── planner.py               Routes qa/read/edit/build, refines input
│   ├── executor.py               Coder LLM: step → precise action (text/line/content/append)
│   ├── validator.py               Deterministic-first, LLM-fallback validation
│   ├── prompts.py                  Canonical system prompts + tool signatures
│   ├── quick_actions.py             Fast-path file creation without the full LLM loop
│   └── validation_utils.py           Framework-contamination detection (AST + regex)
│
├── tools/
│   ├── registry.py               Unified tool registry — Dispatcher calls this
│   ├── local/
│   │   ├── file_tools.py           read_file, read_file_numbered, write_file, edit_file
│   │   ├── shell_tools.py            run_command, run_command_external
│   │   └── search_tools.py            search_codebase (ChromaDB similarity search)
│   └── mcp/
│       └── mcp_tools.py             Wraps MCP StructuredTools as async-safe callables
│
├── state/
│   └── temp_db.py                 RunState — owns all data for a single agent run
│
├── main.py                    Terminal UI — rendering, commands, live status panel
├── cli.py                      `codi` command entry point
├── config.py                    Mode, models, providers
├── config_loader.py               API key resolution (.env / config.json)
├── context_trimmer.py               Token-budget-aware context trimming
├── indexer.py                        ChromaDB vector index builder/query
├── llm_factory.py                     Returns the right LLM for the active mode
├── logger.py                            Appends JSON events to codi.log
├── log_viewer.py                          Live telemetry dashboard (/logs)
├── mcp_client.py                            Standalone MCP client helper
├── mcp_servers.json                           MCP server configuration
├── memory.py                                    Per-session history + compression
├── status_stream.py                               Live status line pub/sub
├── quantized_embeddings.py                          TurboQuant-compressed embeddings
├── pyproject.toml / setup.cfg                         Package definition
└── .env                                                 API keys — never commit
```

---

## Adding a custom tool

Tools are plain Python functions: `fn(args: dict) -> str`.

**1. Write it** in `tools/local/`:
```python
# tools/local/my_tools.py
def count_lines(args: dict) -> str:
    """Count lines in a file."""
    import os
    path = args.get("path", "")
    if not os.path.exists(path):
        return f"ERROR: file not found: {path}"
    with open(path) as f:
        return f"{sum(1 for _ in f)} lines in {path}"

def register_my_tools(registry):
    registry.register_local("count_lines", count_lines)
```

**2. Register it** in `tools/registry.py`'s `load_all`:
```python
from tools.local.my_tools import register_my_tools
# ...
register_my_tools(self)
```

**3. Restart Codi.** Verify with `/tools`. The Coder LLM sees the tool name and docstring automatically — add a signature to `core/prompts.py`'s `_TOOL_SIGNATURES` if you want it to see the exact args schema too.

---

## Configuration reference

All in `config.py`:

```python
MODE = "local"              # local | hybrid | cloud | air | llamacpp
CLOUD_PROVIDER = "groq"     # groq | anthropic | openai | gemini

HYBRID_TOKEN_LIMIT = 3500
HYBRID_REQUIRE_CONFIRM = False

REFINER_MODEL_LOCAL = "phi3:mini"
CODER_MODEL_LOCAL   = "qwen2.5-coder:7b"

REFINER_MODEL_CLOUD = "llama-3.1-8b-instant"
CODER_MODEL_CLOUD   = "llama-3.3-70b-versatile"

AIR_LLM_URL = "http://192.168.1.XXX:8080"
AIR_LLM_TIMEOUT = 180

LLAMACPP_URL = "http://127.0.0.1:8080"
LLAMACPP_TIMEOUT = 120

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_THINK = False   # strip <think> blocks from Qwen3-style reasoning output
```

And in `state/temp_db.py`:
```python
max_iterations: int = 8   # hard cap on the execution loop
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
| `CODI_CHROMA_DIR` | Auto-set to the per-project vector DB path |
| `GITHUB_PERSONAL_ACCESS_TOKEN` | GitHub MCP server token |
| `STITCH_API_KEY` | Google Stitch MCP server key |
| `BRAVE_API_KEY` | Brave Search MCP server key |
| `CODI_AUTO_APPROVE_SHELL` | `1` to skip the `run_command_external` permission prompt |
| `CODI_EXTERNAL_SHELL_TIMEOUT` | Seconds to wait for an external shell command to finish (default 300) |
| `OLLAMA_BASE_URL` | Override the default `http://localhost:11434` |
| `OLLAMA_THINK` | `1` to keep Qwen3-style `<think>` reasoning in raw output |
| `CODI_CALLER_DEBUG` | `1` to include caller file/line metadata in every log entry |

All can be set in `.env` (repo root) or as shell environment variables.

---

## Troubleshooting

### `codi: command not found`
Your virtual environment isn't active:
```bash
source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows
```
Or reinstall: `pip install -e .`

### Ollama errors / connection refused
```bash
ollama serve
ollama list       # confirm your configured models are actually pulled
ollama pull phi3:mini
ollama pull qwen2.5-coder:7b
```

### MCP tool calls fail immediately (`StructuredTool does not support sync invocation`)
Fixed as of the persistent-session update to `mcp_manager.py` / `tools/mcp/mcp_tools.py`. If you still see this, confirm you're on the latest version of both files — the root cause was MCP sessions being closed the instant `load_all()` returned, before any tool could actually be called.

### MCP server fails to connect
```bash
npx --version
uvx --version    # install with: pip install uv
```
Disable the broken server while debugging: `/mcp off <server-name>`, then restart.

### `run_command_external` doesn't open a window
- **Windows**: confirm `powershell` is on `PATH`.
- **macOS**: confirm Terminal.app is installed (default) — Codi calls `open -a Terminal`.
- **Linux**: install one of `gnome-terminal`, `konsole`, or `xterm`. Without any of these, Codi silently falls back to running the command hidden, still logging output to the same temp file.

### Token limit / context overflow
```
❯ /clear
```
For large projects, cloud mode has a bigger context window: `/mode cloud`.

### Codi loops without doing anything
Usually means the Coder LLM isn't strong enough for the task. Try `/mode cloud`, a bigger local model, or break the request into smaller steps.

### Check the logs
```
❯ /logs
```
Live dashboard of every tool call, LLM decision, and error (`Ctrl+C` to exit). Raw JSON-lines file: `codi.log` in the repo root.

### Re-index the project
```
❯ /index
```

---

## Tips for best results

- **Be specific about files and frameworks.** Codi respects explicit choices — say FastAPI, get FastAPI, not Flask (enforced by the framework-lock/contamination checker).
- **Reference line numbers or function/class names explicitly** when you want a surgical edit (*"replace lines 40-55"*, *"delete the old_helper function"*) — this routes to the precise line-range strategy instead of a broader text-match attempt.
- **For multi-file tasks**, break them into focused requests. One clear ask per session beats one massive one.
- **Use `/clear` liberally** — session memory has limits; a long session producing degraded output usually just needs a fresh start.
- **Local mode is great for focused tasks** (write a function, fix a bug, read a file). For complex multi-step reasoning, cloud mode is more reliable.
- **Check `/tools` after startup** to confirm all MCP servers loaded — a failed server simply won't appear in the list.
- **`run_command_external` for anything you want to watch live** — dev servers, installers, long builds. `run_command` for everything else.

---

## Contributing

1. Fork the repo and create a feature branch.
2. Keep new tools as plain `fn(args: dict) -> str` callables in `tools/local/` — no LangChain binding required.
3. Run `/tools` and `/logs` locally to confirm your change registers and behaves as expected before opening a PR.
4. Match the existing JSON result shape (`{"success": bool, ...}`) for any new file/shell tool so the Validator's deterministic checks keep working without modification.
5. Open a PR with a clear description of the behavior change and, where relevant, a before/after `/logs` snippet.

---

## License

MIT Licencse
