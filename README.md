# Codi — Local / Cloud AI Coding Agent

Codi is a Python CLI agent for codebase-aware development workflows. It combines model-driven reasoning, tool execution, semantic search, and optional MCP plugins for flexible developer automation.

## 🚀 What Codi does

- Runs as a CLI assistant from `main.py`
- Uses prompt refinement for action-oriented tasks
- Decides between direct Q&A and tool-enabled workflows
- Supports file editing, shell command execution, and semantic code search
- Indexes code into ChromaDB for RAG-style search
- Loads external MCP tools from `mcp_servers.json`
- Maintains session memory and logs agent events to `codi.log`

## 🧩 Core capabilities

- `read_file(path)` — read any file content
- `write_file(path, content)` — write file content and create directories
- `run_command(command)` — execute shell commands safely
- `search_codebase(query)` — semantic search over indexed code
- `list_files(dir_path)` — list files recursively
- `/index <path>` — index a repository into `chroma_db`
- MCP plugin support for additional toolsets
- `/logs` — live telemetry viewer

## 📥 Installation

1. Create and activate a virtual environment.

   Windows PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and populate your API keys.

   ```powershell
   copy .env.example .env
   ```

## ⚙️ Configuration

### `config.py`

- `MODE` controls the LLM route:
  - `local` — use Ollama via `langchain_ollama`
  - `cloud` — use a remote cloud provider
  - `remote` — forward requests to a remote LLM server
- `CLOUD_PROVIDER` selects between `groq`, `openai`, `anthropic`, and `gemini`
- Model names are configured separately for refiner and coder roles

### API keys

Codi loads API keys in this order:
1. environment variables
2. `config.json` if present

Supported env vars:
- `CODI_GEMINI_API_KEY`
- `CODI_GROQ_API_KEY`
- `CODI_OPENAI_API_KEY`
- `CODI_ANTHROPIC_API_KEY`

Use `.env` to keep credentials local.

## 💡 Usage

Start the CLI:
```bash
python main.py
```

### Commands

- `/index <path>` — index a directory for semantic search
- `/mcp` — list MCP servers and their enabled state
- `/mcp on <name>` — enable a server in `mcp_servers.json`
- `/mcp off <name>` — disable a server
- `/logs` — open live telemetry
- `/clear` — clear session memory
- `/history` — show current memory
- `/help` — show help text
- `/quit` — exit

### Task input

Codi automatically routes simple questions to a direct response path and action tasks through the tool-enabled agent workflow.

Example:
```text
>> create a CLI command that writes a Python module and updates README.md
```

## 🧠 Architecture

- `main.py` — CLI, command loop, prompt refinement, agent invocation
- `agent.py` — LangGraph workflow, direct vs agent routing, verification and synthesis
- `llm_factory.py` — local/cloud/remote LLM selection
- `refiner.py` — conditional task refinement
- `tools.py` — native tools and MCP tool loading
- `indexer.py` — ChromaDB indexing and persistence
- `memory.py` — session memory tracking and summary compression
- `logger.py` — structured event logging
- `log_viewer.py` — live terminal dashboard
- `mcp_client.py` / `mcp_manager.py` — MCP server connections

## 🔌 MCP Plugins

`mcp_servers.json` defines external MCP server integrations.

- only enabled servers are loaded
- local mode filters MCP toolsets to keep prompt context small
- cloud mode loads all enabled MCP tools

Use `/mcp on <name>` and `/mcp off <name>` to toggle servers at runtime.

## 🛡️ Security and repo hygiene

This repo includes local state and should not expose secrets publicly.

Do not commit:
- `.env`
- `.agent_history`
- `codi.log`
- `chroma_db/`
- `__pycache__/`

A `.gitignore` is included to exclude these files.

## 🧪 Notes

- `refiner.py` rewrites longer action-oriented prompts only
- `tools.py` has a command sandbox guard for dangerous shell patterns
- `indexer.py` skips `.git`, `node_modules`, `.venv`, and `chroma_db`
- `memory.py` compresses old conversations into a summary

