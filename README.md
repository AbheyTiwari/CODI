# Codi — Local AI Coding Agent

Codi is a lightweight Python CLI agent for code-aware development workflows. It uses local or cloud LLMs, semantic search, and tool execution to help you inspect, edit, and automate code in any directory.

## 🚀 Quick Start

1. Open PowerShell or Command Prompt in the project folder.
2. Create and activate a Python virtual environment.

   PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Command Prompt:
   ```cmd
   python -m venv .venv
   .\.venv\Scriptsctivate.bat
   ```

3. Install the package and dependencies:
   ```powershell
   pip install -e .
   ```

4. Optionally copy environment settings and add API keys:
   ```powershell
   copy .env.example .env
   ```

## 🧩 Run Codi from any directory

After installing this package into your active environment, you can run Codi from any working directory by calling:

```powershell
codi
```

or in Command Prompt:

```cmd
codi
```

This works in any directory as long as the virtual environment is active and the `codi` entry point is installed.

## ✅ What Codi can do

- Start an interactive AI coding assistant
- Read and write files
- Execute shell commands safely
- Search the repository using semantic embeddings
- Index a code folder into `chroma_db/`
- Load MCP plugin tools from `mcp_servers.json`
- Maintain session memory and command history

## 📌 Main Commands

- `/index <path>` — index a folder for semantic search
- `/mcp` — show configured MCP servers
- `/mcp on <name>` — enable an MCP server
- `/mcp off <name>` — disable an MCP server
- `/logs` — view runtime logs
- `/clear` — clear session memory
- `/history` — display memory history
- `/help` — show help text
- `/quit` — exit Codi

## ⚙️ Configuration

Settings are stored in `config.py` and optionally in `.env`.

Key configuration options:
- `MODE` — `local`, `cloud`, or `remote`
- `CLOUD_PROVIDER` — `groq`, `openai`, `anthropic`, or `gemini`
- API keys via `.env` or environment variables

Supported environment variables:
- `CODI_GEMINI_API_KEY`
- `CODI_GROQ_API_KEY`
- `CODI_OPENAI_API_KEY`
- `CODI_ANTHROPIC_API_KEY`

## 🧠 Project Structure

- `main.py` — CLI entry point and command loop
- `agent.py` — agent workflow and tool orchestration
- `llm_factory.py` — local/cloud/remote model selection
- `refiner.py` — prompt refinement logic
- `tools.py` — built-in tools and MCP tool loading
- `indexer.py` — ChromaDB index management
- `memory.py` — session memory persistence
- `logger.py` — event logging
- `log_viewer.py` — live log viewer
- `mcp_client.py` / `mcp_manager.py` — MCP connection management

## 🔧 Install globally in a virtual env

To make `codi` available everywhere while using the same virtual environment, activate the env and install the package:

```powershell
pip install -e .
```

Then run from any folder:

```powershell
codi
```

## 📚 Notes and Tips

- If `codi` is not recognized, ensure your virtual environment is activated.
- Use `/index .` to index the current repository.
- Keep `.env` out of version control.
- `.gitignore` already excludes `.env`, `.agent_history`, `codi.log`, `chroma_db/`, and `__pycache__/`.

## 📦 Package Entry Point

This project exposes the `codi` console script via `pyproject.toml`:

```toml
[project.scripts]
codi = "main:main"
```

That means `codi` launches the same CLI as `python main.py`.
