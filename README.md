# Codi - Your Local AI Coding Agent

![Codi CLI Banner](https://img.shields.io/badge/Status-Active-brightgreen.svg) ![Local AI](https://img.shields.io/badge/AI-100%25_Local-purple.svg) ![MCP Ready](https://img.shields.io/badge/MCP-Supported-blue.svg)

Codi is a fully autonomous, local-first coding assistant designed to operate directly from your CLI. Leveraging the LangChain ReAct framework, ChromaDB vector indexing, and seamless MCP server extensions, Codi can read your codebase, write files, run terminal commands, and scaffold complete projects autonomously.

## 🚀 Features

- **100% Local & Private by Default**: Runs entirely on your hardware using Ollama (defaulting to `qwen2.5-coder:7b` for reasoning and `llama3:8b` for prompt refinement). No data ever leaves your machine unless you explicitly switch to Cloud Mode.
- **Codebase Aware (`/index`)**: Ingests your entire directory into a ChromaDB vector store. Codi can semantically search your specific architecture to understand context before making edits.
- **Multi-Step Execution (`/plan`)**: Built-in prescriptive task chaining. Instead of generating a massive, easily-broken single output, Codi breaks down tasks (e.g., scaffolding a full page) into distinct semantic steps (Plan -> CSS -> HTML -> JS -> Verify) to guarantee precision.
- **Cognitive Prompt Refinement**: Codi intercepts your chaotic or brief requests and rewrites them using a dedicated LLM (`llama3:8b`), explicitly restricting corner-cutting and placeholders to ensure high-quality outputs.
- **Model Context Protocol (MCP)**: Native support for zero-key plugins. Extend Codi instantly with tools for `fetch`, `git`, `memory`, `sequential-thinking`, and more through `mcp_servers.json`.
- **Cloud LLM Routing**: Easily swap the backend engine to Groq, Anthropic, or OpenAI by switching the `MODE` in `config.py`.

## 📦 Installation & Setup

1. **Clone the repository** and navigate to the project root:
   ```bash
   cd coding-agent
   ```
2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install python-dotenv langchain-ollama langchain-community langchain-core prompt-toolkit rich chromadb sentence-transformers mcp langchain-mcp-adapters
   ```

3. **Install Ollama** (if running locally) and pull the required models:
   ```bash
   ollama pull qwen2.5-coder:7b
   ollama pull llama3:8b
   ```

## ⚙️ Configuration

Codi features dynamic engine routing via `config.py`. 

- To run purely local: `MODE = "local"`
- To run via cloud API for heavier reasoning: `MODE = "cloud"` and set `CLOUD_PROVIDER` (`groq`, `openai`, `anthropic`). Add your API keys to a local `.env` file or directly in `config.py`.

*(Note: Groq's `llama-3.1-70b-versatile` is highly recommended for complex multi-file DOM/CSS scaffolding while in cloud mode).*

## 💻 Usage

Start the cinematic CLI agent:
```bash
python main.py
```

### Built-in Commands:
- `/index <path>` : Recursively scans and vectorizes the codebase at `<path>`. This is required before Codi can semantically search your files.
- `/plan <task>`  : Triggers the multi-step autonomous execution loop. Provide a high-level task, and Codi will write complete files step-by-step.
- `/mcp`          : Lists all configured MCP servers and their current status.
- `/mcp on <name>`: Enables an MCP server dynamically.
- `/mcp off <name>`: Disables an MCP server.
- `/clear`        : Wipes the AI's short-term session memory.
- `/history`      : Prints the ongoing conversation history.
- `/quit`         : Exit the CLI.

## 🔌 MCP Servers (Plugins)

Model Context Protocol (MCP) servers allow Codi to instantly gain new tools without custom Python code.  
Codi natively bundles configurations for **21 powerful MCP servers** inside `mcp_servers.json`. Most are disabled by default (except for filesystem/git essentials) but can be instantly activated using `/mcp on <name>`.

### The Pre-Configured MCP Arsenal:
- **Core Engineering:** `filesystem`, `memory`, `sequential-thinking`, `git`, `python-sandbox`
- **Third-Party APIs:** `brave-search`, `github`, `gitlab`, `searxng` (free, open-source web search)
- **Databases:** `postgres`, `sqlite`, `mysql`, `redis`
- **DevOps & Infra:** `docker`, `kubernetes`
- **Monitoring & QA:** `sentry`, `prometheus`
- **Automation & Scraping:** `puppeteer`, `playwright`, `fetch`
- **Design:** `stitch` (Google UI/UX code extractor via SSE)
- **API Interfaces:** `openapi`

Easily add any open-source or custom MCP module to the JSON registry to instantly give Codi those capabilities.

## 🧠 Architecture
- `main.py`: The core REPL loop, terminal UI (`rich`), and command routing.
- `agent.py`: Tool binding, LangChain standard ReAct template parsing, and executor setup.
- `llm_factory.py`: Handles dynamic LLM switching dynamically between standard and cloud endpoints.
- `refiner.py`: The prompt pre-processor.
- `tools.py`: Native tools natively loaded into LangChain (`run_command`, `read_file`, `write_file`).
- `mcp_client.py`: Async subprocess spawner for establishing Stdio connections to standard `@modelcontextprotocol` NPX / UVX packages.

Enjoy building with Codi!
