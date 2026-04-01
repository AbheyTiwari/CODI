import os
import sys
import subprocess
import ast
import traceback
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel
from indexer import get_vectorstore
from logger import log
from context_trimmer import trim_tool_output, estimate_tokens

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

class WriteFileInput(BaseModel):
    path: str
    content: str

class CommandInput(BaseModel):
    command: str

DANGEROUS_PATTERNS = [
    "rm -rf", "rm -f", "mkfs", "dd if=",
    "chmod 777", "> /dev/", "format c:",
    "DROP TABLE", "DELETE FROM", ":(){:|:&};:",
]

ESSENTIAL_MCP_LOCAL = {
    "read_file", "write_file", "create_directory", "list_directory",
    "git_status", "git_diff", "git_commit", "git_log",
    "fetch", "create_entities", "search_nodes",
}

def _working_dir() -> str:
    """Always return the directory the user launched codi from."""
    return os.environ.get("CODI_WORKING_DIR", os.getcwd())

def _run_command(command: str) -> str:
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in command.lower():
            log("tool_call", {"tool": "run_command", "status": "blocked", "input": command})
            return f"BLOCKED: dangerous pattern '{pattern}'."

    log("tool_call", {"tool": "run_command", "input": command})
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",       # force UTF-8 — fixes Windows cp1252 crash
            errors="replace",       # replace undecodable bytes instead of crashing
            timeout=60,
            cwd=_working_dir(),
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        raw = "\n".join(filter(None, [out, err])) or "(no output)"
        output_str = trim_tool_output(raw, max_tokens=600)
        log("tool_result", {"tool": "run_command", "output": output_str[:200], "status": "ok"})
        return output_str
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after 60s"
    except Exception as e:
        return f"ERROR: {e}"

run_command = StructuredTool.from_function(
    func=_run_command,
    name="run_command",
    description=(
        "Run a shell/PowerShell command and return output. "
        "Runs in the user's project directory. "
        "Use for git, file ops, installs, running scripts."
    ),
    args_schema=CommandInput,
)

@tool
def read_file(path: str) -> str:
    """Read a file. Relative paths resolve from the user's project directory. If file is large, content will be trimmed — re-read with line ranges if needed."""
    if not os.path.isabs(path):
        path = os.path.join(_working_dir(), path)
    log("tool_call", {"tool": "read_file", "input": path})
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        total_tokens = estimate_tokens(content)
        trimmed = trim_tool_output(content, max_tokens=800)
        was_trimmed = len(trimmed) < len(content)
        log("tool_result", {"tool": "read_file", "path": path, "length": len(content), "trimmed": was_trimmed, "status": "ok"})
        if was_trimmed:
            trimmed += f"\n\n⚠ FILE TRUNCATED: Showing ~800 tokens of ~{total_tokens} total. To see specific sections, read the file in smaller parts."
        return trimmed
    except Exception as e:
        return f"Error reading file: {e}"

def _write_file(path: str, content: str) -> str:
    if not os.path.isabs(path):
        path = os.path.join(_working_dir(), path)
    log("tool_call", {"tool": "write_file", "input": path, "length": len(content)})
    if path.endswith('.py'):
        try:
            ast.parse(content)
        except SyntaxError:
            err = traceback.format_exc()
            return f"WRITE REJECTED: SyntaxError in {path}:\n{err}"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        log("tool_result", {"tool": "write_file", "path": path, "status": "ok"})
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"

write_file = StructuredTool.from_function(
    func=_write_file,
    name="write_file",
    description="Write content to a file. Relative paths resolve from the user's project directory.",
    args_schema=WriteFileInput,
)

@tool
def list_files(dir_path: str = ".") -> str:
    """
    List files in a directory recursively.
    Defaults to the user's current project directory if no path given.
    Pass '.' or leave empty to list the project root.
    """
    # Resolve relative paths from working dir, not the repo root
    if dir_path in (".", "", None):
        dir_path = _working_dir()
    elif not os.path.isabs(dir_path):
        dir_path = os.path.join(_working_dir(), dir_path)

    log("tool_call", {"tool": "list_files", "input": dir_path})
    skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build', 'chroma_db'}
    try:
        file_list = []
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            for file in files:
                file_list.append(os.path.join(root, file))
        result = "\n".join(file_list)
        trimmed = trim_tool_output(result, max_tokens=400)
        log("tool_result", {"tool": "list_files", "count": len(file_list), "status": "ok"})
        return trimmed
    except Exception as e:
        return f"Error listing files: {e}"

@tool
def search_codebase(query: str) -> str:
    """Semantic search across the indexed project. Returns top 5 relevant code chunks."""
    log("tool_call", {"tool": "search_codebase", "input": query})
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return "Codebase not indexed yet. Run /index first."
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return "No matching code chunks found."
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            chunk  = trim_tool_output(doc.page_content, max_tokens=200)
            results.append(f"--- Chunk {i+1} from {source} ---\n{chunk}\n")
        log("tool_result", {"tool": "search_codebase", "chunks": len(docs), "status": "ok"})
        return "\n".join(results)
    except Exception as e:
        return f"Error searching codebase: {e}"


def get_all_tools():
    from config import MODE
    from mcp_manager import MCPManager

    base_tools = [run_command, read_file, write_file, list_files, search_codebase]

    # Always resolve mcp_servers.json from repo root, not cwd
    mcp_config = os.path.join(_REPO_ROOT, "mcp_servers.json")
    if not os.path.exists(mcp_config):
        print(f"  [Tools] {len(base_tools)} native + 0 MCP = {len(base_tools)} total")
        return base_tools

    manager   = MCPManager(mcp_config)
    mcp_tools = manager.load_all()

    if MODE in ("local", "air"):
        filtered = [t for t in mcp_tools if t.name in ESSENTIAL_MCP_LOCAL]
        print(f"  [Tools] {len(base_tools)} native + {len(filtered)} MCP "
              f"({MODE}, filtered from {len(mcp_tools)}) = {len(base_tools)+len(filtered)} total")
        return base_tools + filtered

    print(f"  [Tools] {len(base_tools)} native + {len(mcp_tools)} MCP = {len(base_tools)+len(mcp_tools)} total")
    return base_tools + mcp_tools