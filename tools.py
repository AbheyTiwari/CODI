import os
import subprocess
import ast
import traceback
from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel
from indexer import get_vectorstore
from logger import log
from mcp_manager import MCPManager

class WriteFileInput(BaseModel):
    path: str
    content: str

class SearchInput(BaseModel):
    query: str

class CommandInput(BaseModel):
    command: str

DANGEROUS_PATTERNS = [
    "rm -rf", "rm -f", "mkfs", "dd if=",
    "chmod 777", "> /dev/", "format c:",
    "DROP TABLE", "DELETE FROM", ":(){:|:&};:",
    "..\\ ", "../"
]

def _run_command(command: str) -> str:
    # Sandbox Jailing Guard
    if "cd .." in command.lower() or "cd \\" in command.lower() or "cd /" in command.lower() or "pushd .." in command.lower():
        log("tool_call", {"tool": "run_command", "status": "blocked", "reason": "JailEscape"})
        return "BLOCKED: Escape attempted. Directory traversal is forbidden."

    # Block dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in command.lower():
            log("tool_call", {"tool": "run_command", "status": "blocked", "input": command})
            return f"BLOCKED: Command contains dangerous pattern '{pattern}'."

    # Log call BEFORE execution
    log("tool_call", {"tool": "run_command", "input": command})
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True,
            text=True, timeout=60,
            cwd=os.environ.get("CODI_WORKING_DIR", os.getcwd())
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        output_str = "\n".join(filter(None, [out, err])) or "(no output)"
        log("tool_result", {"tool": "run_command", "output": output_str[:200], "status": "ok"})
        return output_str
    except subprocess.TimeoutExpired:
        log("tool_result", {"tool": "run_command", "output": "TIMEOUT", "status": "error"})
        return "ERROR: Timed out after 60s"
    except Exception as e:
        log("tool_result", {"tool": "run_command", "error": str(e), "status": "error"})
        return f"ERROR: {e}"

run_command = StructuredTool.from_function(
    func=_run_command,
    name="run_command",
    description="Run a shell command and return stdout + stderr.",
    args_schema=CommandInput
)

@tool
def read_file(path: str) -> str:
    """Reads and returns the full contents of any file by path."""
    log("tool_call", {"tool": "read_file", "input": path})
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        log("tool_result", {"tool": "read_file", "path": path, "length": len(content), "status": "ok"})
        return content
    except Exception as e:
        log("tool_result", {"tool": "read_file", "path": path, "error": str(e), "status": "error"})
        return f"Error reading file: {e}"

def _write_file(path: str, content: str) -> str:
    # Log CALL first — previously result was logged before call (inverted order bug)
    log("tool_call", {"tool": "write_file", "input": path, "length": len(content)})

    if path.endswith('.py'):
        try:
            ast.parse(content)
        except SyntaxError:
            error_msg = traceback.format_exc()
            log("tool_result", {"tool": "write_file", "path": path, "status": "rejected", "reason": "SyntaxError"})
            return f"WRITE REJECTED: SyntaxError in {path}. Fix before retrying:\n{error_msg}"

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
            f.write(content)
        log("tool_result", {"tool": "write_file", "path": path, "length": len(content), "status": "ok"})
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        log("tool_result", {"tool": "write_file", "path": path, "error": str(e), "status": "error"})
        return f"ERROR: {e}"

write_file = StructuredTool.from_function(
    func=_write_file,
    name="write_file",
    description="Write content to a file at path. Creates directories automatically.",
    args_schema=WriteFileInput
)

@tool
def search_codebase(query: str) -> str:
    """Semantic search across ChromaDB — returns top 5 relevant chunks with file + chunk metadata."""
    log("tool_call", {"tool": "search_codebase", "input": query})
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return "Codebase not indexed yet. Run /index <path> first."

        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return "No matching code chunks found."

        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown file")
            results.append(f"--- Chunk {i+1} from {source} ---\n{doc.page_content}\n")
        result = "\n".join(results)
        log("tool_result", {"tool": "search_codebase", "chunks": len(docs), "status": "ok"})
        return result
    except Exception as e:
        if "not indexed" in str(e).lower() or "does not exist" in str(e).lower():
            return "Codebase not indexed yet. Run /index <path> first."
        log("tool_result", {"tool": "search_codebase", "error": str(e), "status": "error"})
        return f"Error searching codebase: {e}"

@tool
def list_files(dir_path: str) -> str:
    """Lists all files in a directory recursively (skips junk dirs)."""
    log("tool_call", {"tool": "list_files", "input": dir_path})
    skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build'}
    try:
        file_list = []
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            for file in files:
                file_list.append(os.path.join(root, file))
        log("tool_result", {"tool": "list_files", "count": len(file_list), "status": "ok"})
        return "\n".join(file_list)
    except Exception as e:
        log("tool_result", {"tool": "list_files", "error": str(e), "status": "error"})
        return f"Error listing files: {e}"


# In local mode, only load essential MCP tools.
# 60+ tool schemas consume the entire context window of a 7B model before
# the actual prompt even starts — causing silent freezes and garbage output.
# Cloud mode gets everything since large models have 128k+ context.
ESSENTIAL_MCP_LOCAL = {
    # filesystem
    "read_file", "write_file", "create_directory", "list_directory",
    # git
    "git_status", "git_diff", "git_commit", "git_log",
    # fetch
    "fetch",
    # memory
    "create_entities", "search_nodes",
}

def get_all_tools():
    from config import MODE

    base_tools = [run_command, read_file, write_file, search_codebase, list_files]

    if os.path.exists("mcp_servers.json"):
        manager = MCPManager("mcp_servers.json")
        mcp_tools = manager.load_all()
    else:
        mcp_tools = []

    if MODE == "local":
        filtered = [t for t in mcp_tools if t.name in ESSENTIAL_MCP_LOCAL]
        print(f"  [Tools] {len(base_tools)} native + {len(filtered)} MCP (local, filtered from {len(mcp_tools)}) = {len(base_tools) + len(filtered)} total")
        return base_tools + filtered
    else:
        print(f"  [Tools] {len(base_tools)} native + {len(mcp_tools)} MCP = {len(base_tools) + len(mcp_tools)} total")
        return base_tools + mcp_tools