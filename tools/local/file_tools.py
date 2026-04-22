# tools/local/file_tools.py
# ─────────────────────────────────────────────────────────────────────────────
# File I/O tools. All callables receive a plain dict of args.
# All return a plain string. No LangChain, no decorators.
# ─────────────────────────────────────────────────────────────────────────────

import ast
import os
import traceback

from context_trimmer import trim_tool_output
from logger import log


def _working_dir() -> str:
    return os.environ.get("CODI_WORKING_DIR", os.getcwd())


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_working_dir(), path)


# ── Tool functions ────────────────────────────────────────────────────────────

def read_file(args: dict) -> str:
    """Read a file. Relative paths resolve from the project directory."""
    path = _abs(args.get("path", ""))
    log("tool_call", {"tool": "read_file", "path": path})
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        result = trim_tool_output(content, max_tokens=800)
        log("tool_result", {"tool": "read_file", "length": len(content), "status": "ok"})
        return result
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def write_file(args: dict) -> str:
    """Write content to a file. Validates Python syntax before writing .py files."""
    path    = _abs(args.get("path", ""))
    content = args.get("content", "")
    log("tool_call", {"tool": "write_file", "path": path, "length": len(content)})

    if path.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError:
            err = traceback.format_exc()
            return f"WRITE REJECTED — SyntaxError in {path}:\n{err}"

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        log("tool_result", {"tool": "write_file", "path": path, "status": "ok"})
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


def list_files(args: dict) -> str:
    """List files recursively in a directory."""
    dir_path = args.get("dir", args.get("path", "."))
    if dir_path in (".", "", None):
        dir_path = _working_dir()
    else:
        dir_path = _abs(dir_path)

    log("tool_call", {"tool": "list_files", "path": dir_path})
    skip = {".git", "node_modules", "__pycache__", "venv", "dist", "build", "chroma_db"}
    try:
        files = []
        for root, dirs, fnames in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
            for f in fnames:
                files.append(os.path.join(root, f))
        result = trim_tool_output("\n".join(files), max_tokens=400)
        log("tool_result", {"tool": "list_files", "count": len(files), "status": "ok"})
        return result
    except Exception as e:
        return f"ERROR listing {dir_path}: {e}"


def create_directory(args: dict) -> str:
    """Create a directory (and any missing parents)."""
    path = _abs(args.get("path", ""))
    try:
        os.makedirs(path, exist_ok=True)
        log("tool_result", {"tool": "create_directory", "path": path, "status": "ok"})
        return f"Created directory: {path}"
    except Exception as e:
        return f"ERROR creating directory {path}: {e}"


# ── Registrar ─────────────────────────────────────────────────────────────────

def register_file_tools(registry):
    registry.register_local("read_file",        read_file)
    registry.register_local("write_file",       write_file)
    registry.register_local("list_files",       list_files)
    registry.register_local("create_directory", create_directory)
