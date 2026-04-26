# File I/O tools. All callables receive a plain dict of args and return a string.

import ast
import os
import subprocess
import time
import traceback

from context_trimmer import trim_tool_output
from logger import log


# ── Typing Effect Configuration ──────────────────────────────────────────────
TYPING_DELAY = 0.1  # seconds between characters (adjust for speed)
TYPING_ENABLED = True  # Set to False to disable typing effect


def _write_with_typing_effect(file_obj, content: str, delay: float = TYPING_DELAY):
    """Write content to file character by character with a typing effect.
    
    Adaptive speed: larger files type faster so total time stays reasonable.
    """
    content_len = len(content)
    if content_len == 0:
        return
    
    # Adaptive delay: aim for max ~3 seconds total typing time
    # Small files (< 100 chars): use full delay for visibility
    # Large files (> 500 chars): speed up to stay under 3 seconds
    target_max_time = 3.0  # seconds
    adaptive_delay = min(delay, target_max_time / content_len)
    
    for char in content:
        file_obj.write(char)
        file_obj.flush()  # Ensure character is written immediately
        if adaptive_delay > 0:
            time.sleep(adaptive_delay)


def _open_in_vscode(path: str):
    """Open the file in VS Code so the user can see the typing effect live."""
    try:
        # Use 'code' command to open file in VS Code
        subprocess.Popen(
            ["code", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
    except Exception:
        # Silently fail if VS Code CLI is not available
        pass


def _working_dir() -> str:
    return os.environ.get("CODI_WORKING_DIR", os.getcwd())


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_working_dir(), path)


def _path_arg(args: dict) -> str:
    raw_path = args.get("path") or args.get("filename") or args.get("file") or ""
    return _abs(str(raw_path)) if raw_path else ""


def read_file(args: dict) -> str:
    """Read a file. Relative paths resolve from the project directory."""
    path = _path_arg(args)
    if not path:
        return "ERROR reading file: missing path"

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
    """Write text to a file. Args: path, content or content_lines list; validates .py syntax."""
    path = _path_arg(args)
    if not path:
        return "ERROR writing file: missing path"

    content = _coerce_content(args)
    log("tool_call", {"tool": "write_file", "path": path, "length": len(content)})

    syntax_error = _python_syntax_error(path, content)
    if syntax_error:
        return syntax_error

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Open file in VS Code so user can see the typing effect
        _open_in_vscode(path)
        
        with open(path, "w", encoding="utf-8") as f:
            if TYPING_ENABLED and len(content) > 0:
                _write_with_typing_effect(f, content)
            else:
                f.write(content)
        log("tool_result", {"tool": "write_file", "path": path, "status": "ok"})
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


def edit_file(args: dict) -> str:
    """Edit an existing file. Args: path plus old/new, replacements, append, prepend, insert_after, or insert_before."""
    path = _path_arg(args)
    if not path:
        return "ERROR editing file: missing path"

    log("tool_call", {"tool": "edit_file", "path": path})

    if not os.path.exists(path):
        if args.get("create_if_missing"):
            original = ""
        else:
            return f"ERROR editing {path}: file does not exist"
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()
        except Exception as e:
            return f"ERROR reading {path}: {e}"

    try:
        content, changes = _apply_edit_operations(original, args)
    except ValueError as e:
        return f"ERROR editing {path}: {e}"

    if changes == 0:
        return f"ERROR editing {path}: no edit operation was provided"

    syntax_error = _python_syntax_error(path, content)
    if syntax_error:
        return syntax_error

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Open file in VS Code before editing so user can see the typing effect
        _open_in_vscode(path)
        
        with open(path, "w", encoding="utf-8") as f:
            if TYPING_ENABLED and len(content) > 0:
                _write_with_typing_effect(f, content)
            else:
                f.write(content)
        log("tool_result", {
            "tool": "edit_file",
            "path": path,
            "status": "ok",
            "changes": changes,
        })
        return f"Edited {path} ({changes} change{'s' if changes != 1 else ''})"
    except Exception as e:
        return f"ERROR editing {path}: {e}"


def list_files(args: dict) -> str:
    """List files recursively in a directory."""
    dir_path = args.get("dir", args.get("path", "."))
    if dir_path in (".", "", None):
        dir_path = _working_dir()
    else:
        dir_path = _abs(str(dir_path))

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
    path = _path_arg(args)
    if not path:
        return "ERROR creating directory: missing path"

    try:
        os.makedirs(path, exist_ok=True)
        log("tool_result", {"tool": "create_directory", "path": path, "status": "ok"})
        return f"Created directory: {path}"
    except Exception as e:
        return f"ERROR creating directory {path}: {e}"


def _coerce_content(args: dict) -> str:
    """Accept the content shapes that small local models commonly produce."""
    lines = args.get("content_lines")
    if isinstance(lines, list):
        return "\n".join(str(line) for line in lines)

    for key in ("content", "text", "body"):
        if key in args:
            value = args.get(key)
            if isinstance(value, list):
                return "\n".join(str(line) for line in value)
            if value is None:
                return ""
            return str(value)

    return ""


def _python_syntax_error(path: str, content: str) -> str:
    if not path.lower().endswith(".py"):
        return ""
    try:
        ast.parse(content)
        return ""
    except SyntaxError:
        err = traceback.format_exc()
        return f"WRITE REJECTED - SyntaxError in {path}:\n{err}"


def _replace_text(content: str, old: str, new: str, count: int | None = 1) -> tuple[str, int]:
    if old == "":
        raise ValueError("old text for replacement cannot be empty")
    occurrences = content.count(old)
    if occurrences == 0:
        raise ValueError(f"text not found: {old[:80]}")
    if count is None or count <= 0:
        return content.replace(old, new), occurrences
    return content.replace(old, new, count), min(count, occurrences)


def _coerce_count(value) -> int | None:
    if value is None:
        return 1
    if isinstance(value, str) and value.lower() in ("all", "every"):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _edit_payload(args: dict) -> str:
    for key in ("content", "text", "insert", "value", "new"):
        if key in args:
            value = args.get(key)
            if isinstance(value, list):
                return "\n".join(str(line) for line in value)
            return "" if value is None else str(value)
    return ""


def _apply_edit_operations(content: str, args: dict) -> tuple[str, int]:
    changes = 0
    replacements = args.get("replacements")

    if isinstance(replacements, list):
        for item in replacements:
            if not isinstance(item, dict):
                raise ValueError("each replacement must be an object with old and new text")
            old = str(item.get("old") or item.get("search") or item.get("from") or "")
            new = str(item.get("new") or item.get("replace") or item.get("to") or "")
            count = _coerce_count(item.get("count", args.get("count")))
            content, changed = _replace_text(content, old, new, count)
            changes += changed

    old_text = args.get("old") or args.get("search") or args.get("from")
    if old_text is not None:
        new_text = args.get("new")
        if new_text is None:
            new_text = args.get("replace", args.get("to", ""))
        content, changed = _replace_text(
            content,
            str(old_text),
            str(new_text),
            _coerce_count(args.get("count")),
        )
        changes += changed

    if "append" in args or "append_content" in args:
        addition = args.get("append", args.get("append_content"))
        addition = "" if addition is None else str(addition)
        separator = "" if not content or content.endswith("\n") or addition.startswith("\n") else "\n"
        content = f"{content}{separator}{addition}"
        changes += 1

    if "prepend" in args or "prepend_content" in args:
        addition = args.get("prepend", args.get("prepend_content"))
        addition = "" if addition is None else str(addition)
        separator = "" if not content or addition.endswith("\n") else "\n"
        content = f"{addition}{separator}{content}"
        changes += 1

    if "insert_after" in args:
        marker = str(args.get("insert_after") or "")
        payload = _edit_payload(args)
        if not marker:
            raise ValueError("insert_after marker cannot be empty")
        pos = content.find(marker)
        if pos == -1:
            raise ValueError(f"insert_after marker not found: {marker[:80]}")
        pos += len(marker)
        content = content[:pos] + payload + content[pos:]
        changes += 1

    if "insert_before" in args:
        marker = str(args.get("insert_before") or "")
        payload = _edit_payload(args)
        if not marker:
            raise ValueError("insert_before marker cannot be empty")
        pos = content.find(marker)
        if pos == -1:
            raise ValueError(f"insert_before marker not found: {marker[:80]}")
        content = content[:pos] + payload + content[pos:]
        changes += 1

    return content, changes


def register_file_tools(registry):
    registry.register_local("create_file",       write_file)
    registry.register_local("read_file",         read_file)
    registry.register_local("write_file",        write_file)
    registry.register_local("edit_file",         edit_file)
    registry.register_local("list_files",        list_files)
    registry.register_local("create_directory",  create_directory)
