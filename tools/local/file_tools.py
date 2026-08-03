# File I/O tools. All callables receive a plain dict of args and return a string.

import ast
import json
import os
import re
import subprocess
import time
import traceback
from tools.local.project_inspector import inspect_file as _inspect_file, inspect_project
from context_trimmer import trim_tool_output
from logger import log


# ── Typing Effect Configuration ──────────────────────────────────────────────
TYPING_DELAY = 0.1  # seconds between characters (adjust for speed)
TYPING_ENABLED = False  # Disabled: char-by-char writes block the agent loop for seconds per file

def inspect_file(args) -> str:
    """
    Inspect a source file and return its structure instead of the raw contents.
    Uses Python AST for .py files and lightweight parsing for other file types.
    """

    # Keep the tool boundary string-based, but delegate all parsing to the
    # canonical structured inspector.
    return json.dumps(_inspect_file(args), ensure_ascii=False)


def _write_with_typing_effect(file_obj, content: str, delay: float = TYPING_DELAY):
    """Write content to file character by character with a typing effect.
    
    Adaptive speed: larger files type faster so total time stays reasonable.
    """
    content_len = len(content)
    if content_len == 0:
        return
    
    # Adaptive delay: aim for max ~3 seconds total typing time
    target_max_time = 3.0  # seconds
    adaptive_delay = min(delay, target_max_time / content_len)
    
    for char in content:
        file_obj.write(char)
        file_obj.flush()
        if adaptive_delay > 0:
            time.sleep(adaptive_delay)


def _open_in_vscode(path: str):
    """Open the file in VS Code so the user can see the typing effect live."""
    try:
        subprocess.Popen(
            ["code", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )
    except Exception:
        pass


def _working_dir() -> str:
    return os.environ.get("CODI_WORKING_DIR", os.getcwd())


def _normcase_path(path: str) -> str:
    """Fully normalize a path for cross-platform comparison: resolve symlinks,
    collapse '..'/'.'/redundant separators, then lowercase-normalize case on
    case-insensitive filesystems (Windows/macOS-default). This is the single
    source of truth both sides of a containment check must go through —
    comparing a realpath'd candidate against a NON-normcased working_dir
    (the previous bug) meant a single drive-letter or filename-case mismatch
    made os.path.commonpath() report two DIFFERENT roots, which silently
    rejected every read/write as "path escapes project directory" any time
    CODI_WORKING_DIR was set (or typed via `cd`) with different casing than
    what realpath() canonicalizes it to."""
    return os.path.normcase(os.path.normpath(os.path.realpath(path)))


def _abs(path: str) -> str:
    """
    Resolve `path` against CODI_WORKING_DIR and verify it does not escape
    the project directory.

    FIX: previously this normalized ONLY candidate_real via realpath() before
    calling os.path.commonpath([working_dir, candidate_real]) — working_dir
    itself was realpath'd but neither side was case-normalized. On Windows,
    this means:
      - CODI_WORKING_DIR = "C:\\Users\\abhey\\Project" (as typed/set by cli.py)
      - candidate_real via realpath() may canonicalize the drive letter or
        any segment to a different case (e.g. junctions, subst drives, or
        just OS-level case folding quirks)
      - os.path.commonpath() compares path components as plain strings, so
        "C:\\Users" and "c:\\users" are treated as UNRELATED roots
      - commonpath() then returns something that isn't working_dir, the
        check fails, and _abs() returns "ERROR: path escapes project
        directory" for a perfectly valid in-project file
      - This return value is a plain string, not a tool-shaped error, so
        every caller (read_file, write_file, edit_file, list_files, ...)
        just treats it as "the path" and the actual file op then 404s or
        no-ops against a bogus literal path containing the word ERROR —
        the agent loop sees a generic tool failure with no indication the
        real cause was a working-directory case mismatch.

    The fix: normalize BOTH sides identically via _normcase_path() (realpath
    + normpath + normcase) before comparing, and use a prefix check instead
    of relying solely on commonpath()'s own (non-case-normalizing) string
    comparison. commonpath() across different drives on Windows also raises
    ValueError, which was being caught and collapsed into the same generic
    "escapes project directory" message — that masked a genuinely different
    failure (wrong drive entirely) behind the same text as a same-drive case
    mismatch. Both cases now still return the escape error (that part of the
    behavior is correct and intentional — this is a real security boundary),
    but the underlying normalization bug that made VALID paths fail no
    longer exists.
    """
    working_dir = _working_dir()
    candidate = path if os.path.isabs(path) else os.path.join(working_dir, path)

    normalized_working_dir = _normcase_path(working_dir)
    normalized_candidate = _normcase_path(candidate)

    # Prefix check on fully-normalized paths — avoids commonpath()'s
    # cross-drive ValueError entirely and doesn't depend on case matching
    # between the two inputs, only on them agreeing AFTER normalization.
    if normalized_candidate != normalized_working_dir and not normalized_candidate.startswith(
        normalized_working_dir + os.sep
    ):
        log("file_tools_path_escape", {
            "requested_path": path,
            "working_dir": working_dir,
            "normalized_working_dir": normalized_working_dir,
            "normalized_candidate": normalized_candidate,
        })
        return "ERROR: path escapes project directory"

    # Return the realpath'd-but-not-case-mangled candidate for actual file
    # I/O. We deliberately do NOT return normalized_candidate (which was
    # lowercased on Windows/macOS via normcase) — that would break
    # case-sensitive filesystems and mangle the path shown back to the user.
    # normalize just structurally (realpath + normpath), keep original case.
    return os.path.normpath(os.path.realpath(candidate))


def _path_arg(args) -> str:
    # Accept both str and dict — fast path in main.py passes a string,
    # dispatcher and agent pass a dict.
    if isinstance(args, str):
        return _abs(args) if args else ""
    if not isinstance(args, dict):
        return ""
    raw_path = args.get("path") or args.get("filename") or args.get("file") or ""
    return _abs(str(raw_path)) if raw_path else ""


def _refresh_exact_index(path: str) -> None:
    """Keep the SQLite symbol index synchronized without failing file I/O."""
    try:
        from state.code_index import index_file
        index_file(path)
    except Exception as exc:
        log("code_index_refresh_error", {"path": path, "error": str(exc)[:160]})


def read_file(args) -> str:
    """Read a file. Relative paths resolve from the project directory."""
    path = _path_arg(args)
    if not path:
        return "ERROR reading file: missing path"
    if path.startswith("ERROR"):
        return f"ERROR reading file: {path}"

    log("tool_call", {"tool": "read_file", "path": path})
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        result = trim_tool_output(content, max_tokens=800)
        log("tool_result", {"tool": "read_file", "length": len(content), "status": "ok"})
        return result
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def read_agent_history(_args=None) -> str:
    """Read CODI's own persistent command history, outside the user project."""
    history_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".agent_history")
    try:
        with open(history_path, "r", encoding="utf-8", errors="replace") as f:
            return trim_tool_output(f.read(), max_tokens=800)
    except Exception as e:
        return f"ERROR reading CODI agent history: {e}"


def read_file_numbered(args) -> str:
    """Read a file WITH 1-indexed line numbers prefixed (e.g. '  42\\t<code>').
    Use this before any surgical line-range edit (replace_lines / delete_lines /
    insert_at_line) so the exact line numbers are known ahead of time instead
    of being guessed. Args: path, optionally start_line/end_line to view a
    slice of a large file instead of the whole thing."""
    path = _path_arg(args)
    if not path:
        return "ERROR reading file: missing path"
    if path.startswith("ERROR"):
        return f"ERROR reading file: {path}"

    start_line = None
    end_line = None
    if isinstance(args, dict):
        start_line = args.get("start_line")
        end_line = args.get("end_line")

    log("tool_call", {"tool": "read_file_numbered", "path": path})
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return f"ERROR reading {path}: {e}"

    total = len(lines)
    if total == 0:
        return f"[{path} — 0 lines, file is empty]"

    try:
        s = max(1, int(start_line)) if start_line else 1
    except (TypeError, ValueError):
        s = 1
    try:
        e = min(total, int(end_line)) if end_line else total
    except (TypeError, ValueError):
        e = total

    if s > e:
        s, e = 1, total

    numbered = [f"{i:>5}\t{lines[i - 1].rstrip(chr(10)).rstrip(chr(13))}" for i in range(s, e + 1)]
    body = "\n".join(numbered)
    result = trim_tool_output(body, max_tokens=1500)
    log("tool_result", {"tool": "read_file_numbered", "lines": total, "status": "ok"})
    return f"[{path} — {total} lines total, showing {s}-{e}]\n{result}"


def write_file(args: dict) -> str:
    """Write text to a file. Args: path, content or content_lines list; warns on .py syntax errors."""
    path = _path_arg(args)
    if not path:
        return "ERROR writing file: missing path"
    if path.startswith("ERROR"):
        return f"ERROR writing file: {path}"

    content = _coerce_content(args)
    log("tool_call", {"tool": "write_file", "path": path, "length": len(content)})

    syntax_warning = _python_syntax_check(path, content)
    if not syntax_warning:
        syntax_warning = _java_structural_check(path, content)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        if TYPING_ENABLED:
            _open_in_vscode(path)
        
        with open(path, "w", encoding="utf-8") as f:
            if TYPING_ENABLED and len(content) > 0:
                _write_with_typing_effect(f, content)
            else:
                f.write(content)
        _refresh_exact_index(path)
        log("tool_result", {"tool": "write_file", "path": path, "status": "ok"})
        result = {
            "success":       True,
            "tool":          "write_file",
            "file_modified": path,
            "bytes_written": len(content),
            "syntax_ok":     not bool(syntax_warning),
        }
        if syntax_warning:
            result["syntax_warning"] = syntax_warning
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "tool": "write_file", "error": str(e), "path": path})


def create_file(args: dict) -> str:
    """Create a new file without overwriting an existing one."""
    path = _path_arg(args)
    if not path:
        return "ERROR creating file: missing path"
    if path.startswith("ERROR"):
        return f"ERROR creating file: {path}"
    if os.path.exists(path):
        return (
            f"ERROR creating file: {path} already exists. "
            "Use read_file followed by edit_file for a surgical change; never recreate it."
        )
    result = write_file(args)
    try:
        payload = json.loads(result)
        if isinstance(payload, dict):
            payload["tool"] = "create_file"
            return json.dumps(payload)
    except (TypeError, ValueError):
        pass
    return result


def edit_file(args: dict) -> str:
    """Edit an existing file. Args: path plus old/new, replacements, append,
    prepend, insert_after, insert_before — OR surgical line-range operations
    replace_lines / delete_lines / insert_at_line (use read_file_numbered
    first to get accurate line numbers)."""
    path = _path_arg(args)
    if not path:
        return "ERROR editing file: missing path"
    if path.startswith("ERROR"):
        return f"ERROR editing file: {path}"

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

    syntax_warning = _python_syntax_check(path, content)
    if not syntax_warning:
        syntax_warning = _java_structural_check(path, content)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        if TYPING_ENABLED:
            _open_in_vscode(path)
        
        with open(path, "w", encoding="utf-8") as f:
            if TYPING_ENABLED and len(content) > 0:
                _write_with_typing_effect(f, content)
            else:
                f.write(content)
        _refresh_exact_index(path)
        log("tool_result", {
            "tool": "edit_file",
            "path": path,
            "status": "ok",
            "changes": changes,
        })
        result = {
            "success":       True,
            "tool":          "edit_file",
            "file_modified": path,
            "changes":       changes,
            "syntax_ok":     not bool(syntax_warning),
        }
        if syntax_warning:
            result["syntax_warning"] = syntax_warning
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "tool": "edit_file", "error": str(e), "path": path})


def list_files(args) -> str:
    """List files recursively in a directory."""
    if isinstance(args, str):
        args = {"path": args}
    dir_path = args.get("dir", args.get("path", "."))
    if dir_path in (".", "", None):
        dir_path = _working_dir()
    else:
        dir_path = _abs(str(dir_path))
        if dir_path.startswith("ERROR"):
            return f"ERROR listing: {dir_path}"

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
    if path.startswith("ERROR"):
        return f"ERROR creating directory: {path}"

    try:
        os.makedirs(path, exist_ok=True)
        log("tool_result", {"tool": "create_directory", "path": path, "status": "ok"})
        return f"Created directory: {path}"
    except Exception as e:
        return f"ERROR creating directory {path}: {e}"


def _coerce_content(args: dict) -> str:
    """Accept the content shapes that small local models commonly produce."""
    if not isinstance(args, dict):
        return ""

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


def _python_syntax_check(path: str, content: str) -> str:
    """Return a warning string if .py content has syntax errors.
    The file is still written — the warning helps the agent self-correct."""
    if not path.lower().endswith(".py"):
        return ""
    try:
        ast.parse(content)
        return ""
    except SyntaxError as e:
        return f"WARNING: SyntaxError in {path} (line {e.lineno}): {e.msg} — file was written, please fix."


def _java_structural_check(path: str, content: str) -> str:
    """Cheap Java brace-balance sanity check used before Maven is available."""
    if not path.lower().endswith(".java"):
        return ""
    opens = content.count("{")
    closes = content.count("}")
    if opens == closes:
        return ""
    return f"WARNING: Unbalanced braces in {path} ({opens} opening, {closes} closing) — file was written, please fix."


def _normalize_whitespace(text: str) -> str:
    """Normalize line endings and strip trailing whitespace per line."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    return "\n".join(line.rstrip() for line in lines)


def _line_number_at(content: str, char_index: int) -> int:
    """1-indexed line number of a character offset within content."""
    return content.count("\n", 0, char_index) + 1


def _occurrence_contexts(content: str, needle: str, max_occurrences: int = 6) -> list[dict]:
    """
    Return line numbers + a short surrounding snippet for every occurrence of
    `needle` in `content`.
    """
    occurrences = []
    start = 0
    while True:
        pos = content.find(needle, start)
        if pos == -1:
            break
        line_no = _line_number_at(content, pos)
        ctx_start = max(0, pos - 40)
        ctx_end = min(len(content), pos + len(needle) + 40)
        snippet = content[ctx_start:ctx_end].replace("\n", "\\n")
        occurrences.append({"line": line_no, "context": snippet})
        start = pos + max(len(needle), 1)
        if len(occurrences) >= max_occurrences:
            break
    return occurrences


def _replace_text(content: str, old: str, new: str, count: int | None = 1) -> tuple[str, int]:
    """
    Replace old with new inside content.

    Tries exact match first. If that fails, attempts three cheap normalizations
    before giving up.
    """
    if old == "":
        raise ValueError("old text for replacement cannot be empty")

    def _do_replace(c: str, o: str, n: str) -> tuple[str, int]:
        occurrences = c.count(o)
        if occurrences == 0:
            return c, 0
        if count is None or count <= 0:
            return c.replace(o, n), occurrences
        return c.replace(o, n, count), min(count, occurrences)

    def _ambiguous_error(source: str, needle: str, occurrences: int) -> ValueError:
        contexts = _occurrence_contexts(source, needle)
        lines = "; ".join(f"line {c['line']}: ...{c['context']}..." for c in contexts)
        return ValueError(
            f"text match is ambiguous ({occurrences} occurrences). "
            "Use a longer unique old snippet or explicitly set count. "
            f"Occurrences found at: {lines}"
        )

    exact_occurrences = content.count(old)
    if exact_occurrences > 1 and (count is None or count == 1):
        raise _ambiguous_error(content, old, exact_occurrences)
    result, found = _do_replace(content, old, new)
    if found:
        return result, found

    norm_content = _normalize_whitespace(content)
    norm_old     = _normalize_whitespace(old)
    norm_new     = _normalize_whitespace(new)

    norm_occurrences = norm_content.count(norm_old)
    if norm_occurrences > 1 and (count is None or count == 1):
        raise _ambiguous_error(norm_content, norm_old, norm_occurrences)
    result, found = _do_replace(norm_content, norm_old, norm_new)
    if found:
        log("edit_fuzzy_match", {"reason": "trailing_whitespace", "old": old[:60]})
        return result, found

    import re as _re
    def _collapse(t: str) -> str:
        return _re.sub(r"[ \t]+", " ", t)

    coll_content = _collapse(norm_content)
    coll_old     = _collapse(norm_old)
    coll_new     = _collapse(norm_new)

    coll_occurrences = coll_content.count(coll_old)
    if coll_occurrences > 1 and (count is None or count == 1):
        raise _ambiguous_error(coll_content, coll_old, coll_occurrences)
    result, found = _do_replace(coll_content, coll_old, coll_new)
    if found:
        log("edit_fuzzy_match", {"reason": "indentation_collapse", "old": old[:60]})
        return result, found

    raise ValueError(f"text not found (tried exact + whitespace normalization): {old[:80]}")


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@"
)


def _parse_unified_diff(patch_text: str) -> list[dict]:
    """
    Parse a unified diff into a list of hunks.
    """
    if not patch_text or not patch_text.strip():
        raise ValueError("patch is empty")

    lines = patch_text.splitlines()
    hunks: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") or line.startswith("+++ "):
            i += 1
            continue
        match = _HUNK_HEADER_RE.match(line)
        if not match:
            i += 1
            continue
        old_start = int(match.group("old_start"))
        old_len = int(match.group("old_len") or 1)
        new_start = int(match.group("new_start"))
        new_len = int(match.group("new_len") or 1)
        i += 1
        body: list[tuple[str, str]] = []
        while i < len(lines) and not _HUNK_HEADER_RE.match(lines[i]) and not lines[i].startswith(("--- ", "+++ ")):
            raw = lines[i]
            if raw.startswith(("+", "-", " ")):
                body.append((raw[0], raw[1:]))
            elif raw == "":
                body.append((" ", ""))
            else:
                body.append((" ", raw))
            i += 1
        hunks.append({
            "old_start": old_start, "old_len": old_len,
            "new_start": new_start, "new_len": new_len,
            "lines": body,
        })

    if not hunks:
        raise ValueError("no valid @@ hunk headers found in patch")
    return hunks


def _apply_hunks(original: str, hunks: list[dict]) -> str:
    """
    Apply parsed unified-diff hunks to `original`, sequentially, verifying
    each hunk's context/removal lines match the file at the claimed position
    before mutating anything.
    """
    result_lines = original.splitlines(keepends=True)
    offset = 0

    for index, hunk in enumerate(sorted(hunks, key=lambda h: h["old_start"])):
        start_idx = hunk["old_start"] - 1 + offset
        if start_idx < 0 or start_idx > len(result_lines):
            raise ValueError(
                f"hunk {index + 1} @@ -{hunk['old_start']} out of bounds "
                f"(file currently has {len(result_lines)} lines)"
            )

        cursor = start_idx
        new_segment: list[str] = []
        for kind, text in hunk["lines"]:
            line_with_nl = text if text.endswith("\n") else text + "\n"
            if kind == " ":
                if cursor >= len(result_lines) or result_lines[cursor].rstrip("\n") != text.rstrip("\n"):
                    actual = result_lines[cursor].rstrip("\n") if cursor < len(result_lines) else "<EOF>"
                    raise ValueError(
                        f"hunk {index + 1} context mismatch at line {cursor + 1}: "
                        f"expected {text!r}, found {actual!r}"
                    )
                new_segment.append(result_lines[cursor])
                cursor += 1
            elif kind == "-":
                if cursor >= len(result_lines) or result_lines[cursor].rstrip("\n") != text.rstrip("\n"):
                    actual = result_lines[cursor].rstrip("\n") if cursor < len(result_lines) else "<EOF>"
                    raise ValueError(
                        f"hunk {index + 1} removal mismatch at line {cursor + 1}: "
                        f"expected to remove {text!r}, found {actual!r}"
                    )
                cursor += 1
            elif kind == "+":
                new_segment.append(line_with_nl)

        result_lines[start_idx:cursor] = new_segment
        offset += len(new_segment) - (cursor - start_idx)

    return "".join(result_lines)


def apply_patch(args: dict) -> str:
    """Apply a unified diff (one or more @@ hunks) to an existing file.
    Args: path, patch (unified diff text)."""
    path = _path_arg(args)
    if not path:
        return "ERROR applying patch: missing path"
    if path.startswith("ERROR"):
        return f"ERROR applying patch: {path}"
    if not os.path.exists(path):
        return f"ERROR applying patch: {path} does not exist. Use create_file for new files."

    patch_text = ""
    if isinstance(args, dict):
        patch_text = args.get("patch") or args.get("diff") or ""

    log("tool_call", {"tool": "apply_patch", "path": path, "patch_len": len(patch_text)})

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
    except Exception as e:
        return f"ERROR reading {path}: {e}"

    try:
        hunks = _parse_unified_diff(patch_text)
        new_content = _apply_hunks(original, hunks)
    except ValueError as e:
        return f"ERROR applying patch to {path}: {e}"

    syntax_warning = _python_syntax_check(path, new_content)
    if not syntax_warning:
        syntax_warning = _java_structural_check(path, new_content)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        _refresh_exact_index(path)
        log("tool_result", {"tool": "apply_patch", "path": path, "status": "ok", "hunks": len(hunks)})
        result = {
            "success": True,
            "tool": "apply_patch",
            "file_modified": path,
            "hunks_applied": len(hunks),
            "syntax_ok": not bool(syntax_warning),
        }
        if syntax_warning:
            result["syntax_warning"] = syntax_warning
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"success": False, "tool": "apply_patch", "error": str(e), "path": path})


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
    if not isinstance(args, dict):
        return ""

    for key in ("content", "text", "insert", "value", "new"):
        if key in args:
            value = args.get(key)
            if isinstance(value, list):
                return "\n".join(str(line) for line in value)
            return "" if value is None else str(value)
    return ""


def _coerce_occurrence(value, default: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _find_occurrence(text: str, marker: str, occurrence: int = 1) -> int:
    start = 0
    for index in range(occurrence):
        pos = text.find(marker, start)
        if pos == -1:
            raise ValueError(f"marker not found at occurrence {index + 1}: {marker[:80]}")
        if index == occurrence - 1:
            return pos
        start = pos + len(marker)
    raise ValueError(f"marker not found: {marker[:80]}")


def _apply_edit_operations(content: str, args: dict) -> tuple[str, int]:
    if not isinstance(args, dict):
        return content, 0

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
        occurrence = _coerce_occurrence(args.get("occurrence", 1))
        pos = _find_occurrence(content, marker, occurrence)
        pos += len(marker)
        content = content[:pos] + payload + content[pos:]
        changes += 1

    if "insert_before" in args:
        marker = str(args.get("insert_before") or "")
        payload = _edit_payload(args)
        if not marker:
            raise ValueError("insert_before marker cannot be empty")
        occurrence = _coerce_occurrence(args.get("occurrence", 1))
        pos = _find_occurrence(content, marker, occurrence)
        content = content[:pos] + payload + content[pos:]
        changes += 1

    if "replace_lines" in args:
        spec = args.get("replace_lines")
        if not isinstance(spec, dict) or "start" not in spec or "end" not in spec:
            raise ValueError("replace_lines requires {'start': int, 'end': int, 'content': str}")
        lines = content.splitlines(keepends=True)
        try:
            start = int(spec["start"])
            end = int(spec["end"])
        except (TypeError, ValueError):
            raise ValueError("replace_lines 'start'/'end' must be integers")
        if start < 1 or end > len(lines) or start > end:
            raise ValueError(
                f"replace_lines range {start}-{end} out of bounds (file has {len(lines)} lines)"
            )
        new_text = str(spec.get("content", ""))
        if new_text and not new_text.endswith("\n"):
            new_text += "\n"
        content = "".join(lines[:start - 1]) + new_text + "".join(lines[end:])
        changes += 1

    if "delete_lines" in args:
        spec = args.get("delete_lines")
        if not isinstance(spec, dict) or "start" not in spec or "end" not in spec:
            raise ValueError("delete_lines requires {'start': int, 'end': int}")
        lines = content.splitlines(keepends=True)
        try:
            start = int(spec["start"])
            end = int(spec["end"])
        except (TypeError, ValueError):
            raise ValueError("delete_lines 'start'/'end' must be integers")
        if start < 1 or end > len(lines) or start > end:
            raise ValueError(
                f"delete_lines range {start}-{end} out of bounds (file has {len(lines)} lines)"
            )
        content = "".join(lines[:start - 1]) + "".join(lines[end:])
        changes += 1

    if "insert_at_line" in args:
        spec = args.get("insert_at_line")
        if not isinstance(spec, dict) or "line" not in spec:
            raise ValueError("insert_at_line requires {'line': int, 'content': str}")
        lines = content.splitlines(keepends=True)
        try:
            line_no = int(spec["line"])
        except (TypeError, ValueError):
            raise ValueError("insert_at_line 'line' must be an integer")
        if line_no < 0 or line_no > len(lines) + 1:
            raise ValueError(
                f"insert_at_line {line_no} out of bounds (file has {len(lines)} lines)"
            )
        new_text = str(spec.get("content", ""))
        if new_text and not new_text.endswith("\n"):
            new_text += "\n"
        idx = max(0, line_no - 1)
        content = "".join(lines[:idx]) + new_text + "".join(lines[idx:])
        changes += 1

    return content, changes


def register_file_tools(registry):
    registry.register_local("create_file",        create_file)
    registry.register_local("read_file",           read_file)
    registry.register_local("read_agent_history",  read_agent_history)
    registry.register_local("read_file_numbered",  read_file_numbered)
    registry.register_local("write_file",          write_file)
    registry.register_local("edit_file",           edit_file)
    registry.register_local("apply_patch",         apply_patch)
    registry.register_local("list_files",          list_files)
    registry.register_local("create_directory",    create_directory)
    registry.register_local("inspect_file",         inspect_file)
    registry.register_local("inspect_project",      lambda args: json.dumps(inspect_project(args), ensure_ascii=False))