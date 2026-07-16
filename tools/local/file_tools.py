# tools/local/file_tools.py
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

# ── Placeholder-path detection ────────────────────────────────────────────────
# FIX (bug: silent write to literal placeholder path): the coder LLM
# occasionally emits a generic example path instead of a real one (e.g.
# "/path/to/file.txt") when the task didn't force it to name a concrete
# file. This mirrors the already-documented "relative/path" /
# "PATH_TO_INSPECT" placeholder-copying bug in core/prompts.py's
# need_context instruction — except prior to this fix, nothing caught the
# same failure mode for write_file/create_file/edit_file. Combined with the
# _abs() bug below (which used to return an "ERROR: ..." STRING as if it
# were a valid path), this previously caused CODI to silently create a real
# file on disk literally named "ERROR: path escapes project directory" and
# report success. Reject known placeholder shapes before they ever reach a
# filesystem call.
_PLACEHOLDER_PATH_PATTERNS = (
    re.compile(r"^/?path/to/", re.IGNORECASE),
    re.compile(r"^path_to_", re.IGNORECASE),
    re.compile(r"^relative/path", re.IGNORECASE),
    re.compile(r"^<.*>$"),
    re.compile(r"^\[.*\]$"),
    re.compile(r"^(tool_name|file_path|filename_here|example\.txt)$", re.IGNORECASE),
)


def _is_placeholder_path(raw_path: str) -> bool:
    candidate = (raw_path or "").strip().strip("'\"`")
    if not candidate:
        return False
    return any(pattern.search(candidate) for pattern in _PLACEHOLDER_PATH_PATTERNS)


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


class _PathEscapeError(ValueError):
    """Raised by _abs() when a resolved path falls outside the project dir."""


def _abs(path: str) -> str:
    """
    Resolve `path` against the project working directory and enforce that
    it stays inside it.

    FIX (bug): this previously RETURNED the string "ERROR: path escapes
    project directory" as if it were a valid resolved path. Every caller
    (_path_arg -> write_file/create_file/edit_file/read_file/...) had no
    check for that sentinel, so a path-escape attempt silently proceeded
    to open()/write() a real file on disk literally named
    "ERROR: path escapes project directory" in the project root, and
    reported success:true. Raising here instead forces every caller to
    explicitly handle the failure — see _path_arg() below, which is now
    the single place that turns this into a proper "" (empty path) result
    that write_file/create_file/edit_file already know means "refuse and
    report ERROR", rather than silently taking a wrong action based on
    guessing that this string looked worth writing to.
    """
    working_dir = os.path.realpath(_working_dir())
    candidate = path if os.path.isabs(path) else os.path.join(working_dir, path)
    candidate_real = os.path.realpath(candidate)
    try:
        if os.path.commonpath([working_dir, candidate_real]) != working_dir:
            raise _PathEscapeError(f"path escapes project directory: {path!r}")
    except ValueError:
        # os.path.commonpath raises ValueError when paths are on different
        # drives (Windows) — that is unambiguously also outside the project.
        raise _PathEscapeError(f"path escapes project directory: {path!r}")
    return candidate_real


def _path_arg(args) -> str:
    """
    Resolve the path argument from a tool call. Returns "" (empty string)
    on ANY failure — missing path, placeholder path, or path-escape
    attempt — so every downstream tool (write_file/create_file/edit_file/
    read_file/...) hits their existing `if not path: return "ERROR ..."`
    guard instead of silently operating on a bogus string.
    """
    # Accept both str and dict — fast path in main.py passes a string,
    # dispatcher and agent pass a dict.
    if isinstance(args, str):
        raw_path = args
    elif isinstance(args, dict):
        raw_path = args.get("path") or args.get("filename") or args.get("file") or ""
    else:
        return ""

    if not raw_path:
        return ""

    raw_path = str(raw_path)

    if _is_placeholder_path(raw_path):
        log("path_placeholder_rejected", {"raw_path": raw_path[:200]})
        return ""

    try:
        return _abs(raw_path)
    except _PathEscapeError as exc:
        log("path_escape_rejected", {"raw_path": raw_path[:200], "error": str(exc)})
        return ""


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
        return "ERROR reading file: missing, placeholder, or out-of-project path"

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
        return "ERROR reading file: missing, placeholder, or out-of-project path"

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
        return "ERROR writing file: missing, placeholder, or out-of-project path"

    content = _coerce_content(args)
    log("tool_call", {"tool": "write_file", "path": path, "length": len(content)})

    syntax_warning = _python_syntax_check(path, content)
    if not syntax_warning:
        syntax_warning = _java_structural_check(path, content)

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        if TYPING_ENABLED:
            # Open file in VS Code so user can see the typing effect
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
        return "ERROR creating file: missing, placeholder, or out-of-project path"
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
        return "ERROR editing file: missing, placeholder, or out-of-project path"

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
            # Open file in VS Code before editing so user can see the typing effect
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
        resolved = _path_arg({"path": str(dir_path)})
        if not resolved:
            return f"ERROR listing {dir_path}: missing, placeholder, or out-of-project path"
        dir_path = resolved

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
        return "ERROR creating directory: missing, placeholder, or out-of-project path"

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
    `needle` in `content`. Used to build an actionable disambiguation hint
    when a replacement is rejected as ambiguous — without this, the repair
    prompt only knows "there are 2 occurrences" and has no way to tell which
    extra characters would make its next `old` guess unique.
    """
    occurrences = []
    start = 0
    while True:
        pos = content.find(needle, start)
        if pos == -1:
            break
        line_no = _line_number_at(content, pos)
        # Grab a little context before/after so the model can see what
        # differs between occurrences (e.g. one is in <head>, one is in a
        # <script> block, one is inside a comment, etc.)
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
    before giving up — catches the most common local-model mistakes:
      1. Trailing whitespace differences  (model strips trailing spaces)
      2. Line ending differences          (CRLF vs LF)
      3. Indentation collapse             (model uses spaces instead of tabs)

    Never silently corrupts the file — if all attempts fail, raises ValueError
    so the agent knows to retry with the correct old string.

    On an ambiguous match, the ValueError message includes the line number
    and surrounding context of every occurrence found, so a repair prompt
    has enough information to pick a snippet that's actually unique instead
    of blindly resubmitting the same (still-ambiguous) text.
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

    # ── Attempt 1: exact match ────────────────────────────────────────────────
    exact_occurrences = content.count(old)
    if exact_occurrences > 1 and (count is None or count == 1):
        raise _ambiguous_error(content, old, exact_occurrences)
    result, found = _do_replace(content, old, new)
    if found:
        return result, found

    # ── Attempt 2: normalize trailing whitespace on both sides ────────────────
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

    # ── Attempt 3: collapse runs of spaces/tabs to single space ──────────────
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

    # ── All attempts failed ───────────────────────────────────────────────────
    raise ValueError(f"text not found (tried exact + whitespace normalization): {old[:80]}")


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@"
)


def _parse_unified_diff(patch_text: str) -> list[dict]:
    """
    Parse a unified diff into a list of hunks:
      {"old_start": int, "old_len": int, "new_start": int, "new_len": int,
       "lines": [(" "|"+"|"-", text), ...]}
    Tolerant of a leading '--- a/...' / '+++ b/...' file-header pair (ignored —
    the target path always comes from the tool's own "path" arg, never parsed
    out of the diff, so a model-supplied header can't redirect the write).
    Raises ValueError with a specific, actionable message on malformed input.
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
                # Tolerate a missing leading space on unchanged lines — small
                # models frequently drop it. Treat as context.
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
    before mutating anything. Applied in ascending order of old_start with a
    cumulative line-offset so each hunk's declared position (which refers to
    the ORIGINAL file) maps correctly onto the progressively-edited buffer.
    Raises ValueError naming the hunk and expected-vs-actual context on any
    mismatch, so the coder LLM gets a concrete, actionable repair signal
    (mirroring _replace_text's ambiguous-match diagnostics above). Never
    partially applies on failure — the caller only writes on full success.
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
    Args: path, patch (unified diff text). Prefer this over write_file/edit_file
    when a change touches several scattered locations in the same file —
    each hunk is verified against actual file content before anything is
    written, and the whole patch is rejected atomically if any hunk fails
    to match (no partial writes)."""
    path = _path_arg(args)
    if not path:
        return "ERROR applying patch: missing, placeholder, or out-of-project path"
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

    # ── Line-range operations ──────────────────────────────────────────────
    # Surgical edits by 1-indexed line number instead of exact text matching.
    # Pair these with read_file_numbered so the caller knows real line numbers
    # before editing — this is what lets CODI touch any portion of any file
    # type without ever having to regenerate/rewrite the whole thing.

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