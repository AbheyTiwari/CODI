# tools/local/search_tools.py

import functools
import os
import re

from context_trimmer import trim_tool_output
from logger import log

_SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", "dist", "build",
              "chroma_db", ".idea", ".mypy_cache", ".pytest_cache"}
_TEXT_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".json",
              ".md", ".txt", ".yml", ".yaml", ".toml", ".sh", ".java", ".go",
              ".rs", ".rb", ".php", ".sql", ".c", ".cpp", ".h", ".hpp"}


@functools.lru_cache(maxsize=128)
def _cached_search(query: str, k: int = 5):
    from indexer import get_vectorstore

    vs = get_vectorstore()
    if vs is None:
        return ()

    docs = vs.similarity_search(query, k=k)
    return tuple((doc.page_content, dict(doc.metadata)) for doc in docs)


def search_codebase(args: dict) -> str:
    """Semantic search across the indexed project codebase. Returns top matching chunks."""
    query = args.get("query", "")
    if not query:
        return "ERROR: no query provided"

    log("tool_call", {"tool": "search_codebase", "query": query})
    try:
        docs = _cached_search(query, k=5)
        if not docs:
            return "Codebase not indexed yet. Run /index first." if not docs else "No matching code chunks found."
        results = []
        for i, (content, metadata) in enumerate(docs):
            source = metadata.get("source", "unknown")
            chunk = trim_tool_output(content, max_tokens=200)
            results.append(f"--- Chunk {i+1} [{source}] ---\n{chunk}")
        log("tool_result", {"tool": "search_codebase", "chunks": len(docs), "status": "ok"})
        return "\n\n".join(results)
    except Exception as e:
        return f"ERROR searching codebase: {e}"


def _working_dir() -> str:
    return os.environ.get("CODI_WORKING_DIR", os.getcwd())


def grep_codebase(args: dict) -> str:
    """Search file contents for a literal string or regex across the project.
    Returns 'path:line: text' matches — use this BEFORE read_file to locate
    code instead of loading whole files. Args: pattern (required),
    path (optional dir, default project root), regex (optional bool,
    default false), max_results (optional int, default 100)."""
    a = args if isinstance(args, dict) else {}
    pattern = a.get("pattern", "")
    if not pattern:
        return "ERROR: no pattern provided"

    root = a.get("path") or _working_dir()
    if not os.path.isabs(root):
        root = os.path.join(_working_dir(), root)

    use_regex = bool(a.get("regex", False))
    try:
        max_results = int(a.get("max_results", 100))
    except (TypeError, ValueError):
        max_results = 100

    matcher = None
    if use_regex:
        try:
            matcher = re.compile(pattern)
        except re.error as e:
            return f"ERROR: invalid regex: {e}"

    log("tool_call", {"tool": "grep_codebase", "pattern": pattern, "regex": use_regex, "path": root})

    results: list[str] = []
    try:
        for r, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for fname in files:
                if os.path.splitext(fname)[1].lower() not in _TEXT_EXTS:
                    continue
                fpath = os.path.join(r, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            hit = matcher.search(line) if use_regex else (pattern in line)
                            if hit:
                                rel = os.path.relpath(fpath, root)
                                results.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                                if len(results) >= max_results:
                                    raise StopIteration
                except (OSError, UnicodeDecodeError):
                    continue
    except StopIteration:
        pass
    except Exception as e:
        return f"ERROR searching: {e}"

    if not results:
        return "No matches found."

    log("tool_result", {"tool": "grep_codebase", "matches": len(results)})
    return trim_tool_output("\n".join(results), max_tokens=800)


def glob_files(args: dict) -> str:
    """Find files by name pattern (e.g. 'src/**/*.py', '*.json') without
    reading their contents. Args: pattern (required), path (optional root,
    default project root)."""
    a = args if isinstance(args, dict) else {}
    pattern = a.get("pattern", "")
    if not pattern:
        return "ERROR: no pattern provided"

    root = a.get("path") or _working_dir()
    if not os.path.isabs(root):
        root = os.path.join(_working_dir(), root)

    log("tool_call", {"tool": "glob_files", "pattern": pattern, "path": root})

    try:
        from pathlib import Path
        root_path = Path(root)
        matches = [
            str(p.relative_to(root_path))
            for p in root_path.glob(pattern)
            if p.is_file() and not any(part in _SKIP_DIRS for part in p.parts)
        ]
    except Exception as e:
        return f"ERROR globbing: {e}"

    if not matches:
        return "No files matched."

    log("tool_result", {"tool": "glob_files", "matches": len(matches)})
    return trim_tool_output("\n".join(sorted(matches)[:300]), max_tokens=500)


def register_search_tools(registry):
    registry.register_local("search_codebase", search_codebase)
    registry.register_local("grep_codebase", grep_codebase)
    registry.register_local("glob_files", glob_files)