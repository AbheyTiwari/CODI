"""Persistent exact code index used alongside Chroma's semantic search."""
from __future__ import annotations

import ast
import hashlib
import os
import re
import sqlite3
from pathlib import Path


_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "chroma_db", ".codi", "dist", "build"}
_TEXT_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".sql", ".html", ".css"}
_SYMBOL_RE = re.compile(r"\b(?:class|def|function|const|let|var|interface|type|enum)\s+([A-Za-z_]\w*)")


def _root(root: str | None = None) -> Path:
    return Path(root or os.environ.get("CODI_WORKING_DIR") or os.getcwd()).resolve()


def _database_path(root: str | None = None) -> Path:
    directory = _root(root) / ".codi"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "code_index.sqlite3"


def _connect(root: str | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path(root))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL,
            language TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS symbols (
            path TEXT NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            scope TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (path, name, kind, line),
            FOREIGN KEY (path) REFERENCES files(path) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
        """
    )
    return connection


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _language(path: Path) -> str:
    return {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript"}.get(path.suffix.lower(), path.suffix.lstrip(".") or "text")


def _python_symbols(source: str) -> list[tuple[str, str, int, int, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[tuple[str, str, int, int, str]] = []

    def walk(nodes, scope: str = "") -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = getattr(node, "end_lineno", node.lineno)
                found.append((node.name, "function", node.lineno, end, scope))
                for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                    found.append((argument.arg, "parameter", argument.lineno, argument.lineno, f"{scope}.{node.name}".strip(".")))
                if node.args.vararg:
                    found.append((node.args.vararg.arg, "parameter", node.args.vararg.lineno, node.args.vararg.lineno, f"{scope}.{node.name}".strip(".")))
                if node.args.kwarg:
                    found.append((node.args.kwarg.arg, "parameter", node.args.kwarg.lineno, node.args.kwarg.lineno, f"{scope}.{node.name}".strip(".")))
                walk(node.body, f"{scope}.{node.name}".strip("."))
            elif isinstance(node, ast.ClassDef):
                end = getattr(node, "end_lineno", node.lineno)
                found.append((node.name, "class", node.lineno, end, scope))
                walk(node.body, f"{scope}.{node.name}".strip("."))
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        found.append((target.id, "variable", target.lineno, getattr(target, "end_lineno", target.lineno), scope))
    walk(tree.body)
    return found


def _generic_symbols(source: str) -> list[tuple[str, str, int, int, str]]:
    found = []
    for line_number, line in enumerate(source.splitlines(), 1):
        for match in _SYMBOL_RE.finditer(line):
            keyword = line[match.start():match.start(1)].strip().split()[0]
            found.append((match.group(1), keyword, line_number, line_number, ""))
    return found


def index_file(path: str | os.PathLike[str], root: str | None = None) -> dict:
    base = _root(root)
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = base / absolute
    absolute = absolute.resolve()
    try:
        relative = absolute.relative_to(base).as_posix()
    except ValueError:
        return {"success": False, "error": "file is outside the working directory"}
    if not absolute.is_file() or absolute.suffix.lower() not in _TEXT_EXTENSIONS:
        return {"success": False, "error": "file is missing or unsupported", "path": relative}
    source = absolute.read_text(encoding="utf-8", errors="replace")
    digest = _sha256(source)
    language = _language(absolute)
    symbols = _python_symbols(source) if language == "python" else _generic_symbols(source)
    db = _connect(str(base))
    try:
        current = db.execute("SELECT sha256 FROM files WHERE path = ?", (relative,)).fetchone()
        if current and current["sha256"] == digest:
            return {"success": True, "path": relative, "updated": False, "symbols": len(symbols)}
        db.execute("DELETE FROM files WHERE path = ?", (relative,))
        db.execute("INSERT INTO files(path, sha256, language, updated_at) VALUES (?, ?, ?, strftime('%s','now'))", (relative, digest, language))
        db.executemany("INSERT INTO symbols(path, name, kind, line, end_line, scope) VALUES (?, ?, ?, ?, ?, ?)", [(relative, *symbol) for symbol in symbols])
        db.commit()
    finally:
        db.close()
    return {"success": True, "path": relative, "updated": True, "symbols": len(symbols)}


def index_project(root: str | None = None) -> dict:
    base = _root(root)
    indexed = 0
    for path in base.rglob("*"):
        if not path.is_file() or any(part in _SKIP_DIRS for part in path.relative_to(base).parts):
            continue
        result = index_file(path, str(base))
        indexed += int(bool(result.get("success")))
    return {"success": True, "root": str(base), "files_indexed": indexed, "database": str(_database_path(str(base)))}


def find_symbols(name: str, root: str | None = None, path: str | None = None) -> list[dict]:
    db = _connect(root)
    try:
        query = "SELECT path, name, kind, line, end_line, scope FROM symbols WHERE name = ?"
        args: list[str] = [name]
        if path:
            query += " AND path = ?"
            args.append(path.replace("\\", "/"))
        query += " ORDER BY path, line"
        return [dict(row) for row in db.execute(query, args).fetchall()]
    finally:
        db.close()


def find_references(name: str, root: str | None = None, path: str | None = None, limit: int = 200) -> list[dict]:
    base = _root(root)
    candidates = [Path(path)] if path else [item for item in base.rglob("*") if item.is_file()]
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    results: list[dict] = []
    for candidate in candidates:
        absolute = candidate if candidate.is_absolute() else base / candidate
        if not absolute.is_file() or absolute.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        try:
            relative = absolute.resolve().relative_to(base).as_posix()
            for line_number, line in enumerate(absolute.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    results.append({"path": relative, "line": line_number, "text": line})
                    if len(results) >= limit:
                        return results
        except (OSError, ValueError):
            continue
    return results
