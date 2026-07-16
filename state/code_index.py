"""Incremental, graph-aware project index used for repository understanding.

The index deliberately uses SQLite (including FTS when available) so it is
portable, cheap to refresh, and useful even when semantic embeddings are not.
Chroma remains the semantic half of retrieval; this module supplies exact
symbols, imports, calls, ownership, and component resolution.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import sqlite3
from pathlib import Path

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "chroma_db", ".codi", "dist", "build", ".next", "coverage"}
_TEXT_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".php", ".c", ".h", ".cpp", ".hpp", ".cs", ".sql", ".html", ".css"}
_SYMBOL_RE = re.compile(r"\b(?:class|def|function|const|let|var|interface|type|enum)\s+([A-Za-z_$][\w$]*)")
_IMPORT_RE = re.compile(r"(?:import\s+(?:.+?\s+from\s+)?|require\(['\"])([^'\";\s)]+)")
_CALL_RE = re.compile(r"\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\s*\(")
_STOP_CALLS = {"if", "for", "while", "switch", "catch", "function", "def", "print", "return", "new"}
_INDEX_FORMAT = "3"


def _root(root: str | None = None) -> Path:
    return Path(root or os.environ.get("CODI_WORKING_DIR") or os.getcwd()).resolve()


def _database_path(root: str | None = None) -> Path:
    directory = _root(root) / ".codi"; directory.mkdir(parents=True, exist_ok=True)
    return directory / "code_index.sqlite3"


def _connect(root: str | None = None) -> sqlite3.Connection:
    db = sqlite3.connect(_database_path(root)); db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, language TEXT NOT NULL, updated_at INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS symbols (path TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL, line INTEGER NOT NULL, end_line INTEGER NOT NULL, scope TEXT NOT NULL DEFAULT '', PRIMARY KEY (path,name,kind,line), FOREIGN KEY(path) REFERENCES files(path) ON DELETE CASCADE);
    CREATE TABLE IF NOT EXISTS edges (source_path TEXT NOT NULL, source_symbol TEXT NOT NULL DEFAULT '', target TEXT NOT NULL, edge_type TEXT NOT NULL, line INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(source_path,source_symbol,target,edge_type,line), FOREIGN KEY(source_path) REFERENCES files(path) ON DELETE CASCADE);
    CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target, edge_type);
    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_path, edge_type);
    """)
    try:
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS symbol_search USING fts5(name, path, scope, kind)")
    except sqlite3.OperationalError:  # SQLite builds without FTS still work.
        pass
    return db


def _sha256(source: str) -> str: return hashlib.sha256(source.encode("utf-8", errors="replace")).hexdigest()
def _language(path: Path) -> str: return {".py":"python", ".js":"javascript", ".jsx":"javascript", ".ts":"typescript", ".tsx":"typescript"}.get(path.suffix.lower(), path.suffix.lstrip(".") or "text")


class _PythonFacts(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: list[tuple[str,str,int,int,str]] = []
        self.edges: list[tuple[str,str,str,int]] = []
        self.scope = ""

    def _scope(self, name: str) -> str: return f"{self.scope}.{name}".strip(".")
    def visit_Import(self, node):
        for alias in node.names: self.edges.append((self.scope, alias.name, "imports", node.lineno))
    def visit_ImportFrom(self, node):
        module = node.module or ""
        self.edges.append((self.scope, module, "imports", node.lineno))
        for alias in node.names: self.edges.append((self.scope, f"{module}.{alias.name}".strip("."), "imports", node.lineno))
    def visit_ClassDef(self, node):
        self.symbols.append((node.name,"class",node.lineno,getattr(node,"end_lineno",node.lineno),self.scope))
        for base in node.bases:
            target = ast.unparse(base) if hasattr(ast, "unparse") else "base"
            self.edges.append((self._scope(node.name), target, "inherits", node.lineno))
        old = self.scope; self.scope = self._scope(node.name); self.generic_visit(node); self.scope = old
    def visit_FunctionDef(self, node): self._function(node)
    def visit_AsyncFunctionDef(self, node): self._function(node)
    def _function(self, node):
        self.symbols.append((node.name,"function",node.lineno,getattr(node,"end_lineno",node.lineno),self.scope))
        child = self._scope(node.name)
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]: self.symbols.append((arg.arg,"parameter",arg.lineno,arg.lineno,child))
        old = self.scope; self.scope = child; self.generic_visit(node); self.scope = old
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name): self.symbols.append((target.id,"variable",target.lineno,getattr(target,"end_lineno",target.lineno),self.scope))
        self.generic_visit(node)
    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name): self.symbols.append((node.target.id,"variable",node.target.lineno,getattr(node.target,"end_lineno",node.target.lineno),self.scope))
        self.generic_visit(node)
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name): target = node.func.id
        elif isinstance(node.func, ast.Attribute): target = node.func.attr
        else: target = ""
        if target: self.edges.append((self.scope, target, "calls", node.lineno))
        self.generic_visit(node)
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load): self.edges.append((self.scope, node.id, "references", node.lineno))


def _facts(source: str, language: str):
    if language == "python":
        try:
            visitor = _PythonFacts(); visitor.visit(ast.parse(source)); return visitor.symbols, visitor.edges
        except SyntaxError: pass
    symbols = []
    for number, line in enumerate(source.splitlines(), 1):
        for match in _SYMBOL_RE.finditer(line):
            kind = line[match.start():match.start(1)].strip().split()[0]; symbols.append((match.group(1), kind, number, number, ""))
    edges = [("", m.group(1), "imports", source[:m.start()].count("\n") + 1) for m in _IMPORT_RE.finditer(source)]
    edges += [("", m.group(1).rsplit(".",1)[-1], "calls", source[:m.start()].count("\n") + 1) for m in _CALL_RE.finditer(source) if m.group(1) not in _STOP_CALLS]
    return symbols, edges


def index_file(path: str | os.PathLike[str], root: str | None = None) -> dict:
    base = _root(root); absolute = Path(path); absolute = (absolute if absolute.is_absolute() else base / absolute).resolve()
    try: relative = absolute.relative_to(base).as_posix()
    except ValueError: return {"success":False,"error":"file is outside the working directory"}
    if not absolute.is_file() or absolute.suffix.lower() not in _TEXT_EXTENSIONS: return {"success":False,"error":"file is missing or unsupported","path":relative}
    source = absolute.read_text(encoding="utf-8", errors="replace"); digest = _sha256(source); language = _language(absolute); symbols, edges = _facts(source, language)
    # A module import and a particular call can be observed through more than
    # one AST shape. Keep graph edges set-like, as intended by their key.
    symbols = list(dict.fromkeys(symbols))
    edges = list(dict.fromkeys(edges))
    db = _connect(str(base))
    try:
        current = db.execute("SELECT sha256 FROM files WHERE path=?", (relative,)).fetchone()
        if current and current["sha256"] == digest: return {"success":True,"path":relative,"updated":False,"symbols":len(symbols)}
        db.execute("DELETE FROM files WHERE path=?", (relative,)); db.execute("DELETE FROM symbol_search WHERE path=?", (relative,)) if _fts_available(db) else None
        db.execute("INSERT INTO files(path,sha256,language,updated_at) VALUES (?,?,?,strftime('%s','now'))", (relative,digest,language))
        db.executemany("INSERT INTO symbols(path,name,kind,line,end_line,scope) VALUES (?,?,?,?,?,?)", [(relative,*s) for s in symbols])
        db.executemany("INSERT INTO edges(source_path,source_symbol,target,edge_type,line) VALUES (?,?,?,?,?)", [(relative,*e) for e in edges])
        if _fts_available(db): db.executemany("INSERT INTO symbol_search(name,path,scope,kind) VALUES (?,?,?,?)", [(s[0],relative,s[4],s[1]) for s in symbols])
        db.commit()
    finally: db.close()
    return {"success":True,"path":relative,"updated":True,"symbols":len(symbols),"edges":len(edges)}


def _fts_available(db):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='symbol_search'").fetchone())


def index_project(root: str | None = None) -> dict:
    base = _root(root); seen=set(); updated=0
    # Reparse once when the extractor schema changes; normal requests remain
    # incremental and skip unchanged content.
    db = _connect(str(base))
    try:
        version = db.execute("SELECT value FROM metadata WHERE key='format'").fetchone()
        if not version or version["value"] != _INDEX_FORMAT:
            db.execute("DELETE FROM files")
            if _fts_available(db): db.execute("DELETE FROM symbol_search")
            db.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES ('format',?)", (_INDEX_FORMAT,))
            db.commit()
    finally: db.close()
    for path in base.rglob("*"):
        if not path.is_file() or any(part in _SKIP_DIRS for part in path.relative_to(base).parts): continue
        result=index_file(path,str(base)); seen.add(path.relative_to(base).as_posix()); updated += int(result.get("updated",False))
    db=_connect(str(base))
    try:
        stale=[row[0] for row in db.execute("SELECT path FROM files") if row[0] not in seen]
        db.executemany("DELETE FROM files WHERE path=?", [(p,) for p in stale]);
        if _fts_available(db): db.executemany("DELETE FROM symbol_search WHERE path=?", [(p,) for p in stale])
        db.commit()
    finally: db.close()
    return {"success":True,"root":str(base),"files_indexed":len(seen),"updated":updated,"removed":len(stale),"database":str(_database_path(str(base)))}


def find_symbols(name: str, root: str | None = None, path: str | None = None) -> list[dict]:
    db=_connect(root)
    try:
        query="SELECT path,name,kind,line,end_line,scope FROM symbols WHERE name=?"; args=[name]
        if path: query += " AND path=?"; args.append(path.replace("\\","/"))
        return [dict(x) for x in db.execute(query+" ORDER BY path,line",args)]
    finally: db.close()


def find_references(name: str, root: str | None = None, path: str | None = None, limit: int = 200) -> list[dict]:
    db=_connect(root)
    try:
        query="SELECT source_path AS path,source_symbol,target,edge_type,line FROM edges WHERE target=?"; args=[name]
        if path: query+=" AND source_path=?"; args.append(path.replace("\\","/"))
        return [dict(x) for x in db.execute(query+" ORDER BY path,line LIMIT ?", [*args,limit])]
    finally: db.close()


def graph_neighbors(target: str, root: str | None = None, direction: str = "both", limit: int = 100) -> list[dict]:
    db=_connect(root)
    try:
        clauses=[]; args=[]
        if direction in {"in","both"}: clauses.append("target=?"); args.append(target)
        if direction in {"out","both"}: clauses.append("source_path=? OR source_symbol=?"); args.extend([target,target])
        return [dict(x) for x in db.execute("SELECT source_path,source_symbol,target,edge_type,line FROM edges WHERE "+" OR ".join(f"({x})" for x in clauses)+" LIMIT ?", [*args,limit])]
    finally: db.close()


def resolve_component(capability: str, root: str | None = None, limit: int = 8) -> list[dict]:
    """Map a responsibility phrase to existing components; never invent paths."""
    tokens=[x.lower() for x in re.findall(r"[A-Za-z_][A-Za-z_0-9-]*", capability) if len(x)>2]
    db=_connect(root)
    try:
        rows=[dict(x) for x in db.execute("SELECT s.path,s.name,s.kind,s.line,s.scope,f.language FROM symbols s JOIN files f ON f.path=s.path")]
    finally: db.close()
    scored=[]
    for row in rows:
        haystack=" ".join([row["path"],row["name"],row["scope"],row["kind"]]).lower().replace("_"," ").replace("-"," ")
        score=sum(3 if token in row["name"].lower().replace("_"," ") else 1 for token in tokens if token in haystack)
        if score: scored.append({**row,"score":score})
    return sorted(scored,key=lambda r:(-r["score"],r["path"],r["line"]))[:limit]


def context_for_query(query: str, root: str | None = None, limit: int = 12) -> dict:
    """Small structured context for a task: candidates plus direct graph edges."""
    components=resolve_component(query,root,limit)
    paths=[]
    for item in components:
        if item["path"] not in paths: paths.append(item["path"])
    edges=[]
    for path in paths[:5]: edges.extend(graph_neighbors(path,root,"both",20))
    # One highest-scoring component per file is the ownership view: helpers
    # used by a pipeline do not outrank the module that declares the focused
    # responsibility itself.
    owners = []
    for path in paths:
        owner = next(item for item in components if item["path"] == path)
        owners.append({"capability": query, "path": path, "symbol": owner["name"], "kind": owner["kind"], "confidence": owner["score"]})
    return {"success":True,"query":query,"components":components,"files":paths[:5],"owners":owners,"graph":edges[:40]}
