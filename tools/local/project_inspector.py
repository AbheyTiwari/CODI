"""Low-cost structural inspection used before source is read."""
from __future__ import annotations

import ast
import os
from pathlib import Path

_LANGUAGES = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".json": "json", ".md": "markdown", ".html": "html", ".css": "css"}
_IGNORED = {".git", "node_modules", "__pycache__", ".venv", "venv", "chroma_db", "dist", "build"}

def _function(node):
    return {"name": node.name, "line": node.lineno, "args": [arg.arg for arg in node.args.args], "returns": ast.unparse(node.returns) if node.returns else None, "decorators": [ast.unparse(item) for item in node.decorator_list], "docstring": ast.get_docstring(node), "is_async": isinstance(node, ast.AsyncFunctionDef)}

def inspect_file(args):
    raw_path = args.get("path") if isinstance(args, dict) else args
    if not raw_path: return {"success": False, "error": "missing path"}
    path = Path(raw_path).expanduser()
    if not path.is_absolute(): path = Path(os.environ.get("CODI_WORKING_DIR", os.getcwd())) / path
    if not path.is_file(): return {"success": False, "file": str(path), "error": "file not found"}
    source = path.read_text(encoding="utf-8", errors="replace")
    language = _LANGUAGES.get(path.suffix.lower(), path.suffix.lstrip("."))
    result = {"success": True, "file": str(path), "language": language, "classes": [], "functions": [], "imports": [], "exports": [], "globals": [], "constants": [], "entrypoint": "__main__" in source}
    if path.suffix.lower() != ".py":
        result.update({"line_count": len(source.splitlines()), "summary": f"Lightweight {language} inspection; use read_file only if source is required."})
        return result
    try: tree = ast.parse(source)
    except SyntaxError as exc: return {"success": False, "file": str(path), "error": f"Python syntax error: {exc}"}
    for node in tree.body:
        if isinstance(node, ast.Import): result["imports"].extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom): result["imports"].append(node.module or "")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name): (result["constants"] if target.id.isupper() else result["globals"]).append(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): result["functions"].append(_function(node))
        elif isinstance(node, ast.ClassDef):
            result["classes"].append({"name": node.name, "line": node.lineno, "bases": [ast.unparse(base) for base in node.bases], "docstring": ast.get_docstring(node), "methods": [_function(child) for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]})
    result["exports"] = [item["name"] for item in result["functions"]] + [item["name"] for item in result["classes"]]
    result["summary"] = f"Python module with {len(result['functions'])} functions and {len(result['classes'])} classes."
    return result

def inspect_project(args=None):
    root = Path((args or {}).get("path") or os.environ.get("CODI_WORKING_DIR", os.getcwd()))
    files = [path for path in root.rglob("*") if path.is_file() and not any(part in _IGNORED for part in path.parts)]
    names = {path.name.lower() for path in files}
    languages = sorted({_LANGUAGES[path.suffix.lower()] for path in files if path.suffix.lower() in _LANGUAGES})
    frameworks = (["python"] if {"pyproject.toml", "requirements.txt", "setup.cfg"} & names else []) + (["node"] if "package.json" in names else [])
    relative = lambda path: str(path.relative_to(root))
    return {"success": True, "root": str(root), "languages": languages, "frameworks": frameworks, "entrypoints": [relative(path) for path in files if path.name in {"main.py", "app.py", "index.js", "index.ts", "manage.py"}], "manifests": [relative(path) for path in files if path.name in {"pyproject.toml", "requirements.txt", "package.json", "setup.cfg", "pytest.ini"}], "tests": [relative(path) for path in files if "test" in path.name.lower()][:100], "readme": "README.md" if "readme.md" in names else None, "files": [relative(path) for path in files], "file_count": len(files)}
