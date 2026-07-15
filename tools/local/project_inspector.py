"""Low-cost structural inspection used before source is read."""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

_LANGUAGES = {".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript", ".json": "json", ".md": "markdown", ".html": "html", ".css": "css"}
_IGNORED = {".git", "node_modules", "__pycache__", ".venv", "venv", "chroma_db", "dist", "build"}

# ── Non-Python structural extraction ────────────────────────────────────────
# inspect_file() previously returned ONLY a line count for every non-.py
# file — no tags, no classes, no functions, nothing. For an HTML/CSS/JS-only
# project (the common case for "build me a website" tasks) this meant the
# dependency graph and every downstream "PROJECT KNOWLEDGE" summary shown to
# the planner LLM was permanently empty ({"index.html": [], "script.js": [],
# "styles.css": []}), regardless of how much actual structure the files had.
# The planner then had no way to know a file already had a hero section, a
# nav, or an existing function, and would invent duplicate/contradictory
# plans. These extractors are intentionally cheap (regex, not a real
# parser) — good enough to give the planner real signal, not full fidelity.

_HTML_TAG_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9-]*)\b[^>]*>", re.DOTALL)
_HTML_ID_RE = re.compile(r'\bid=["\']([^"\']+)["\']')
_HTML_CLASS_RE = re.compile(r'\bclass=["\']([^"\']+)["\']')
_HTML_SCRIPT_SRC_RE = re.compile(r'<script[^>]*\bsrc=["\']([^"\']+)["\']', re.IGNORECASE)
_HTML_LINK_HREF_RE = re.compile(r'<link[^>]*\bhref=["\']([^"\']+)["\']', re.IGNORECASE)
_HTML_INLINE_SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL)
_HTML_LANDMARK_TAGS = {
    "header", "nav", "main", "footer", "section", "article", "aside", "form",
}

_CSS_SELECTOR_RE = re.compile(r"([^{}]+)\{", re.DOTALL)
_CSS_VAR_RE = re.compile(r"--([a-zA-Z0-9_-]+)\s*:")
_CSS_MEDIA_RE = re.compile(r"@media[^{]+\{")
_CSS_KEYFRAMES_RE = re.compile(r"@keyframes\s+([a-zA-Z0-9_-]+)")

_JS_FUNCTION_RE = re.compile(
    r"\bfunction\s+([a-zA-Z_$][\w$]*)\s*\(|"
    r"\b(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>|"
    r"\b(?:const|let|var)\s+([a-zA-Z_$][\w$]*)\s*=\s*(?:async\s+)?function\b"
)
_JS_CLASS_RE = re.compile(r"\bclass\s+([a-zA-Z_$][\w$]*)")
_JS_EVENT_LISTENER_RE = re.compile(r"\.addEventListener\(\s*['\"]([a-zA-Z]+)['\"]")
_JS_IMPORT_RE = re.compile(r"^\s*import\s+.*?from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)
_JS_REQUIRE_RE = re.compile(r"require\(\s*['\"]([^'\"]+)['\"]\s*\)")


def _inspect_html(source: str) -> dict:
    tags = _HTML_TAG_RE.findall(source)
    tag_counts: dict[str, int] = {}
    for tag in tags:
        lowered = tag.lower()
        tag_counts[lowered] = tag_counts.get(lowered, 0) + 1

    ids = _HTML_ID_RE.findall(source)
    classes: list[str] = []
    for match in _HTML_CLASS_RE.findall(source):
        classes.extend(match.split())

    landmarks = sorted({tag for tag in tag_counts if tag in _HTML_LANDMARK_TAGS})
    linked_scripts = _HTML_SCRIPT_SRC_RE.findall(source)
    linked_styles = [href for href in _HTML_LINK_HREF_RE.findall(source) if href.lower().endswith(".css")]
    has_inline_script = bool(_HTML_INLINE_SCRIPT_RE.search(source))

    return {
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda item: -item[1])[:20]),
        "landmarks": landmarks,
        "ids": sorted(set(ids))[:40],
        "classes": sorted(set(classes))[:40],
        "linked_scripts": linked_scripts,
        "linked_stylesheets": linked_styles,
        "has_inline_script": has_inline_script,
        "summary": (
            f"HTML with landmarks={landmarks or 'none'}; "
            f"{len(set(ids))} unique id(s), {len(set(classes))} unique class(es); "
            f"links to scripts={linked_scripts or 'none'}, stylesheets={linked_styles or 'none'}."
        ),
    }


_CSS_KEYFRAME_STEP_RE = re.compile(r"^(from|to|\d+(\.\d+)?%)$", re.IGNORECASE)


def _inspect_css(source: str) -> dict:
    selectors: list[str] = []
    for raw in _CSS_SELECTOR_RE.findall(source):
        cleaned = raw.strip().replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned or cleaned.startswith("@"):
            continue
        # Split comma-separated selector groups into individual selectors.
        for piece in cleaned.split(","):
            piece = piece.strip()
            # Skip from/to/NN% steps — these are @keyframes body markers,
            # not real selectors, but the naive brace-matching regex above
            # can't tell an @keyframes block's contents apart from a
            # top-level rule since it doesn't track nesting.
            if piece and not _CSS_KEYFRAME_STEP_RE.match(piece):
                selectors.append(piece)

    variables = sorted(set(_CSS_VAR_RE.findall(source)))
    media_queries = len(_CSS_MEDIA_RE.findall(source))
    keyframes = sorted(set(_CSS_KEYFRAMES_RE.findall(source)))

    return {
        "selectors": selectors[:60],
        "selector_count": len(selectors),
        "css_variables": variables[:40],
        "media_query_count": media_queries,
        "keyframe_animations": keyframes,
        "summary": (
            f"CSS with {len(selectors)} selector(s), {len(variables)} custom "
            f"propert(y/ies), {media_queries} media quer(y/ies), "
            f"keyframe animations={keyframes or 'none'}."
        ),
    }


def _inspect_js(source: str) -> dict:
    functions = sorted({name for group in _JS_FUNCTION_RE.findall(source) for name in group if name})
    classes = sorted(set(_JS_CLASS_RE.findall(source)))
    events = sorted(set(_JS_EVENT_LISTENER_RE.findall(source)))
    imports = sorted(set(_JS_IMPORT_RE.findall(source)) | set(_JS_REQUIRE_RE.findall(source)))

    return {
        "functions": functions[:40],
        "classes": classes[:20],
        "event_listeners": events,
        "imports": imports,
        "summary": (
            f"JavaScript with {len(functions)} function(s), {len(classes)} class(es); "
            f"listens for events={events or 'none'}; imports={imports or 'none'}."
        ),
    }


_NON_PYTHON_INSPECTORS = {
    "html": _inspect_html,
    "css": _inspect_css,
    "javascript": _inspect_js,
    "typescript": _inspect_js,
}


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
        result.update({"line_count": len(source.splitlines())})
        extractor = _NON_PYTHON_INSPECTORS.get(language)
        if extractor is not None:
            try:
                structural = extractor(source)
            except Exception:
                structural = None
            if structural:
                result.update(structural)
                # Surface functions/classes/imports through the SAME keys the
                # Python branch and downstream KnowledgeBase.record_inspection()
                # already expect, so a non-Python file's structure actually
                # flows into the dependency graph / symbol summary instead of
                # only living in extra HTML/CSS/JS-specific fields nothing
                # else reads.
                if "functions" in structural:
                    result["functions"] = [{"name": name, "line": 0, "args": []} for name in structural["functions"]]
                if "classes" in structural:
                    result["classes"] = [{"name": name, "line": 0, "bases": [], "docstring": None, "methods": []} for name in structural["classes"]]
                if "imports" in structural:
                    result["imports"] = structural["imports"]
                if "linked_scripts" in structural:
                    result["imports"] = sorted(set(result.get("imports", [])) | set(structural["linked_scripts"]))
                if "linked_stylesheets" in structural:
                    result["exports"] = structural["linked_stylesheets"]
                if "ids" in structural or "classes" in structural and language == "html":
                    result["globals"] = structural.get("ids", []) + structural.get("classes", [])
            else:
                result["summary"] = f"Lightweight {language} inspection; use read_file only if source is required."
        else:
            result["summary"] = f"Lightweight {language} inspection; use read_file only if source is required."
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