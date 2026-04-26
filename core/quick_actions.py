from __future__ import annotations

import os
import re
from pathlib import PurePath

from logger import log
from state.temp_db import RunState
from tools.registry import ToolRegistry


KNOWN_EXTENSIONS = (".html", ".css", ".js", ".ts", ".py", ".txt", ".md", ".json")
CREATE_WORDS = ("create", "make", "write", "generate", "build", "save")
EDIT_WORDS = ("edit", "update", "change", "modify", "replace", "append", "prepend", "insert")
FILE_TOKEN_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_.\-\\/]+\.(?:html|css|js|ts|py|txt|md|json))",
    re.IGNORECASE,
)


def try_fast_file_task(user_input: str, registry: ToolRegistry, state: RunState) -> str | None:
    """Handle common local file requests without the multi-LLM agent loop."""
    text = (user_input or "").strip()
    if not text:
        return None

    # DISABLE fast path for requests with specific content requirements
    # Examples: "create portfolio with projects and skills", "create form with validation"
    # These need the full agent pipeline to generate differentiated content
    if _has_content_specifications(text):
        return None

    edit_result = _try_fast_edit(text, registry, state)
    if edit_result:
        return edit_result

    create_result = _try_fast_create(text, registry, state)
    if create_result:
        return create_result

    return None


def is_direct_file_request(text: str) -> bool:
    """True for obvious file creation/editing prompts that should skip prompt refinement."""
    lower = text.lower()
    return (
        _extract_path(text) is not None
        and (any(word in lower for word in CREATE_WORDS) or any(word in lower for word in EDIT_WORDS))
    ) or (
        any(word in lower for word in CREATE_WORDS)
        and _infer_ext(lower) is not None
        and "file" in lower
    )


def _has_content_specifications(text: str) -> bool:
    """
    True if the request specifies WHAT content should be in the file.
    Returns False for simple file creation, True for requests that need the full agent.
    
    Examples of content specifications:
      - "create portfolio with projects and skills"
      - "create form with validation"
      - "create landing page with hero section"
      - "create dashboard showing analytics"
    
    Examples that should NOT match:
      - "create index.html"
      - "create a new file called styles.css"
      - "make simple.html"
    """
    lower = text.lower()
    
    # Content specification keywords that indicate complex requirements
    content_keywords = (
        "section", "page", "form", "button", "input", "nav", "menu", "header",
        "footer", "sidebar", "card", "modal", "dialog", "popup", "project",
        "skill", "portfolio", "landing", "hero", "feature", "dashboard",
        "chart", "graph", "list", "table", "gallery", "slider", "carousel",
        "contact", "search", "filter", "sort", "validation", "error",
        "component", "layout", "design", "style", "theme", "animation",
        "interactive", "responsive", "dark mode", "light mode"
    )
    
    # If the input mentions ANY of these, it likely has content specs
    if any(keyword in lower for keyword in content_keywords):
        return True
    
    # Also check for "with" + specifics (e.g., "with 3 projects", "with validation")
    if "with " in lower and len(text.split()) > 5:
        # Likely contains specifications
        return True
    
    return False


def _try_fast_create(text: str, registry: ToolRegistry, state: RunState) -> str | None:
    lower = text.lower()
    if not any(word in lower for word in CREATE_WORDS):
        return None
    if "folder" in lower or "directory" in lower:
        return None

    explicit_path = _extract_path(text)
    ext = _extension(explicit_path) if explicit_path else _infer_ext(lower)
    if not ext:
        return None

    inferred = explicit_path is None
    path = explicit_path or _extract_named_path(text, ext) or _default_filename(ext)
    if inferred:
        path = _unique_path(path)
    elif _exists(path) and not _allows_overwrite(lower):
        output = f"File already exists, not overwritten: {path}"
        state.add_tool_result("create_file", "error", output)
        return output

    content = _template_for(ext, path)
    if content is None:
        return None

    output = _run_tool(registry, "write_file", {"path": path, "content": content}, state)
    if _is_error_output(output):
        return output

    label = _label_for(ext)
    return f"Created `{path}` with a starter {label} template."


def _try_fast_edit(text: str, registry: ToolRegistry, state: RunState) -> str | None:
    lower = text.lower()
    path = _extract_path(text)
    if not path:
        return None
    if not any(word in lower for word in EDIT_WORDS):
        return None

    quoted = [value for value in _quoted_strings(text) if not _looks_like_path(value)]

    if "replace" in lower and len(quoted) >= 2:
        output = _run_tool(
            registry,
            "edit_file",
            {"path": path, "old": quoted[0], "new": quoted[1]},
            state,
        )
        if _is_error_output(output):
            return output
        return f"Edited `{path}` by replacing the requested text."

    if "append" in lower and quoted:
        output = _run_tool(registry, "edit_file", {"path": path, "append": quoted[0]}, state)
        if _is_error_output(output):
            return output
        return f"Edited `{path}` by appending the requested text."

    if "prepend" in lower and quoted:
        output = _run_tool(registry, "edit_file", {"path": path, "prepend": quoted[0]}, state)
        if _is_error_output(output):
            return output
        return f"Edited `{path}` by prepending the requested text."

    return None


def _run_tool(registry: ToolRegistry, name: str, args: dict, state: RunState) -> str:
    handler = registry.get(name)
    if handler is None:
        output = f"Tool not found: {name}"
        state.add_tool_result(name, "error", output)
        return output

    try:
        output = str(handler(args))
    except Exception as e:
        output = f"Tool error: {e}"

    state.add_tool_result(name, "error" if _is_error_output(output) else "ok", output)
    log("quick_action_tool", {"tool": name, "status": state.tool_results[-1].status})
    return output


def _extract_path(text: str) -> str | None:
    for value in _quoted_strings(text):
        if _looks_like_path(value):
            return _clean_path(value)

    match = FILE_TOKEN_RE.search(text)
    if match:
        return _clean_path(match.group("path"))
    return None


def _extract_named_path(text: str, ext: str) -> str | None:
    match = re.search(r"\b(?:named|called|name it)\s+[`'\"]?([A-Za-z0-9_.-]+)", text, re.IGNORECASE)
    if not match:
        return None
    name = match.group(1).strip("`'\".,;: ")
    if not name:
        return None
    return name if _extension(name) else f"{name}{ext}"


def _quoted_strings(text: str) -> list[str]:
    matches = re.findall(r'"([^"]*)"|\'([^\']*)\'|`([^`]*)`', text)
    return [next(part for part in match if part) for match in matches if any(match)]


def _clean_path(path: str) -> str:
    return path.strip().strip("`'\".,;:)")


def _looks_like_path(value: str) -> bool:
    return _extension(value) in KNOWN_EXTENSIONS


def _extension(path: str | None) -> str:
    if not path:
        return ""
    return os.path.splitext(_clean_path(path))[1].lower()


def _infer_ext(lower: str) -> str | None:
    if re.search(r"\bhtml\b|\bweb\s*page\b", lower):
        return ".html"
    if re.search(r"\bcss\b|\bstylesheet\b", lower):
        return ".css"
    if re.search(r"\bjavascript\b|\bjs\b", lower):
        return ".js"
    if re.search(r"\btypescript\b|\bts\b", lower):
        return ".ts"
    if re.search(r"\bpython\b|\bpy\b", lower):
        return ".py"
    if re.search(r"\bmarkdown\b|\bmd\b", lower):
        return ".md"
    if re.search(r"\bjson\b", lower):
        return ".json"
    if re.search(r"\btext\b|\btxt\b", lower):
        return ".txt"
    return None


def _default_filename(ext: str) -> str:
    return {
        ".html": "simple.html",
        ".css": "styles.css",
        ".js": "script.js",
        ".ts": "script.ts",
        ".py": "script.py",
        ".md": "notes.md",
        ".json": "data.json",
        ".txt": "note.txt",
    }.get(ext, f"file{ext}")


def _template_for(ext: str, path: str) -> str | None:
    title = PurePath(path).stem.replace("-", " ").replace("_", " ").title() or "Simple Page"
    templates = {
        ".html": _html_template(title),
        ".css": "body {\n  margin: 0;\n  font-family: Arial, sans-serif;\n  background: #f4f6f8;\n  color: #1f2937;\n}\n\nmain {\n  max-width: 720px;\n  margin: 48px auto;\n  padding: 24px;\n}\n",
        ".js": "const message = 'Hello from Codi';\n\nfunction greet(name = 'friend') {\n  return `${message}, ${name}!`;\n}\n\nconsole.log(greet());\n",
        ".ts": "const message: string = 'Hello from Codi';\n\nfunction greet(name: string = 'friend'): string {\n  return `${message}, ${name}!`;\n}\n\nconsole.log(greet());\n",
        ".py": "def main():\n    print('Hello from Codi')\n\n\nif __name__ == '__main__':\n    main()\n",
        ".md": f"# {title}\n\nStart writing here.\n",
        ".json": "{\n  \"name\": \"starter\",\n  \"items\": []\n}\n",
        ".txt": "Start writing here.\n",
    }
    return templates.get(ext)


def _html_template(title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: Arial, sans-serif;
      background: #f4f6f8;
      color: #1f2937;
    }}

    main {{
      width: min(680px, calc(100% - 32px));
      padding: 32px;
      border: 1px solid #d8dee6;
      border-radius: 8px;
      background: #ffffff;
    }}

    button {{
      padding: 10px 14px;
      border: 0;
      border-radius: 6px;
      background: #2563eb;
      color: white;
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{title}</h1>
    <p>This is a simple starter HTML file created by Codi.</p>
    <button id="actionButton">Click me</button>
  </main>

  <script>
    document.getElementById('actionButton').addEventListener('click', () => {{
      alert('Hello from Codi!');
    }});
  </script>
</body>
</html>
"""


def _label_for(ext: str) -> str:
    return {
        ".html": "HTML",
        ".css": "CSS",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".py": "Python",
        ".md": "Markdown",
        ".json": "JSON",
        ".txt": "text",
    }.get(ext, ext.lstrip("."))


def _exists(path: str) -> bool:
    return os.path.exists(_resolve_path(path))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.environ.get("CODI_WORKING_DIR", os.getcwd()), path)


def _unique_path(path: str) -> str:
    if not _exists(path):
        return path

    root, ext = os.path.splitext(path)
    for index in range(1, 1000):
        candidate = f"{root}_{index}{ext}"
        if not _exists(candidate):
            return candidate
    return path


def _allows_overwrite(lower: str) -> bool:
    return "overwrite" in lower or "replace file" in lower


def _is_error_output(output: str) -> bool:
    return output.startswith(("ERROR", "WRITE REJECTED", "BLOCKED", "Tool not found", "Tool error"))
