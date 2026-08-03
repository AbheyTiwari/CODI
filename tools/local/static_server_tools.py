# tools/local/static_server_tools.py
# ─────────────────────────────────────────────────────────────────────────────
# Provides a real, working http:// URL for local static files (index.html,
# styles.css, script.js, etc.) so browser_navigate / playwright_navigate can
# actually verify something instead of guessing a URL that was never
# resolvable in the first place.
#
# WHY THIS EXISTS
# Static local HTML has no server, so browser_navigate against any bare
# filename ("index.html") gets treated as https://index.html/ by the
# underlying navigation tool, and a guessed dev-server URL like
# http://localhost:3000/ fails with ERR_CONNECTION_REFUSED because nothing
# is listening there. file:// URLs are also unreliable for anything using
# fetch()/modules/relative-path XHR (blocked by browser CORS rules for the
# file: scheme). The only approach that reliably works for arbitrary static
# projects is a real HTTP server rooted at the project directory.
#
# DESIGN
# - One background http.server process per working directory, reused across
#   calls within the same CODI session (keyed by working dir) rather than
#   spawning a new one on every step.
# - Picks a free port automatically; never assumes a fixed port like 3000/
#   8080, which is exactly the kind of guess that caused the original bug.
# - Health-checked before being reported as ready — callers get back a URL
#   that is CONFIRMED to be answering requests, not just "a process started".
# - Tracked in a small in-process registry so core/executor.py's browser
#   navigation guard can verify a URL a step wants to navigate to actually
#   corresponds to a server CODI itself started, instead of trusting the
#   coder LLM's URL string at face value.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

from logger import log

# working_dir (normalized) -> {"process": Popen, "port": int, "base_url": str}
_SERVERS: dict[str, dict] = {}
_LOCK = threading.Lock()


class _AssetCollector(HTMLParser):
    """Collect local resource URLs referenced by a static HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []
        self.anchors: set[str] = set()
        self.anchor_links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            self.anchors.add(element_id)
        if tag == "a" and values.get("href", "").startswith("#"):
            self.anchor_links.append(values["href"])
        attribute = "href" if tag == "link" else "src"
        value = values.get(attribute)
        if value:
            self.assets.append(value)


def _command_available(command: str) -> bool:
    """Return whether an optional local validation command is installed."""
    return shutil.which(command) is not None


def _normalize_dir(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _working_dir() -> str:
    return os.environ.get("CODI_WORKING_DIR", os.getcwd())


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _is_alive(process: subprocess.Popen) -> bool:
    return process.poll() is None


def _health_check(base_url: str, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url, timeout=1.0) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.15)
    return False


def _start_server_locked(working_dir: str) -> dict:
    """Caller must hold _LOCK. Starts a fresh http.server for working_dir."""
    port = _pick_free_port()
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=working_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        log("static_server_start_error", {"working_dir": working_dir, "error": str(e)})
        raise

    base_url = f"http://127.0.0.1:{port}/"
    entry = {"process": process, "port": port, "base_url": base_url}
    log("static_server_started", {"working_dir": working_dir, "port": port})
    return entry


def ensure_static_server(args: dict | None = None) -> str:
    """
    Start (or reuse) a static file server rooted at CODI_WORKING_DIR and
    return its base URL as JSON. This is the ONLY correct way to get a
    navigable URL for local static files — never guess a port, never treat
    a bare filename as a URL, never use file:// for anything that uses
    fetch()/modules/relative XHR.

    Args: none required. Optional "path" to serve a specific file relative
    to the working dir — if given, the returned url points directly at
    that file (still served over the same running server).
    """
    working_dir = _working_dir()
    key = _normalize_dir(working_dir)

    with _LOCK:
        entry = _SERVERS.get(key)
        if entry is None or not _is_alive(entry["process"]):
            try:
                entry = _start_server_locked(working_dir)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "tool": "ensure_static_server",
                    "error": f"Could not start static server: {e}",
                })
            _SERVERS[key] = entry

    base_url = entry["base_url"]

    if not _health_check(base_url):
        log("static_server_health_check_failed", {"working_dir": working_dir, "base_url": base_url})
        return json.dumps({
            "success": False,
            "tool": "ensure_static_server",
            "error": f"Static server started on {base_url} but did not respond to a health check.",
        })

    target_path = ""
    if isinstance(args, dict):
        target_path = str(args.get("path", "") or "").strip().lstrip("/\\")

    url = base_url + target_path if target_path else base_url

    log("static_server_ready", {"working_dir": working_dir, "url": url})
    return json.dumps({
        "success": True,
        "tool": "ensure_static_server",
        "base_url": base_url,
        "url": url,
        "port": entry["port"],
        "note": "Use this exact URL with browser_navigate. Do not guess a different port or use file://.",
    })


def verify_static_site(args: dict | None = None) -> str:
    """Verify a local HTML page and its local assets through CODI's HTTP server.

    This is intentionally a deterministic first gate, not a visual claim: it
    proves that the page is reachable over HTTP and that the CSS/JS/image URLs
    it declares resolve to real project files. Browser-level assertions can be
    layered on top when a Playwright MCP server is configured.
    """
    working_dir = os.path.abspath(_working_dir())
    requested = "index.html"
    if isinstance(args, dict) and args.get("path"):
        requested = str(args["path"])
    requested = requested.lstrip("/\\")
    absolute = os.path.abspath(os.path.join(working_dir, requested))
    try:
        if os.path.commonpath([working_dir, absolute]) != working_dir:
            return json.dumps({"success": False, "tool": "verify_static_site", "error": "path escapes project directory"})
    except ValueError:
        return json.dumps({"success": False, "tool": "verify_static_site", "error": "path is on a different drive"})
    if not absolute.lower().endswith((".html", ".htm")):
        return json.dumps({"success": False, "tool": "verify_static_site", "error": "path must be an HTML file"})
    if not os.path.isfile(absolute):
        return json.dumps({"success": False, "tool": "verify_static_site", "error": f"page does not exist: {requested}"})

    try:
        source = open(absolute, encoding="utf-8", errors="replace").read()
        parser = _AssetCollector()
        parser.feed(source)
    except (OSError, ValueError) as exc:
        return json.dumps({"success": False, "tool": "verify_static_site", "error": str(exc)})

    missing: list[str] = []
    for asset in parser.assets:
        parsed = urlsplit(asset)
        if parsed.scheme or parsed.netloc or asset.startswith(("#", "data:")):
            continue
        asset_path = unquote(parsed.path).lstrip("/\\")
        if not asset_path:
            continue
        candidate = os.path.abspath(os.path.join(os.path.dirname(absolute), asset_path))
        try:
            if os.path.commonpath([working_dir, candidate]) != working_dir or not os.path.isfile(candidate):
                missing.append(asset)
        except ValueError:
            missing.append(asset)
    if missing:
        return json.dumps({"success": False, "tool": "verify_static_site", "error": "Missing local assets: " + ", ".join(missing)})

    broken_anchors = [
        link for link in parser.anchor_links
        if link != "#" and unquote(link[1:]) not in parser.anchors
    ]
    if broken_anchors:
        return json.dumps({"success": False, "tool": "verify_static_site", "error": "Broken navigation anchors: " + ", ".join(broken_anchors)})

    # Parse local JSON data before claiming the page's data dependencies are
    # healthy. A product catalog with malformed JSON otherwise appears valid
    # until the browser reaches its fetch/render path.
    invalid_json: list[str] = []
    for entry in os.listdir(working_dir):
        if not entry.lower().endswith(".json"):
            continue
        try:
            with open(os.path.join(working_dir, entry), encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_json.append(entry)
    if invalid_json:
        return json.dumps({"success": False, "tool": "verify_static_site", "error": "Invalid JSON data: " + ", ".join(invalid_json)})

    # Node's parser catches syntax errors before a browser can execute the
    # linked scripts. Runtime console assertions still require a configured
    # browser adapter, so this result never claims they were observed.
    node = next((candidate for candidate in ("node", "node.exe") if _command_available(candidate)), None)
    if node:
        bad_scripts: list[str] = []
        for asset in parser.assets:
            if not asset.lower().split("?", 1)[0].endswith(".js"):
                continue
            script_path = os.path.join(os.path.dirname(absolute), unquote(urlsplit(asset).path).lstrip("/\\"))
            check = subprocess.run([node, "--check", script_path], capture_output=True, text=True, timeout=10)
            if check.returncode != 0:
                bad_scripts.append(asset)
        if bad_scripts:
            return json.dumps({"success": False, "tool": "verify_static_site", "error": "JavaScript syntax check failed: " + ", ".join(bad_scripts)})

    server_result = json.loads(ensure_static_server({"path": requested}))
    if not server_result.get("success"):
        return json.dumps(server_result)
    try:
        with urllib.request.urlopen(server_result["url"], timeout=5.0) as response:
            status = response.status
    except OSError as exc:
        return json.dumps({"success": False, "tool": "verify_static_site", "error": f"HTTP validation failed: {exc}"})
    if status >= 400:
        return json.dumps({"success": False, "tool": "verify_static_site", "error": f"Page returned HTTP {status}"})
    return json.dumps({"success": True, "tool": "verify_static_site", "url": server_result["url"], "assets_checked": len(parser.assets), "anchors_checked": len(parser.anchor_links), "json_checked": True, "javascript_syntax_checked": bool(node), "browser_runtime_checked": False})


def known_server_urls() -> set[str]:
    """
    Return the set of base URLs (http://127.0.0.1:<port>/) for every
    server CODI currently has running, across all working dirs it has
    served in this process. Used by core/executor.py's browser-navigation
    guard to verify a requested URL actually corresponds to a real,
    CODI-started server rather than a guessed/hallucinated one.
    """
    with _LOCK:
        return {
            entry["base_url"]
            for entry in _SERVERS.values()
            if _is_alive(entry["process"])
        }


def shutdown_static_servers() -> None:
    """Call at program exit to clean up any background static servers."""
    with _LOCK:
        for entry in _SERVERS.values():
            try:
                if _is_alive(entry["process"]):
                    entry["process"].terminate()
            except Exception:
                pass
        _SERVERS.clear()


def register_static_server_tools(registry) -> None:
    registry.register_local("serve_static", ensure_static_server)
    registry.register_local("verify_static_site", verify_static_site)
