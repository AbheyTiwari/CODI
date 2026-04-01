import sys
import os
import time
import hashlib
import json
import requests

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_LAUNCH_DIR = os.getcwd()
os.environ.setdefault("CODI_WORKING_DIR", _LAUNCH_DIR)
os.environ.setdefault("CODI_REPO_ROOT", _REPO_ROOT)

_path_hash  = hashlib.md5(_LAUNCH_DIR.encode()).hexdigest()[:10]
_chroma_dir = os.path.join(_REPO_ROOT, "chroma_db", _path_hash)
os.environ.setdefault("CODI_CHROMA_DIR", _chroma_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(_REPO_ROOT, ".env"))
# ─────────────────────────────────────────────────────────────────────────────

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle

from memory import session_memory
from indexer import index_codebase
from refiner import refine_prompt
from agent import create_agent
from logger import log
from context_trimmer import trim_context_for_llm, estimate_tokens
import config

console = Console()

# ── Theme: unified light green ────────────────────────────────────────────────
THEME = {
    "local":  {"accent": "#77dd77", "label": "LOCAL",  "dim": "#3a6e3a"},
    "hybrid": {"accent": "#77dd77", "label": "HYBRID", "dim": "#3a6e3a"},
    "cloud":  {"accent": "#77dd77", "label": "CLOUD",  "dim": "#3a6e3a"},
    "air":    {"accent": "#77dd77", "label": "AIR",    "dim": "#3a6e3a"},
}

def _t():
    return THEME.get(config.MODE, THEME["hybrid"])

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = """\
  ██████╗ ██████╗ ██████╗ ██╗
 ██╔════╝██╔═══██╗██╔══██╗██║
 ██║     ██║   ██║██║  ██║██║
 ██║     ██║   ██║██║  ██║██║
 ╚██████╗╚██████╔╝██████╔╝██║
  ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝"""

def print_banner():
    t = _t()
    console.print()
    for line in BANNER.splitlines():
        console.print(Text(line, style=f"bold {t['accent']}"), justify="center")
    console.print()

# ── Status (hidden behind /status command) ────────────────────────────────────
def print_status():
    t = _t()
    console.print()
    # Vector DB
    if os.path.isdir(_chroma_dir) and os.listdir(_chroma_dir):
        console.print(Text(f"  ✓  Vector DB", style=t["accent"]))
    else:
        console.print(Text(f"  ○  Vector DB — not indexed", style="dim"))
    # Model backend
    if config.MODE in ("local", "hybrid"):
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                console.print(Text(f"  ✓  Ollama", style=t["accent"]))
            else:
                console.print(Text(f"  ○  Ollama — not responding", style="yellow"))
        except Exception:
            console.print(Text(f"  ✗  Ollama — offline", style="yellow"))
    else:
        console.print(Text(f"  ✓  Cloud ({config.CLOUD_PROVIDER})", style=t["accent"]))
    # MCP
    mcp_path = os.path.join(_REPO_ROOT, "mcp_servers.json")
    if os.path.isfile(mcp_path):
        try:
            cfg = json.load(open(mcp_path))
            enabled = sum(1 for v in cfg.values() if v.get("enabled"))
            console.print(Text(f"  ✓  MCP — {enabled} servers enabled", style=t["accent"]))
        except Exception:
            console.print(Text(f"  ○  MCP — parse error", style="yellow"))
    else:
        console.print(Text(f"  ○  MCP — not found", style="dim"))
    # Mode
    console.print(Text(f"  ✓  Mode — {config.MODE}", style=t["accent"]))
    console.print()

# ── Auto-index ────────────────────────────────────────────────────────────────
def _auto_index():
    skip = {'.git','node_modules','__pycache__','venv','dist',
            'build','.idea','chroma_db','.mypy_cache','.pytest_cache'}
    count = 0
    for root, dirs, files in os.walk(_LAUNCH_DIR):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith('.')]
        count += len(files)
        if count > 5000:
            return
    try:
        index_codebase(_LAUNCH_DIR, db_path=_chroma_dir)
    except Exception:
        pass

# ── Help ──────────────────────────────────────────────────────────────────────
def print_help():
    t = _t()
    table = Table(box=box.SIMPLE, show_header=False, padding=(0,2), expand=False)
    table.add_column(style=t["accent"], no_wrap=True, min_width=20)
    table.add_column(style="dim")
    for cmd, desc in [
        ("/index [path]",   "re-index directory"),
        ("/mode",           "show / switch mode"),
        ("/status",         "system status"),
        ("/mcp",            "list MCP servers"),
        ("/tools",          "list loaded tools"),
        ("/logs",           "telemetry dashboard"),
        ("/clear",          "wipe session memory"),
        ("/history",        "conversation history"),
        ("/help",           "show this"),
        ("/quit",           "exit"),
    ]:
        table.add_row(cmd, desc)
    console.print(Panel(table, title=Text(" commands ", style=t["accent"]),
                        border_style=t["dim"], padding=(0,1)))
    console.print()

# ── Live renderer ─────────────────────────────────────────────────────────────
class LiveRenderer:
    def __init__(self, task: str):
        self.task  = task[:80]
        self.lines = []
        self._live = None

    def _panel(self):
        t = _t()
        body = Text()
        body.append(f"  {self.task}\n\n", style="bold")
        for line in self.lines[-10:]:
            body.append(f"  ·  {line.strip()}\n", style="dim")
        return Panel(body, title=Text(f" {t['label']} · working ", style=f"bold {t['accent']}"),
                     border_style=t["dim"], padding=(0,1))

    def push(self, line: str):
        self.lines.append(line)
        if self._live:
            self._live.update(self._panel())

    def start(self):
        self._live = Live(self._panel(), console=console,
                          refresh_per_second=8, transient=True)
        self._live.start()

    def stop(self):
        if self._live:
            self._live.stop()
            self._live = None

# ── Response renderer ─────────────────────────────────────────────────────────
def render_response(output: str, tool_outputs: list = None):
    t = _t()
    console.print()
    if tool_outputs:
        for line in tool_outputs[-5:]:
            name, _, rest = line.partition(":")
            row = Text()
            row.append(f"  ✓  {name.strip()}", style=t["accent"])
            row.append(f"  {rest.strip()[:80]}", style="dim")
            console.print(row)
        console.print()
    console.print(Panel(Markdown(output), border_style=t["dim"], padding=(0,2)))
    console.print()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _set_mode(new_mode: str):
    valid = {"local", "hybrid", "cloud", "air"}
    if new_mode not in valid:
        console.print(Text(f"  unknown mode '{new_mode}'. valid: {', '.join(valid)}", style="red"))
        return
    config.MODE = new_mode
    t = _t()
    console.print(Text(f"  ✓  mode → {t['label']}", style=t["accent"]))
    log("mode_switch", {"mode": new_mode})

def _mcp_toggle(name: str, enabled: bool):
    import json as _json
    path = os.path.join(_REPO_ROOT, "mcp_servers.json")
    try:
        cfg = _json.load(open(path))
        if name not in cfg:
            console.print(Text(f"  '{name}' not found", style="red"))
            return
        cfg[name]["enabled"] = enabled
        _json.dump(cfg, open(path, "w"), indent=2)
        verb = "enabled" if enabled else "disabled"
        console.print(Text(f"  ✓  {verb} {name} — restart to apply", style=_t()["accent"]))
    except Exception as e:
        console.print(Text(f"  error: {e}", style="red"))

def get_trimmed_history() -> str:
    full = session_memory.as_text()
    return trim_context_for_llm(
        user_input="", history=full, tool_outputs=[], mode=config.MODE
    )["history"]

def _pt_style():
    return PTStyle.from_dict({"prompt": "fg:#77dd77 bold", "": "fg:#cccccc"})

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print_banner()

    with console.status(Text("  loading...", style=_t()["dim"]),
                        spinner="dots", spinner_style=_t()["accent"]):
        _auto_index()
        agent_executor = create_agent()

    # Cache tool list once at init
    from tools import get_all_tools
    _cached_tools = get_all_tools()

    console.print(Text("  ready\n", style=_t()["accent"]))
    print_help()

    session = PromptSession(
        history=FileHistory(os.path.join(_REPO_ROOT, ".agent_history")),
        style=_pt_style(),
    )

    while True:
        t = _t()
        try:
            user_input = session.prompt(
                [("class:prompt", "❯ ")], style=_pt_style()
            ).strip()
        except (KeyboardInterrupt, EOFError):
            console.print(Text("\n  goodbye\n", style="dim"))
            break

        if not user_input:
            continue

        # ── Guard: only block obviously pasted terminal output ────────────────
        if (
            user_input.startswith(">>")
            or user_input.startswith("Graph execution failed")
            or user_input.startswith("Traceback (most recent")
            or user_input.startswith("Thinking...")
        ):
            console.print(Text("  ⚠  that looks like pasted terminal output.", style="yellow"))
            continue

        cmd = user_input.lower().strip()

        # ── Commands ──────────────────────────────────────────────────────────
        if cmd in ("/quit", "/exit"):
            console.print(Text("\n  goodbye\n", style="dim"))
            break

        elif cmd == "/help":
            print_help()

        elif cmd == "/status":
            print_status()

        elif cmd == "/clear":
            session_memory.clear()
            console.print(Text("  ✓  session memory cleared", style=t["accent"]))

        elif cmd == "/history":
            console.print(Panel(session_memory.as_text() or "No history yet.",
                                title=Text("history", style=t["accent"]),
                                border_style=t["dim"]))

        elif cmd.startswith("/index"):
            parts = user_input.split(maxsplit=1)
            path  = os.path.abspath(parts[1].strip() if len(parts) > 1 else _LAUNCH_DIR)
            with console.status(Text(f"  indexing {path}...", style="dim"), spinner="dots"):
                index_codebase(path)
            console.print(Text(f"  ✓  indexed {path}", style=t["accent"]))

        elif cmd == "/mode":
            rows = [
                ("local",  "100% offline · Ollama only"),
                ("hybrid", "Ollama first · cloud fallback"),
                ("cloud",  f"always cloud · {config.CLOUD_PROVIDER}"),
                ("air",    f"Air LLM · {config.AIR_LLM_URL}"),
            ]
            table = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
            table.add_column(no_wrap=True)
            table.add_column(style="dim")
            for mode, desc in rows:
                is_cur = mode == config.MODE
                marker = Text(f"▶  {mode}" if is_cur else f"   {mode}",
                              style=f"bold {t['accent']}" if is_cur else "dim")
                table.add_row(marker, desc)
            console.print(Panel(table, title=Text("mode", style=t["accent"]),
                                border_style=t["dim"]))

        elif cmd.startswith("/mode "):
            _set_mode(user_input.split(maxsplit=1)[1].strip().lower())

        elif cmd == "/mcp":
            try:
                cfg = json.load(open(os.path.join(_REPO_ROOT, "mcp_servers.json")))
                table = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
                table.add_column(no_wrap=True, width=5)
                table.add_column(style=t["accent"], no_wrap=True)
                table.add_column(style="dim")
                for name, srv in cfg.items():
                    on    = srv.get("enabled", False)
                    badge = Text("ON ", style="bold green") if on else Text("OFF", style="dim red")
                    table.add_row(badge, name, srv.get("description",""))
                console.print(Panel(table, title=Text("mcp servers", style=t["accent"]),
                                    border_style=t["dim"]))
            except Exception as e:
                console.print(Text(f"  error: {e}", style="red"))

        elif cmd.startswith("/mcp on "):
            _mcp_toggle(user_input.split(maxsplit=2)[2], True)

        elif cmd.startswith("/mcp off "):
            _mcp_toggle(user_input.split(maxsplit=2)[2], False)

        elif cmd == "/tools":
            table = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
            table.add_column(style=t["accent"], no_wrap=True)
            table.add_column(style="dim")
            for tool in _cached_tools:
                table.add_row(tool.name, (tool.description or "")[:70])
            console.print(Panel(table,
                                title=Text(f"tools  {len(_cached_tools)}", style=t["accent"]),
                                border_style=t["dim"]))

        elif cmd == "/logs":
            import subprocess
            subprocess.run([sys.executable, os.path.join(_REPO_ROOT, "log_viewer.py")])

        # ── Natural language ──────────────────────────────────────────────────
        else:
            refined = refine_prompt(user_input)
            if refined != user_input:
                console.print(Text(f"  → {refined}", style="dim"))

            session_memory.add("user", refined)
            renderer = LiveRenderer(refined)
            renderer.start()

            try:
                log("agent_start", {"input": refined[:200], "mode": config.MODE})
                history_str = get_trimmed_history()
                token_est   = estimate_tokens(refined + history_str)

                if token_est > 3000 and config.MODE in ("local", "air"):
                    renderer.push(f"⚠ context ~{token_est} tokens — consider /clear")

                response     = agent_executor.invoke({"input": refined, "history": history_str})
                output       = response.get("output", "No output returned.")
                tool_outputs = response.get("tool_outputs", [])

                log("agent_end", {"output": output[:200], "mode": config.MODE})
                renderer.stop()
                render_response(output, tool_outputs)
                session_memory.add("assistant", output)

            except Exception as e:
                renderer.stop()
                err = str(e)
                log("agent_error", {"error": err, "mode": config.MODE})
                if "413" in err or "Request too large" in err:
                    console.print(Panel(
                        Text("token limit hit — try /clear or /mode cloud", style="red"),
                        border_style="red", padding=(0,2)))
                elif "Connection refused" in err or "ConnectError" in err:
                    console.print(Panel(
                        Text(f"{config.MODE} backend unreachable — try /mode cloud", style="red"),
                        border_style="red", padding=(0,2)))
                else:
                    console.print(Panel(Text(str(e), style="red"),
                                        title=Text("error", style="red"),
                                        border_style="red", padding=(0,2)))


if __name__ == "__main__":
    main()