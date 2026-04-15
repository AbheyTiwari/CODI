import sys
import os
import time
import hashlib
import threading

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_LAUNCH_DIR = os.getcwd()
os.environ.setdefault("CODI_WORKING_DIR", _LAUNCH_DIR)

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

THEME = {
    "local":  {"accent": "bright_green",   "label": "LOCAL",  "dim": "green"},
    "hybrid": {"accent": "bright_cyan",    "label": "HYBRID", "dim": "cyan"},
    "cloud":  {"accent": "bright_magenta", "label": "CLOUD",  "dim": "magenta"},
    "air":    {"accent": "bright_yellow",  "label": "AIR",    "dim": "yellow"},
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
    console.print(Rule(style=t["dim"]))
    provider = "ollama" if config.MODE == "local" else config.CLOUD_PROVIDER
    status = Text()
    status.append(f" {t['label']} ", style=f"bold reverse {t['accent']}")
    status.append(f"  {provider}  ·  {_LAUNCH_DIR}", style="dim")
    console.print(status)
    console.print(Rule(style=t["dim"]))
    console.print()

def startup_sequence():
    t = _t()
    for label in ["Vector DB", "Cognitive Refiner", "MCP Servers", "Agent Graph"]:
        with console.status(Text(f"  loading {label}...", style="dim"),
                            spinner="dots", spinner_style=t["accent"]):
            time.sleep(0.2)
        console.print(Text(f"  ✓  {label}", style=t["accent"]))
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
            console.print(Text("  ⚠  >5000 files — skipping auto-index. Run /index . manually.", style="yellow"))
            return
    t = _t()
    with console.status(Text(f"  indexing {os.path.basename(_LAUNCH_DIR)}/...", style="dim"),
                        spinner="dots", spinner_style=t["accent"]):
        try:
            index_codebase(_LAUNCH_DIR, db_path=_chroma_dir)
        except Exception as e:
            console.print(Text(f"  ⚠  index warning: {e}", style="yellow"))
    console.print(Text(f"  ✓  indexed {os.path.basename(_LAUNCH_DIR)}/", style=t["accent"]))
    console.print()

# ── Help ──────────────────────────────────────────────────────────────────────
def print_help():
    t = _t()
    table = Table(box=box.SIMPLE, show_header=False, padding=(0,2), expand=False)
    table.add_column(style=t["accent"], no_wrap=True, min_width=20)
    table.add_column(style="dim")
    for cmd, desc in [
        ("/index [path]",   "re-index directory (default: current project)"),
        ("/mode",           "show current mode"),
        ("/mode <name>", "switch mode: local | hybrid | cloud | air"),
        ("/mcp",            "list MCP servers"),
        ("/mcp on <name>",  "enable MCP server (restart to apply)"),
        ("/mcp off <name>", "disable MCP server (restart to apply)"),
        ("/tools",          "list all loaded tools"),
        ("/logs",           "live telemetry dashboard"),
        ("/clear",          "wipe session memory"),
        ("/history",        "print conversation history"),
        ("/help",           "show this message"),
        ("/quit",           "exit"),
    ]:
        table.add_row(cmd, desc)
    console.print(Panel(table, title=Text("commands", style=t["accent"]),
                        border_style=t["dim"], padding=(0,1)))
    console.print()

# ── Live renderer ─────────────────────────────────────────────────────────────
class LiveRenderer:
    def __init__(self, task: str):
        self.task  = task[:80]
        self.lines = []
        self._live = None
        self._lock = threading.Lock()

    def _panel(self):
        t = _t()
        body = Text()
        body.append(f"  {self.task}\n\n", style="bold")
        for line in self.lines[-10:]:
            body.append(f"  ·  {line.strip()}\n", style="dim")
        return Panel(body, title=Text(f" {t['label']} · working ", style=f"bold {t['accent']}"),
                     border_style=t["dim"], padding=(0,1))

    def push(self, line: str):
        with self._lock:
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
    import json
    path = os.path.join(_REPO_ROOT, "mcp_servers.json")
    try:
        cfg = json.load(open(path))
        if name not in cfg:
            console.print(Text(f"  '{name}' not found", style="red"))
            return
        cfg[name]["enabled"] = enabled
        json.dump(cfg, open(path, "w"), indent=2)
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
    colours = {"local":"#00ff88","hybrid":"#00ccff","cloud":"#cc66ff","air":"#ffaa00"}
    c = colours.get(config.MODE, "#00ccff")
    return PTStyle.from_dict({"prompt": f"fg:{c} bold", "": "fg:#cccccc"})

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print_banner()
    startup_sequence()
    _auto_index()

    with console.status(Text("  initialising agent...", style="dim"), spinner="dots"):
        agent_executor = create_agent()
    console.print(Text(f"  ✓  agent ready\n", style=_t()["accent"]))
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
        # Note: bare URLs are NOT blocked — users should be able to say
        # "check this repo https://github.com/..." in their message
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
            import json
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
            from tools import get_all_tools
            all_t = get_all_tools()
            table = Table(box=box.SIMPLE, show_header=False, padding=(0,2))
            table.add_column(style=t["accent"], no_wrap=True)
            table.add_column(style="dim")
            for tool in all_t:
                table.add_row(tool.name, (tool.description or "")[:70])
            console.print(Panel(table,
                                title=Text(f"tools  {len(all_t)}", style=t["accent"]),
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