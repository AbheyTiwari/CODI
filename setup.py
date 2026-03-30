#!/usr/bin/env python3
"""
CODI Setup Script
Run once after cloning: python setup.py
Works on Windows, macOS, Linux — no manual config needed.
"""

import os
import sys
import json
import shutil
import platform
import subprocess
from pathlib import Path

ROOT    = Path(__file__).parent.resolve()
ENV_FILE = ROOT / ".env"
MCP_FILE = ROOT / "mcp_servers.json"

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
IS_LIN = platform.system() == "Linux"
HOME   = Path.home()

# ── Bootstrap Rich (install it if missing, we need it for the UI) ─────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.table import Table
    from rich import box
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich.table import Table
    from rich import box

console = Console()

# ── Theme ─────────────────────────────────────────────────────────────────────
A  = "bright_cyan"      # accent
DIM = "dim cyan"

def ok(msg):
    console.print(Text(f"  ✓  {msg}", style="bright_green"))

def warn(msg):
    console.print(Text(f"  ⚠  {msg}", style="yellow"))

def err(msg):
    console.print(Text(f"  ✗  {msg}", style="red"))

def info(msg):
    console.print(Text(f"  →  {msg}", style="dim"))

def step(n, total, title):
    console.print()
    console.print(Rule(
        Text(f" {n}/{total}  {title} ", style=f"bold {A}"),
        style=DIM,
    ))
    console.print()

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = """\
  ██████╗ ██████╗ ██████╗ ██╗
 ██╔════╝██╔═══██╗██╔══██╗██║
 ██║     ██║   ██║██║  ██║██║
 ██║     ██║   ██║██║  ██║██║
 ╚██████╗╚██████╔╝██████╔╝██║
  ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝"""

console.print()
for line in BANNER.splitlines():
    console.print(Text(line, style=f"bold {A}"), justify="center")
console.print()
console.print(Rule(style=DIM))

# System info row
sysinfo = Text()
sysinfo.append(f"  {platform.system()} {platform.release()}", style="dim")
sysinfo.append("  ·  ", style="dim")
sysinfo.append(f"Python {sys.version.split()[0]}", style="dim")
sysinfo.append("  ·  ", style="dim")
sysinfo.append(str(ROOT), style="dim")
console.print(sysinfo)
console.print(Rule(style=DIM))
console.print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Resolve tool paths
# ─────────────────────────────────────────────────────────────────────────────
step(1, 5, "Resolving tool paths")

NPX    = shutil.which("npx") or shutil.which("npx.cmd")
UVX    = shutil.which("uvx")
UV     = shutil.which("uv")
OLLAMA = shutil.which("ollama")

table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
table.add_column(style=A, no_wrap=True, width=10)
table.add_column(style="dim")
table.add_column()

def _tool_row(name, path, needed_for):
    if path:
        table.add_row(name, path, Text("✓", style="bright_green"))
    else:
        table.add_row(name, f"not found — {needed_for}", Text("✗", style="red"))

_tool_row("npx",    NPX,    "npm MCP servers")
_tool_row("uvx",    UVX,    "uvx MCP servers")
_tool_row("ollama", OLLAMA, "local/hybrid mode")

console.print(table)

if not NPX:
    warn("Install Node.js to enable npm MCP servers: https://nodejs.org")

if not UVX:
    if UV:
        info("uv found — installing uvx via pip")
    else:
        info("uvx not found — installing via pip install uv")
    subprocess.run([sys.executable, "-m", "pip", "install", "uv", "-q"], capture_output=True)
    UVX = shutil.which("uvx")
    if UVX:
        ok(f"uvx installed → {UVX}")
    else:
        warn("uvx still not found. Fix: pip install uv  then restart terminal")

if not OLLAMA:
    warn("Ollama not found. Install for local/hybrid mode: https://ollama.com")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Fix mcp_servers.json
# ─────────────────────────────────────────────────────────────────────────────
step(2, 5, "Configuring MCP servers")

PLACEHOLDER = "__CODI_ROOT__"

if not MCP_FILE.exists():
    warn("mcp_servers.json not found — skipping")
else:
    raw     = MCP_FILE.read_text(encoding="utf-8")
    changes = 0
    actual_root = str(ROOT).replace("\\", "/")

    if PLACEHOLDER in raw:
        raw = raw.replace(PLACEHOLDER, actual_root)
        changes += 1
        ok(f"Resolved __CODI_ROOT__ → {actual_root}")

    data = json.loads(raw)

    for name, cfg in data.items():
        args     = cfg.get("args", [])
        new_args = []
        for arg in args:
            if not IS_WIN and ("C:\\" in str(arg) or "C:/" in str(arg)):
                new_args.append(actual_root)
                changes += 1
                warn(f"Replaced Windows path in {name}")
            elif not IS_WIN and "/home/kali" in str(arg) and str(ROOT) not in str(arg):
                new_args.append(actual_root)
                changes += 1
                warn(f"Replaced kali path in {name}")
            else:
                new_args.append(arg)
        cfg["args"] = new_args

        if cfg.get("enabled") and cfg.get("type") not in ("sse", "http"):
            cmd = cfg.get("command", "")
            if cmd == "uvx" and not UVX:
                cfg["enabled"] = False
                warn(f"Disabled {name} (uvx unavailable)")
                changes += 1
            elif cmd == "npx" and not NPX:
                cfg["enabled"] = False
                warn(f"Disabled {name} (npx unavailable)")
                changes += 1

        if name == "sqlite" and cfg.get("enabled"):
            cfg["enabled"] = False
            warn("Disabled sqlite (@modelcontextprotocol/server-sqlite removed from npm)")
            changes += 1

    MCP_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    if changes:
        ok(f"mcp_servers.json updated ({changes} change{'s' if changes != 1 else ''})")
    else:
        ok("mcp_servers.json already correct")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Create / clean .env
# ─────────────────────────────────────────────────────────────────────────────
step(3, 5, "Setting up .env")

CLEAN_ENV = """\
# CODI environment variables
# Do NOT commit this file to git

# LLM Providers (fill in the ones you want to use)
CODI_GROQ_API_KEY=
CODI_ANTHROPIC_API_KEY=
CODI_GEMINI_API_KEY=
CODI_OPENAI_API_KEY=

# Air LLM - local Wi-Fi inference (see github.com/lyogavin/airllm)
# Set to the LAN IP of the device running the AirLLM server
CODI_AIR_LLM_URL=

# MCP Servers
GITHUB_PERSONAL_ACCESS_TOKEN=
GITLAB_PERSONAL_ACCESS_TOKEN=
BRAVE_API_KEY=
SENTRY_AUTH_TOKEN=
STITCH_API_KEY=

# Optional - removes HuggingFace rate limit warnings
HF_TOKEN=
"""

if ENV_FILE.exists():
    raw      = ENV_FILE.read_text(encoding="utf-8")
    has_fancy = "\u2500" in raw or "\u2014" in raw or "──" in raw
    bad_chars = any(ord(c) > 127 for c in raw if c not in "'\"\n\r\t ")
    if has_fancy or bad_chars:
        clean = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                if all(ord(c) < 128 for c in stripped):
                    clean.append(line)
            else:
                clean.append(line)
        ENV_FILE.write_text("\n".join(clean) + "\n", encoding="utf-8")
        ok(".env cleaned (removed non-ASCII characters)")
    else:
        ok(".env exists and looks clean")

    existing = ENV_FILE.read_text(encoding="utf-8")
    added    = []
    for line in CLEAN_ENV.splitlines():
        if "=" in line and not line.startswith("#"):
            key = line.split("=")[0].strip()
            if key and key not in existing:
                with open(ENV_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n{line}\n")
                added.append(key)
    if added:
        ok(f"Added missing keys: {', '.join(added)}")
else:
    ENV_FILE.write_text(CLEAN_ENV, encoding="utf-8")
    ok(".env template created — fill in your API keys")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
step(4, 5, "Installing Python dependencies")

torch_ok = False
try:
    import torch
    torch_ok = True
    ok(f"torch {torch.__version__} already installed")
except ImportError:
    info("Installing PyTorch CPU (avoids pulling 366 MB CUDA build)")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu", "-q"],
    )
    torch_ok = r.returncode == 0
    if torch_ok:
        ok("torch (CPU) installed")
    else:
        warn("torch install failed — sentence-transformers may pull CUDA build")

req = ROOT / "requirements.txt"
if req.exists():
    with console.status(Text("  installing requirements.txt...", style="dim"),
                        spinner="dots", spinner_style=A):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req), "-q"])
    if r.returncode == 0:
        ok("All requirements installed")
    else:
        err("Some requirements failed — check output above")
else:
    warn("requirements.txt not found — skipping")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Create launcher
# ─────────────────────────────────────────────────────────────────────────────
step(5, 5, "Creating launcher")

PYTHON = Path(sys.executable)

if IS_WIN:
    bat = ROOT / "codi.bat"
    bat.write_text(
        f'@echo off\n"{PYTHON}" "{ROOT / "main.py"}" %*\n',
        encoding="utf-8"
    )
    ok("Created codi.bat")
    info(f"Add {ROOT} to PATH to use 'codi' from anywhere")
else:
    sh = ROOT / "codi.sh"
    sh.write_text(f'#!/bin/bash\n"{PYTHON}" "{ROOT / "main.py"}" "$@"\n')
    sh.chmod(0o755)
    ok("Created codi.sh")

    shell = os.environ.get("SHELL", "")
    rc    = HOME / (".zshrc" if "zsh" in shell else ".bashrc")
    alias = f'alias codi="python {ROOT / "main.py"}"\n'

    if rc.exists():
        content = rc.read_text(encoding="utf-8")
        if "alias codi=" not in content:
            rc.write_text(content + "\n# CODI\n" + alias, encoding="utf-8")
            ok(f"Added 'codi' alias to ~/{rc.name}")
            info(f"Run:  source ~/{rc.name}  (or open a new terminal)")
        else:
            ok(f"'codi' alias already in ~/{rc.name}")
    else:
        warn(f"Could not find {rc} — add alias manually:")
        info(alias.strip())

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
console.print()
console.print(Rule(style=DIM))
console.print()

# Next steps table
next_steps = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=False)
next_steps.add_column(style=A, no_wrap=True)
next_steps.add_column(style="dim")

if IS_WIN:
    next_steps.add_row("codi.bat",      "launch from this directory")
    next_steps.add_row("codi",          "from anywhere (after adding to PATH)")
else:
    next_steps.add_row("python main.py","from this directory")
    next_steps.add_row("codi",          f"from anywhere (after: source ~/{rc.name})")

next_steps.add_row("", "")
next_steps.add_row("config.py",    "set MODE, models, API providers")
next_steps.add_row(".env",         "add API keys")

console.print(Panel(
    next_steps,
    title=Text(" setup complete ", style=f"bold {A}"),
    border_style=DIM,
    padding=(0, 1),
))
console.print()

if not OLLAMA and ENV_FILE.stat().st_size < 200:
    console.print(Panel(
        Text(
            "  Neither Ollama nor API keys detected.\n"
            "  Set MODE='cloud' in config.py and add CODI_GROQ_API_KEY to .env\n"
            "  Get a free Groq key at: console.groq.com",
            style="yellow"
        ),
        border_style="yellow",
        padding=(0, 1),
    ))
    console.print()
