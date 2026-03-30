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

ROOT = Path(__file__).parent.resolve()
ENV_FILE = ROOT / ".env"
MCP_FILE = ROOT / "mcp_servers.json"

# ── Colors ────────────────────────────────────────────────────────
G = lambda s: f"\033[92m{s}\033[0m"
Y = lambda s: f"\033[93m{s}\033[0m"
R = lambda s: f"\033[91m{s}\033[0m"
B = lambda s: f"\033[1m{s}\033[0m"

def ok(msg):   print(f"  {G('✓')} {msg}")
def warn(msg): print(f"  {Y('⚠')} {msg}")
def err(msg):  print(f"  {R('✗')} {msg}")
def info(msg): print(f"  {B('→')} {msg}")
def head(msg): print(f"\n  {B(msg)}")

IS_WIN  = platform.system() == "Windows"
IS_MAC  = platform.system() == "Darwin"
IS_LIN  = platform.system() == "Linux"
HOME    = Path.home()

print()
print(B("  ╔══════════════════════════════╗"))
print(B("  ║     CODI Setup Assistant     ║"))
print(B("  ╚══════════════════════════════╝"))
print(f"\n  OS     : {platform.system()} {platform.release()}")
print(f"  Python : {sys.version.split()[0]}")
print(f"  Root   : {ROOT}")

# ─────────────────────────────────────────────────────────────────
# STEP 1 — Resolve tool paths
# ─────────────────────────────────────────────────────────────────
head("[1/5] Resolving tool paths")

NPX = shutil.which("npx") or shutil.which("npx.cmd")
UVX = shutil.which("uvx")
UV  = shutil.which("uv")
OLLAMA = shutil.which("ollama")

if NPX:
    ok(f"npx   → {NPX}")
else:
    warn("npx not found — npm MCP servers disabled")
    warn("Install Node.js: https://nodejs.org")

if UVX:
    ok(f"uvx   → {UVX}")
else:
    if UV:
        info("uv found — trying to install uvx via pip install uv")
    else:
        info("uvx not found — installing via pip install uv")
    subprocess.run([sys.executable, "-m", "pip", "install", "uv", "-q"],
                   capture_output=True)
    UVX = shutil.which("uvx")
    if UVX:
        ok(f"uvx   → {UVX}")
    else:
        warn("uvx still not found — uvx MCP servers disabled")
        warn("Fix: pip install uv  then restart terminal")

if OLLAMA:
    ok(f"ollama → {OLLAMA}")
else:
    warn("ollama not found — local/hybrid mode needs it")
    warn("Install: https://ollama.com")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Fix mcp_servers.json
# ─────────────────────────────────────────────────────────────────
head("[2/5] Configuring MCP servers")

PLACEHOLDER = "__CODI_ROOT__"

if not MCP_FILE.exists():
    warn("mcp_servers.json not found — skipping")
else:
    raw = MCP_FILE.read_text(encoding="utf-8")
    changes = 0

    # Replace __CODI_ROOT__ placeholder with actual path
    actual_root = str(ROOT).replace("\\", "/")
    if PLACEHOLDER in raw:
        raw = raw.replace(PLACEHOLDER, actual_root)
        changes += 1
        ok(f"Resolved __CODI_ROOT__ → {actual_root}")

    data = json.loads(raw)

    for name, cfg in data.items():
        # Fix any remaining Windows paths on non-Windows
        args = cfg.get("args", [])
        new_args = []
        for arg in args:
            if not IS_WIN and ("C:\\" in str(arg) or "C:/" in str(arg)):
                # Replace Windows path with project root
                new_args.append(actual_root)
                changes += 1
                warn(f"Replaced Windows path in {name} args: {arg}")
            elif not IS_WIN and "/home/kali" in str(arg) and str(ROOT) not in str(arg):
                new_args.append(actual_root)
                changes += 1
                warn(f"Replaced kali path in {name} args: {arg}")
            else:
                new_args.append(arg)
        cfg["args"] = new_args

        # Disable servers whose commands aren't available
        if cfg.get("enabled") and cfg.get("type") not in ("sse", "http"):
            cmd = cfg.get("command", "")
            if cmd == "uvx" and not UVX:
                cfg["enabled"] = False
                warn(f"Disabled {name} (uvx unavailable)")
                changes += 1
            elif cmd in ("npx",) and not NPX:
                cfg["enabled"] = False
                warn(f"Disabled {name} (npx unavailable)")
                changes += 1

        # sqlite npm package was removed — always disable
        if name == "sqlite" and cfg.get("enabled"):
            cfg["enabled"] = False
            warn("Disabled sqlite (@modelcontextprotocol/server-sqlite removed from npm)")
            changes += 1

    MCP_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if changes:
        ok(f"mcp_servers.json updated ({changes} changes)")
    else:
        ok("mcp_servers.json already correct")

# ─────────────────────────────────────────────────────────────────
# STEP 3 — Create / clean .env
# ─────────────────────────────────────────────────────────────────
head("[3/5] Setting up .env")

CLEAN_ENV = """\
# CODI environment variables
# Do NOT commit this file to git

# LLM Providers (fill in the ones you want to use)
CODI_GROQ_API_KEY=
CODI_ANTHROPIC_API_KEY=
CODI_GEMINI_API_KEY=
CODI_OPENAI_API_KEY=

# Air LLM - phone inference over Wi-Fi
# Get app: https://play.google.com/store/apps/details?id=com.airlm.app
# Set to your phone's LAN IP shown in the app, e.g. http://192.168.1.45:8080
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
    raw = ENV_FILE.read_text(encoding="utf-8")
    # Check for Unicode characters that break python-dotenv
    bad_chars = any(ord(c) > 127 for c in raw if c not in "'\"\n\r\t ")
    has_fancy  = "\u2500" in raw or "\u2014" in raw or "──" in raw
    if has_fancy or bad_chars:
        # Strip comment lines with bad chars, keep KEY=value lines
        clean = []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                # Only keep ASCII-clean comment lines
                if all(ord(c) < 128 for c in stripped):
                    clean.append(line)
                # else: drop fancy comment line
            else:
                clean.append(line)
        ENV_FILE.write_text("\n".join(clean) + "\n", encoding="utf-8")
        ok(".env cleaned (removed Unicode characters that broke python-dotenv)")
    else:
        ok(".env exists and looks clean")

    # Ensure all expected keys exist (don't overwrite existing values)
    existing = ENV_FILE.read_text(encoding="utf-8")
    added = []
    for line in CLEAN_ENV.splitlines():
        if "=" in line and not line.startswith("#"):
            key = line.split("=")[0].strip()
            if key and key not in existing:
                with open(ENV_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n{line}\n")
                added.append(key)
    if added:
        ok(f"Added missing keys to .env: {', '.join(added)}")

else:
    ENV_FILE.write_text(CLEAN_ENV, encoding="utf-8")
    ok(".env template created — fill in your API keys")

# ─────────────────────────────────────────────────────────────────
# STEP 4 — Install Python dependencies
# ─────────────────────────────────────────────────────────────────
head("[4/5] Installing Python dependencies")

# Install CPU torch first to prevent 366MB CUDA download
torch_ok = False
try:
    import torch
    torch_ok = True
    ok(f"torch {torch.__version__} already installed")
except ImportError:
    info("Installing PyTorch CPU (prevents auto-downloading 366MB CUDA build)")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch",
         "--index-url", "https://download.pytorch.org/whl/cpu", "-q"],
    )
    torch_ok = r.returncode == 0
    if torch_ok:
        ok("torch (CPU) installed")
    else:
        warn("torch install failed — sentence-transformers may pull CUDA")

req = ROOT / "requirements.txt"
if req.exists():
    info("Installing requirements.txt")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req), "-q"])
    if r.returncode == 0:
        ok("All requirements installed")
    else:
        err("Some requirements failed — check output above")
else:
    warn("requirements.txt not found — skipping")

# ─────────────────────────────────────────────────────────────────
# STEP 5 — Create launcher
# ─────────────────────────────────────────────────────────────────
head("[5/5] Creating launcher")

PYTHON = Path(sys.executable)

if IS_WIN:
    bat = ROOT / "codi.bat"
    bat.write_text(
        f'@echo off\n"{PYTHON}" "{ROOT / "main.py"}" %*\n',
        encoding="utf-8"
    )
    ok(f"Created codi.bat")
    info(f"Add {ROOT} to PATH to run 'codi' from anywhere")
    info(f"Or run: {ROOT}\\codi.bat")

else:
    # Shell script
    sh = ROOT / "codi.sh"
    sh.write_text(f'#!/bin/bash\n"{PYTHON}" "{ROOT / "main.py"}" "$@"\n')
    sh.chmod(0o755)
    ok("Created codi.sh")

    # Add alias to shell rc
    shell = os.environ.get("SHELL", "")
    rc = HOME / (".zshrc" if "zsh" in shell else ".bashrc")
    alias = f'alias codi="python {ROOT / "main.py"}"\n'

    if rc.exists():
        content = rc.read_text(encoding="utf-8")
        if "alias codi=" not in content:
            rc.write_text(content + "\n# CODI\n" + alias, encoding="utf-8")
            ok(f"Added 'codi' alias to {rc.name}")
            info(f"Run: source ~/{rc.name}  (or open new terminal)")
        else:
            ok(f"'codi' alias already in {rc.name}")
    else:
        warn(f"Could not find {rc} — add alias manually:")
        print(f"      {alias.strip()}")

# ─────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────
print()
print(B("  ╔══════════════════════════════╗"))
print(B("  ║       Setup Complete!        ║"))
print(B("  ╚══════════════════════════════╝"))
print()
print(f"  Start CODI:")
print(f"    {G('python main.py')}     (from this directory)")
if not IS_WIN:
    print(f"    {G('codi')}              (from anywhere, after sourcing shell rc)")
print()
print(f"  Edit {Y('config.py')} to set MODE and models.")
print(f"  Edit {Y('.env')} to add API keys.")
print()

if not OLLAMA and not (ROOT / ".env").stat().st_size > 100:
    warn("Neither Ollama nor API keys detected.")
    warn("Set MODE='cloud' in config.py and add a CODI_GROQ_API_KEY to .env")
    print()