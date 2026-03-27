# codi/mcp/mcp_client.py
import asyncio
import json
import os
import shutil
import sys
import threading
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

# ── Config path resolution ──────────────────────────────────────────────────
MCP_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "mcp_servers.json")

# ── Dedicated event loop on its own thread ──────────────────────────────────
_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None

def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    if _loop is None or not _loop.is_running():
        _loop = asyncio.new_event_loop()
        _loop_thread = threading.Thread(
            target=_loop.run_forever,
            daemon=True,
            name="mcp-event-loop"
        )
        _loop_thread.start()
    return _loop

# ── Command resolver ─────────────────────────────────────────────────────────
def _resolve_command(command: str) -> str:
    """
    Resolve 'npx' / 'uvx' to full path so Windows subprocess can find them.
    shutil.which() checks PATH and handles .cmd shims automatically.
    """
    resolved = shutil.which(command)
    if resolved:
        return resolved
    if sys.platform == "win32":
        for ext in (".cmd", ".exe", ".bat"):
            resolved = shutil.which(command + ext)
            if resolved:
                return resolved
    return command  # fall back — error surfaces at connection time

# ── Config loader ────────────────────────────────────────────────────────────
def load_mcp_config() -> dict:
    if not os.path.exists(MCP_CONFIG_FILE):
        print(f"  [MCP] Config not found at: {MCP_CONFIG_FILE}")
        return {}
    with open(MCP_CONFIG_FILE, "r") as f:
        return json.load(f)

# ── Env resolver ─────────────────────────────────────────────────────────────
def _resolve_env(raw_env: dict) -> dict:
    resolved = {}
    for key, val in raw_env.items():
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            env_key = val[2:-1]
            resolved[key] = os.environ.get(env_key, "")
        else:
            resolved[key] = os.environ.get(key, val)
    return resolved

# ── SSE loader ───────────────────────────────────────────────────────────────
async def _load_sse(server_name: str, cfg: dict) -> list:
    url = cfg.get("url")
    if not url:
        raise ValueError(f"{server_name}: SSE type requires a 'url' field in mcp_servers.json")

    headers = cfg.get("headers", {})
    env_vars = _resolve_env(cfg.get("env", {}))
    headers.update({k: v for k, v in env_vars.items() if v})

    async with sse_client(url, headers=headers) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await load_mcp_tools(session)

# ── stdio loader ─────────────────────────────────────────────────────────────
async def _load_stdio(server_name: str, cfg: dict) -> list:
    resolved_env = _resolve_env(cfg.get("env", {}))
    merged_env = {**os.environ, **resolved_env}

    # Resolve command to full path so Windows subprocess can find npx/uvx
    command = _resolve_command(cfg["command"])

    params = StdioServerParameters(
        command=command,
        args=cfg.get("args", []),
        env=merged_env
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await load_mcp_tools(session)

# ── Main async dispatcher ─────────────────────────────────────────────────────
async def _load_server(server_name: str, cfg: dict) -> list:
    t = cfg.get("type")
    if t in ("sse", "http"):
        return await _load_sse(server_name, cfg)
    return await _load_stdio(server_name, cfg)

# ── Public sync interface ─────────────────────────────────────────────────────
def get_mcp_tools(server_name: str, server_config: dict) -> list:
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(
        _load_server(server_name, server_config),
        loop
    )
    try:
        return future.result(timeout=30)
    except TimeoutError:
        print(f"  [MCP] {server_name}: timed out after 30s")
        return []
    except Exception as e:
        print(f"  [MCP] {server_name}: failed — {e}")
        return []

def load_all_mcp_tools() -> list:
    config = load_mcp_config()
    if not config:
        return []

    all_tools = []
    for name, cfg in config.items():
        if not cfg.get("enabled", False):
            continue
        print(f"  [MCP] Connecting to {name}...")
        tools = get_mcp_tools(name, cfg)
        print(f"  [MCP] {name}: loaded {len(tools)} tools")
        all_tools.extend(tools)

    return all_tools

def shutdown():
    global _loop
    if _loop and _loop.is_running():
        _loop.call_soon_threadsafe(_loop.stop)