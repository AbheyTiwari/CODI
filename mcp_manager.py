import asyncio
import threading
import json
import os
import shutil
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from logger import log


def _resolve_command(command: str) -> str:
    """
    Resolve a command like 'npx' or 'uvx' to its full executable path.
    On Windows, shutil.which() finds the .cmd shim that subprocess needs.
    Falls back to the original string if not found (non-Windows or already absolute).
    """
    resolved = shutil.which(command)
    if resolved:
        return resolved
    # On Windows, also try .cmd and .exe extensions explicitly
    if sys.platform == "win32":
        for ext in (".cmd", ".exe", ".bat"):
            resolved = shutil.which(command + ext)
            if resolved:
                return resolved
    return command  # give up gracefully — error will surface at connection time


class MCPManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.tools = []
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        except Exception as e:
            log("mcp_manager_crash", {"error": str(e)})

    def load_all(self) -> list:
        if not os.path.exists(self.config_path):
            return []
        try:
            config = json.load(open(self.config_path))
        except Exception as e:
            log("mcp_load_error", {"error": f"Invalid config: {e}"})
            return []

        futures = []
        for name, cfg in config.items():
            if cfg.get("enabled", False):
                future = asyncio.run_coroutine_threadsafe(
                    self._load_server(name, cfg),
                    self._loop
                )
                futures.append((name, future))

        new_tools = []
        for name, future in futures:
            try:
                tools = future.result(timeout=30)
                print(f"  [MCP] {name}: loaded {len(tools)} tools")
                log("mcp_load", {"server": name, "tools": len(tools)})
                new_tools.extend(tools)
            except Exception as e:
                print(f"  [MCP] {name}: failed — {e}")
                log("mcp_load_error", {"server": name, "error": str(e)})

        with self._lock:
            self.tools.extend(new_tools)
            return list(self.tools)

    def reload(self) -> list:
        with self._lock:
            self.tools.clear()
        return self.load_all()

    async def _load_server(self, name: str, cfg: dict) -> list:
        try:
            if cfg.get("type") in ("sse", "http"):
                return await self._load_sse(name, cfg)
            return await self._load_stdio(name, cfg)
        except Exception as e:
            log("mcp_server_crash", {"server": name, "error": str(e)})
            raise

    async def _load_stdio(self, name: str, cfg: dict) -> list:
        env_vars = cfg.get("env", {})
        merged_env = {**os.environ}
        for k, v in env_vars.items():
            merged_env[k] = os.environ.get(k, str(v))

        # Resolve command to full path so Windows subprocess can find npx/uvx
        command = _resolve_command(cfg["command"])

        # Resolve ${ENV_VAR} placeholders in args (e.g. ${CODI_WORKING_DIR})
        import re
        raw_args = cfg.get("args", [])
        resolved_args = []
        for arg in raw_args:
            def _replacer(match, _env=merged_env):
                return _env.get(match.group(1), match.group(0))
            resolved_args.append(re.sub(r'\$\{(\w+)\}', _replacer, str(arg)))

        params = StdioServerParameters(
            command=command,
            args=resolved_args,
            env=merged_env
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await load_mcp_tools(session)

    async def _load_sse(self, name: str, cfg: dict) -> list:
        url = cfg["url"]
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await load_mcp_tools(session)