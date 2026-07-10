# tools/mcp/mcp_tools.py
# ─────────────────────────────────────────────────────────────────────────────
# Wraps MCP tool objects so they conform to the registry interface:
#   fn(args: dict) -> str
#
# The Dispatcher doesn't know or care that these are MCP under the hood.
#
# IMPORTANT: langchain_mcp_adapters tools returned by load_mcp_tools() are
# async-only (they define a coroutine, not a sync func). Calling .invoke()
# on them raises "StructuredTool does not support sync invocation". They
# must be called with .ainvoke(), scheduled on the SAME event loop the
# owning MCPManager opened the session on (see mcp_manager.py) — calling
# ainvoke() from a random thread with no running loop, or from a different
# loop than the one the session belongs to, will also fail.
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import os
from logger import log

# MCP tools that are loaded in local/air mode (bandwidth-sensitive)
ESSENTIAL_MCP_LOCAL = {
    "read_file", "write_file", "create_directory", "list_directory",
    "git_status", "git_diff", "git_commit", "git_log",
    "fetch", "create_entities", "search_nodes",
}

# Keep the manager alive for the lifetime of the process — tool handlers
# below hold a reference to its event loop and rely on its connections
# staying open. Losing this reference would let the manager (and its loop
# thread) be garbage collected.
_active_manager = None


def _wrap_mcp_tool(tool_obj, loop: asyncio.AbstractEventLoop, timeout: float = 60.0) -> callable:
    """
    Convert a LangChain MCP StructuredTool into a plain callable(args: dict) -> str
    by running its async invocation on the MCP manager's dedicated event loop.
    """
    def handler(args: dict) -> str:
        try:
            future = asyncio.run_coroutine_threadsafe(tool_obj.ainvoke(args), loop)
            result = future.result(timeout=timeout)
            return str(result)
        except asyncio.TimeoutError:
            return f"MCP tool error ({tool_obj.name}): timed out after {timeout}s"
        except Exception as e:
            return f"MCP tool error ({tool_obj.name}): {e}"
    handler.__doc__ = tool_obj.description or ""
    handler.__name__ = tool_obj.name
    return handler


def register_mcp_tools(registry, mode: str = "cloud"):
    """Load all enabled MCP servers and register their tools."""
    global _active_manager

    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(_repo_root, "mcp_servers.json")

    if not os.path.exists(config_path):
        log("mcp_tools_skip", {"reason": "no mcp_servers.json"})
        return

    try:
        from mcp_manager import MCPManager
        manager   = MCPManager(config_path)
        mcp_tools = manager.load_all()
    except Exception as e:
        log("mcp_tools_error", {"error": str(e)})
        print(f"  [MCP] Failed to load: {e}")
        return

    _active_manager = manager  # keep alive — see module docstring

    if mode in ("local", "air"):
        mcp_tools = [t for t in mcp_tools if t.name in ESSENTIAL_MCP_LOCAL]

    for tool_obj in mcp_tools:
        registry.register_mcp(tool_obj.name, _wrap_mcp_tool(tool_obj, manager.loop))

    log("mcp_tools_loaded", {"count": len(mcp_tools), "mode": mode})
    print(f"  [MCP] {len(mcp_tools)} tools registered")


def shutdown_mcp():
    """Call at program exit to cleanly close all MCP connections."""
    global _active_manager
    if _active_manager is not None:
        _active_manager.shutdown()
        _active_manager = None