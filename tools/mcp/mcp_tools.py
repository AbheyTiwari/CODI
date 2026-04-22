# tools/mcp/mcp_tools.py
# ─────────────────────────────────────────────────────────────────────────────
# Wraps MCP tool objects so they conform to the registry interface:
#   fn(args: dict) -> str
#
# The Dispatcher doesn't know or care that these are MCP under the hood.
# ─────────────────────────────────────────────────────────────────────────────

import os
from logger import log

# MCP tools that are loaded in local/air mode (bandwidth-sensitive)
ESSENTIAL_MCP_LOCAL = {
    "read_file", "write_file", "create_directory", "list_directory",
    "git_status", "git_diff", "git_commit", "git_log",
    "fetch", "create_entities", "search_nodes",
}


def _wrap_mcp_tool(tool_obj) -> callable:
    """
    Convert a LangChain MCP tool object into a plain callable(args: dict) -> str.
    """
    def handler(args: dict) -> str:
        try:
            result = tool_obj.invoke(args)
            return str(result)
        except Exception as e:
            return f"MCP tool error ({tool_obj.name}): {e}"
    handler.__doc__ = tool_obj.description or ""
    handler.__name__ = tool_obj.name
    return handler


def register_mcp_tools(registry, mode: str = "cloud"):
    """Load all enabled MCP servers and register their tools."""
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

    if mode in ("local", "air"):
        mcp_tools = [t for t in mcp_tools if t.name in ESSENTIAL_MCP_LOCAL]

    for tool_obj in mcp_tools:
        registry.register_mcp(tool_obj.name, _wrap_mcp_tool(tool_obj))

    log("mcp_tools_loaded", {"count": len(mcp_tools), "mode": mode})
    print(f"  [MCP] {len(mcp_tools)} tools registered")
