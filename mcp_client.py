# mcp_client.py
import asyncio
import json
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

MCP_CONFIG_FILE = "mcp_servers.json"

def load_mcp_config() -> dict:
    if not os.path.exists(MCP_CONFIG_FILE):
        return {}
    with open(MCP_CONFIG_FILE, "r") as f:
        return json.load(f)

async def get_mcp_tools_async(server_name: str, server_config: dict):
    """Connect to a single MCP server and return its tools as LangChain tools."""
    params = StdioServerParameters(
        command=server_config["command"],
        args=server_config.get("args", []),
        env={**os.environ, **server_config.get("env", {})}
    )
    tools = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
    return tools

def get_mcp_tools(server_name: str, server_config: dict):
    """Sync wrapper around the async MCP tool loader."""
    try:
        return asyncio.run(get_mcp_tools_async(server_name, server_config))
    except Exception as e:
        print(f"  [MCP] Failed to load {server_name}: {e}")
        return []

def load_all_mcp_tools() -> list:
    """Load tools from all configured MCP servers."""
    config = load_mcp_config()
    if not config:
        return []
    
    all_tools = []
    for name, cfg in config.items():
        if cfg.get("enabled", True):
            print(f"  [MCP] Connecting to {name}...")
            tools = get_mcp_tools(name, cfg)
            print(f"  [MCP] {name}: loaded {len(tools)} tools")
            all_tools.extend(tools)
    
    return all_tools
