# tools/registry.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for all tools.
# The Dispatcher calls this. Nothing else.
#
# Tools are plain callables:  fn(args: dict) -> str
# No LangChain binding. No LLM awareness. Dumb executors.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import Callable
from logger import log


class ToolRegistry:
    def __init__(self):
        self._local: dict[str, Callable] = {}
        self._mcp:   dict[str, Callable] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register_local(self, name: str, fn: Callable):
        """Register a local Python function as a tool."""
        self._local[name] = fn
        log("registry_register", {"tool": name, "type": "local"})

    def register_mcp(self, name: str, fn: Callable):
        """Register an MCP-backed callable as a tool."""
        self._mcp[name] = fn
        log("registry_register", {"tool": name, "type": "mcp"})

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Callable | None:
        """Return handler for a tool name. Local takes priority over MCP."""
        return self._local.get(name) or self._mcp.get(name)

    def list_names(self) -> list[str]:
        return sorted(set(list(self._local) + list(self._mcp)))

    def list_all(self) -> list[dict]:
        out = []
        for name, fn in self._local.items():
            out.append({"name": name, "type": "local", "doc": (fn.__doc__ or "").strip()})
        for name, fn in self._mcp.items():
            out.append({"name": name, "type": "mcp",   "doc": (fn.__doc__ or "").strip()})
        return sorted(out, key=lambda x: x["name"])

    def summary(self) -> str:
        lines = []
        for t in self.list_all():
            lines.append(f"  [{t['type']}] {t['name']}: {t['doc'][:80]}")
        return "\n".join(lines) or "  (no tools loaded)"

    # ── Bulk loader ───────────────────────────────────────────────────────────

    def load_all(self, mode: str = "cloud") -> "ToolRegistry":
        """
        Load local tools + MCP tools based on mode.
        Call this once at startup.
        """
        from tools.local.file_tools   import register_file_tools
        from tools.local.shell_tools  import register_shell_tools
        from tools.local.search_tools import register_search_tools
        from tools.mcp.mcp_tools      import register_mcp_tools

        register_file_tools(self)
        register_shell_tools(self)
        register_search_tools(self)
        register_mcp_tools(self, mode=mode)

        log("registry_loaded", {
            "local": len(self._local),
            "mcp":   len(self._mcp),
            "total": len(self.list_names()),
        })
        return self


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this wherever you need the registry after startup.
registry = ToolRegistry()
