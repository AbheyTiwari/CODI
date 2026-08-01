"""Runtime contracts for CODI's built-in local tools.

LLM output is untrusted input.  This module keeps the dispatcher from
passing malformed action payloads into file and shell handlers, where a
generic tool failure would otherwise trigger an unproductive repair loop.
MCP tools intentionally remain permissive because their schemas are owned by
the connected server.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolContract:
    required: tuple[str, ...] = ()


LOCAL_TOOL_CONTRACTS: dict[str, ToolContract] = {
    "create_file": ToolContract(("path",)),
    "write_file": ToolContract(("path",)),
    "read_file": ToolContract(("path",)),
    "read_file_numbered": ToolContract(("path",)),
    "edit_file": ToolContract(("path",)),
    "apply_patch": ToolContract(("path", "patch")),
    "create_directory": ToolContract(("path",)),
    "inspect_file": ToolContract(("path",)),
    "run_command": ToolContract(("command",)),
    "run_command_external": ToolContract(("command",)),
    "search_codebase": ToolContract(("query",)),
    "grep_codebase": ToolContract(("pattern",)),
    "glob_files": ToolContract(("pattern",)),
    "find_symbol": ToolContract(("name",)),
    "find_references": ToolContract(("name",)),
    "resolve_component": ToolContract(("capability",)),
    "retrieve_context": ToolContract(("query",)),
    "find_owner": ToolContract(("capability",)),
    "trace_dependencies": ToolContract(("target",)),
    "verify_static_site": ToolContract(),
    "restore_change_checkpoint": ToolContract(("run_id", "path", "confirm")),
}


def validate_tool_args(tool: str, args: Any) -> str | None:
    """Return a clear, actionable error when a built-in tool call is invalid."""
    contract = LOCAL_TOOL_CONTRACTS.get(tool)
    if contract is None:
        return None
    if not isinstance(args, dict):
        return f"Tool contract error for {tool}: arguments must be an object."

    for field in contract.required:
        value = args.get(field)
        if not isinstance(value, str) or not value.strip():
            nested = args.get("args")
            hint = ""
            if isinstance(nested, dict) and field in nested:
                hint = f" Found '{field}' inside args.args; it must be top-level."
            return (
                f"Tool contract error for {tool}: required argument '{field}' "
                f"must be a non-empty string.{hint}"
            )
    return None
