"""LLM-facing exact code-index tools backed by Python's built-in SQLite."""
from __future__ import annotations

import json

from state.code_index import (context_for_query, find_references, find_symbols,
                              graph_neighbors, index_project, resolve_component)


def refresh_code_index(args: dict | None = None) -> str:
    """Index the working project for exact file and symbol lookup."""
    return json.dumps(index_project((args or {}).get("path")), ensure_ascii=False)


def find_symbol(args: dict) -> str:
    """Find exact symbol declarations. Args: name, optional path."""
    name = str((args or {}).get("name", "")).strip()
    if not name:
        return "ERROR: missing symbol name"
    return json.dumps({"success": True, "symbols": find_symbols(name, path=(args or {}).get("path"))}, ensure_ascii=False)


def find_references_exact(args: dict) -> str:
    """Find indexed call/import references. Args: name, optional path, optional limit."""
    name = str((args or {}).get("name", "")).strip()
    if not name:
        return "ERROR: missing symbol name"
    return json.dumps({"success": True, "references": find_references(name, path=(args or {}).get("path"), limit=int((args or {}).get("limit", 200)))}, ensure_ascii=False)


def resolve_component_tool(args: dict) -> str:
    """Resolve a capability (for example 'Upload Handler') to existing symbols and files. Never guesses filenames."""
    capability = str((args or {}).get("capability", "")).strip()
    if not capability:
        return "ERROR: missing capability"
    return json.dumps({"success": True, "capability": capability, "components": resolve_component(capability, limit=int((args or {}).get("limit", 8)))}, ensure_ascii=False)


def retrieve_context(args: dict) -> str:
    """Hybrid retrieval: exact component/symbol candidates plus their import/call graph neighborhood. Args: query."""
    query = str((args or {}).get("query", "")).strip()
    if not query:
        return "ERROR: missing query"
    return json.dumps(context_for_query(query, limit=int((args or {}).get("limit", 12))), ensure_ascii=False)


def find_owner(args: dict) -> str:
    """Identify the module that owns a responsibility. Args: capability."""
    capability = str((args or {}).get("capability", "")).strip()
    if not capability:
        return "ERROR: missing capability"
    context = context_for_query(capability, limit=12)
    return json.dumps({"success": True, "capability": capability, "owners": context["owners"]}, ensure_ascii=False)


def trace_dependencies(args: dict) -> str:
    """Trace indexed imports/calls into or out of a file or symbol. Args: target, optional direction in|out|both."""
    target = str((args or {}).get("target", "")).strip()
    if not target:
        return "ERROR: missing target"
    return json.dumps({"success": True, "edges": graph_neighbors(target, direction=str((args or {}).get("direction", "both")), limit=int((args or {}).get("limit", 100)))}, ensure_ascii=False)


def register_code_index_tools(registry) -> None:
    registry.register_local("refresh_code_index", refresh_code_index)
    registry.register_local("find_symbol", find_symbol)
    registry.register_local("find_references", find_references_exact)
    registry.register_local("resolve_component", resolve_component_tool)
    registry.register_local("retrieve_context", retrieve_context)
    registry.register_local("find_owner", find_owner)
    registry.register_local("trace_dependencies", trace_dependencies)
