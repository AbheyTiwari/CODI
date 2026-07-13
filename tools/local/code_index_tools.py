"""LLM-facing exact code-index tools backed by Python's built-in SQLite."""
from __future__ import annotations

import json

from state.code_index import find_references, find_symbols, index_project


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
    """Find exact identifier references. Args: name, optional path, optional limit."""
    name = str((args or {}).get("name", "")).strip()
    if not name:
        return "ERROR: missing symbol name"
    return json.dumps({"success": True, "references": find_references(name, path=(args or {}).get("path"), limit=int((args or {}).get("limit", 200)))}, ensure_ascii=False)


def register_code_index_tools(registry) -> None:
    registry.register_local("refresh_code_index", refresh_code_index)
    registry.register_local("find_symbol", find_symbol)
    registry.register_local("find_references", find_references_exact)
