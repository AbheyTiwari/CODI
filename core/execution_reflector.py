"""Structured post-tool reflection without private model chain-of-thought."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from state.knowledge import KnowledgeBase

@dataclass
class Reflection:
    continue_execution: bool = True
    needs_context: bool = False
    reason: str = ""
    unknowns: list[str] = field(default_factory=list)

class ExecutionReflector:
    def reflect(self, result: dict[str, Any], knowledge: KnowledgeBase) -> Reflection:
        if result.get("signal") == "need_context":
            reason = str(result.get("reason", "More project context is required."))
            knowledge.add_unknown(reason)
            return Reflection(needs_context=True, reason=reason, unknowns=[reason])
        failures = [item for item in result.get("results", []) if item.get("status") != "ok"]
        if failures:
            reason = str(failures[0].get("output", "tool failed"))[:300]
            knowledge.add_unknown(reason)
            # A failed write/edit is not automatically a context problem. The
            # executor and validation repair loop can use the exact failure to
            # retry the same plan step. Only an explicit need_context signal
            # should interrupt execution for further discovery.
            return Reflection(needs_context=False, reason=reason, unknowns=[reason])
        return Reflection(reason="Tool result recorded in knowledge base.")
