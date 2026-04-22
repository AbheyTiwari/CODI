# state/temp_db.py
# ─────────────────────────────────────────────────────────────────────────────
# Centralized state for a single agent run.
#
# Replaces scattered state across agent.py's AgentState TypedDict.
# One object owns everything: plan, tool results, iteration count,
# validation results. Passed through the execution loop by reference.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    tool:   str
    status: str   # "ok" | "error"
    output: str


@dataclass
class RunState:
    # ── Input ─────────────────────────────────────────────────────────────────
    user_input:  str = ""
    history:     str = ""

    # ── Plan ──────────────────────────────────────────────────────────────────
    plan:        str = ""
    plan_steps:  list[str] = field(default_factory=list)

    # ── Execution ─────────────────────────────────────────────────────────────
    iteration:       int = 0
    max_iterations:  int = 8
    tool_results:    list[ToolResult] = field(default_factory=list)

    # ── Validation ────────────────────────────────────────────────────────────
    validation_passed: bool = False
    validation_notes:  str  = ""

    # ── Final output ──────────────────────────────────────────────────────────
    final_output: str = ""
    status:       str = "start"   # start | running | complete | failed

    # ── Raw LLM JSON exchanges (for debugging) ────────────────────────────────
    llm_exchanges: list[dict] = field(default_factory=list)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def add_tool_result(self, tool: str, status: str, output: str):
        self.tool_results.append(ToolResult(tool=tool, status=status, output=output))

    def recent_tool_outputs(self, n: int = 5) -> list[str]:
        return [f"{r.tool}: {r.output}" for r in self.tool_results[-n:]]

    def all_tool_outputs_text(self) -> str:
        return "\n".join(self.recent_tool_outputs(n=len(self.tool_results)))

    def successful_results(self) -> list[ToolResult]:
        return [r for r in self.tool_results if r.status == "ok"]

    def failed_results(self) -> list[ToolResult]:
        return [r for r in self.tool_results if r.status == "error"]

    def record_llm(self, role: str, content: str):
        self.llm_exchanges.append({"role": role, "content": content})

    def is_done(self) -> bool:
        return self.status in ("complete", "failed")

    def exceeds_max(self) -> bool:
        return self.iteration >= self.max_iterations

    def to_summary(self) -> str:
        """Compact string summary for logging or debugging."""
        return json.dumps({
            "input":      self.user_input[:80],
            "iteration":  self.iteration,
            "status":     self.status,
            "tools_run":  len(self.tool_results),
            "plan_steps": len(self.plan_steps),
        }, indent=2)
