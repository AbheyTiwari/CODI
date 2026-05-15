# state/temp_db.py
# ─────────────────────────────────────────────────────────────────────────────
# Centralized state for a single agent run.
#
# Replaces scattered state across agent.py's AgentState TypedDict.
# One object owns everything: plan, tool results, iteration count,
# validation results, and task requirements.
# Passed through the execution loop by reference.
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
class TaskRequirements:
    """
    Extracted from the user's request at planning time.
    Every step and every validation check references this — not the raw input.
    This is what prevents objective drift and framework contamination.

    framework   — primary tech stack: "fastapi" | "flask" | "react" | "vanilla" | None
    must_have   — concrete things that MUST exist in the final output
    must_not    — things that are explicitly forbidden (e.g. "no Flask in a FastAPI project")
    files       — files that must be created or modified
    """
    framework:  str | None       = None
    must_have:  list[str]        = field(default_factory=list)
    must_not:   list[str]        = field(default_factory=list)
    files:      list[str]        = field(default_factory=list)

    def framework_lock(self) -> list[str]:
        """
        Return a list of forbidden patterns based on the locked framework.
        Used by the executor to reject contaminating imports before dispatch.
        """
        locks = {
            "fastapi": ["from flask", "import flask", "from django", "import express"],
            "flask":   ["from fastapi", "import fastapi", "from django"],
            "django":  ["from fastapi", "import fastapi", "from flask", "import flask"],
            "react":   ["import vue", "import angular"],
            "vanilla": ["import react", "import vue", "import angular"],
        }
        if self.framework and self.framework.lower() in locks:
            return locks[self.framework.lower()]
        return []

    def as_prompt_block(self) -> str:
        """Compact string injected into every LLM prompt to anchor context."""
        lines = []
        if self.framework:
            lines.append(f"FRAMEWORK: {self.framework} (do not mix with other frameworks)")
        if self.must_have:
            lines.append("MUST HAVE: " + ", ".join(self.must_have))
        if self.must_not:
            lines.append("MUST NOT:  " + ", ".join(self.must_not))
        if self.files:
            lines.append("FILES:     " + ", ".join(self.files))
        return "\n".join(lines) if lines else "(no constraints extracted)"

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "must_have": self.must_have,
            "must_not":  self.must_not,
            "files":     self.files,
        }


@dataclass
class RunState:
    # ── Input ─────────────────────────────────────────────────────────────────
    user_input:  str = ""
    history:     str = ""

    # ── Task requirements (populated by Improver.create_plan) ─────────────────
    requirements: TaskRequirements = field(default_factory=TaskRequirements)

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
            "input":        self.user_input[:80],
            "iteration":    self.iteration,
            "status":       self.status,
            "tools_run":    len(self.tool_results),
            "plan_steps":   len(self.plan_steps),
            "requirements": self.requirements.to_dict(),
        }, indent=2)