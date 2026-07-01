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
        self._compress_tool_history_if_needed()

    def _compress_tool_history_if_needed(self, threshold: int = 10):
        """Collapse older tool results into a summary once history becomes too long."""
        if len(self.tool_results) <= threshold:
            return

        keep = max(3, threshold // 2)
        older = self.tool_results[:-keep]
        recent = self.tool_results[-keep:]
        summary_lines = ["[SUMMARY] Earlier tool activity:"]
        tool_counts: dict[str, int] = {}
        for result in older:
            tool_counts[result.tool] = tool_counts.get(result.tool, 0) + 1

        summary_lines.extend(f"- {tool}: {count} call(s)" for tool, count in sorted(tool_counts.items()))
        summary_lines.append("[RECENT]")
        summary_lines.extend(f"{r.tool}: {r.output}" for r in recent)

        compressed = ToolResult(
            tool="context_summary",
            status="ok",
            output="\n".join(summary_lines),
        )
        self.tool_results = [compressed, *recent]

    def recent_tool_outputs(self, n: int = 5) -> list[str]:
        return [f"{r.tool}: {r.output}" for r in self.tool_results[-n:]]

    def context_snapshot(self, max_recent: int = 3) -> str:
        """Return a compact summary of tool history for prompt construction."""
        if not self.tool_results:
            return "(no tool activity yet)"

        recent = self.tool_results[-max_recent:]
        recent_lines = [f"{r.tool}: {r.output}" for r in recent]

        older = self.tool_results[:-max_recent]
        if not older:
            return "\n".join(recent_lines)

        tool_counts: dict[str, int] = {}
        files_touched: set[str] = set()
        for result in older:
            tool_counts[result.tool] = tool_counts.get(result.tool, 0) + 1
            if result.output and result.tool in {"create_file", "write_file", "edit_file"}:
                lowered = result.output.lower()
                if "file_modified" in lowered:
                    import re
                    matches = re.findall(r'([A-Za-z0-9_./\\-]+\.(?:py|js|ts|html|css|json|md|txt|svg|sh))', result.output)
                    files_touched.update(matches)

        summary_lines = [
            "[SUMMARY] Earlier tool activity:",
            *[f"- {tool}: {count} call(s)" for tool, count in sorted(tool_counts.items())],
        ]
        if files_touched:
            summary_lines.append(f"- files touched: {', '.join(sorted(files_touched)[:5])}")
        summary_lines.extend(["[RECENT]", *recent_lines])
        return "\n".join(summary_lines)

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

    def to_decision_trace(self) -> dict:
        """
        Return a comprehensive decision trace for observability.
        Includes: total LLM calls, tool calls, iterations, validation layers per iteration,
        files touched, plan adherence, framework lock status.
        """
        # Count LLM calls by role
        llm_calls_by_role = {}
        for exchange in self.llm_exchanges:
            role = exchange.get("role", "unknown")
            llm_calls_by_role[role] = llm_calls_by_role.get(role, 0) + 1
        
        # Extract touched files from tool results
        files_touched = set()
        for result in self.tool_results:
            if result.tool in ("create_file", "write_file", "edit_file") and result.status == "ok":
                # Try to extract file path from output
                if "file_modified" in result.output or "Written" in result.output:
                    # Rough heuristic: file paths often appear after these keywords
                    for line in result.output.split("\n"):
                        if "file_modified" in line or "Written" in line:
                            # Extract path-like strings
                            import re
                            matches = re.findall(r'([A-Za-z0-9_./\-]+\.(?:py|js|ts|html|css|json|md))', line)
                            files_touched.update(matches)
        
        # Extract validation layers from llm_exchanges (they include validation_decision logs)
        validation_layers = {}
        for result in self.tool_results:
            # Count by tool type as a proxy for validation stages
            tool = result.tool
            if tool not in validation_layers:
                validation_layers[tool] = {"ok": 0, "error": 0}
            validation_layers[tool][result.status] += 1
        
        return {
            "total_iterations": self.iteration,
            "max_iterations": self.max_iterations,
            "total_tool_calls": len(self.tool_results),
            "successful_tools": len([r for r in self.tool_results if r.status == "ok"]),
            "failed_tools": len([r for r in self.tool_results if r.status == "error"]),
            "total_llm_calls": len(self.llm_exchanges),
            "llm_calls_by_role": llm_calls_by_role,
            "plan_steps_count": len(self.plan_steps),
            "files_touched": list(files_touched)[:20],  # Limit to first 20
            "files_count": len(files_touched),
            "validation_passed": self.validation_passed,
            "framework_locked": self.requirements.framework or None,
            "status": self.status,
        }