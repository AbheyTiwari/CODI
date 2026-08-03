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
import os
from dataclasses import dataclass, field
from typing import Any
from state.knowledge import KnowledgeBase


@dataclass
class ToolResult:
    tool:   str
    status: str   # "ok" | "error"
    output: str


# ── Dependency graph for parallel atomic-step execution ─────────────────────
# This is ADDITIVE to plan_steps (flat list), not a replacement. Simple/edit-
# mode tasks and anything that doesn't populate plan_graph continue through
# the existing sequential Improver.next_step() loop untouched. plan_graph is
# only consulted by the new wave-execution path (agent.py checks
# `if state.plan_graph and state.plan_graph.steps:` before taking that
# branch — see agent.py for the fallback wiring).
#
# depends_on holds OTHER STEP IDS, not file paths — two steps touching the
# same file aren't automatically dependent (e.g. two independent additive
# writes to different sections), and two steps on different files CAN be
# dependent (step B calls a function step A defines in another module).
# Building depends_on is a two-stage process (see core/improver.py):
#   1. LLM proposes depends_on per step at plan time (semantic intent —
#      "this step needs that one to exist first" isn't always visible from
#      source alone, e.g. ordering that matters for product/business logic).
#   2. A static-analysis pass (AST-based where the target file is Python;
#      regex-based name-reference fallback otherwise) adds any edges the
#      LLM missed by checking whether a step's step text or its target
#      file's existing content references a `provides` name from another
#      step. It only ADDS edges, never removes ones the LLM proposed —
#      a false-positive extra dependency costs a little parallelism; a
#      missed one costs correctness, so the bias is deliberate.
@dataclass
class PlanStep:
    """
    One atomic unit of work in a dependency-graph plan — normally scoped to
    "implement one function," "add one class," "wire one route," not a
    whole file. Executed by its own Executor call; validated independently
    before steps that depend on it are allowed to start.

    id           — stable short identifier ("step_1", "step_2", ...), assigned
                   at plan-graph construction time. Used as the depends_on
                   reference — never the step text, which can be rewritten
                   during repair and would break edge lookups.
    text         — the actual instruction handed to the Executor, same shape
                   as an entry in plan_steps.
    target_file  — the file this step is expected to write/modify. Used both
                   for the AST dependency-inference pass and for the
                   dependency-contract validator ("does this file still
                   contain what an upstream step's `provides` promised?").
    depends_on   — list of OTHER PlanStep.id values that must reach status
                   "validated" before this step is eligible to run.
    provides     — symbol names (function/class/route names, etc.) this step
                   is expected to introduce. Populated by the LLM at plan
                   time and cross-checked post-execution. Used by downstream
                   steps' dependency-contract validation, not by this step
                   itself.
    status       — "pending" | "running" | "validated" | "failed".
                   "validated" (not just "done") is the gate other steps'
                   depends_on checks look for — a step that ran but failed
                   validation must not unblock anything downstream.
    attempts     — how many times this step has been dispatched. Kept here
                   (in addition to RunState.step_attempts, which is keyed by
                   raw step text and used by the sequential path) so the
                   wave scheduler can cap per-step retries without relying
                   on text-matching into a dict built for a different loop.
    result       — last execution result dict (same shape Executor.execute
                   already returns), kept for validator/debugging access.
    validation_notes — human-readable reason for the last validation
                   pass/fail, surfaced in repair prompts and logs.
    """
    id: str
    text: str
    target_file: str | None = None
    depends_on: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)
    status: str = "pending"   # pending | running | validated | failed
    attempts: int = 0
    result: dict[str, Any] | None = None
    validation_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "target_file": self.target_file,
            "depends_on": list(self.depends_on),
            "provides": list(self.provides),
            "status": self.status,
            "attempts": self.attempts,
            "validation_notes": self.validation_notes,
        }


@dataclass
class PlanGraph:
    """
    The full dependency graph for a run. Steps are stored keyed by id for
    O(1) depends_on lookups (the wave scheduler resolves a lot of these
    per iteration, so a list-scan here would get expensive on larger plans).
    """
    steps: dict[str, PlanStep] = field(default_factory=dict)
    max_step_attempts: int = 3

    def add(self, step: PlanStep) -> None:
        self.steps[step.id] = step

    def get(self, step_id: str) -> PlanStep | None:
        return self.steps.get(step_id)

    def ready_steps(self) -> list[PlanStep]:
        """
        Steps eligible to run RIGHT NOW: still pending, every dependency is
        validated, and the step hasn't exhausted its retry budget. This is
        exactly one "wave" for the concurrent scheduler — call this, dispatch
        everything it returns in parallel, validate each result, then call
        it again for the next wave. A step whose dependency FAILED (not just
        "not yet validated") is permanently excluded — see blocked_steps().
        """
        ready = []
        for step in self.steps.values():
            if step.status != "pending":
                continue
            if step.attempts >= self.max_step_attempts:
                continue
            deps = [self.steps.get(dep_id) for dep_id in step.depends_on]
            if any(dep is None for dep in deps):
                continue  # dangling dependency reference — treat as not-ready, not crash
            if all(dep.status == "validated" for dep in deps):
                ready.append(step)
        return ready

    def blocked_steps(self) -> list[PlanStep]:
        """Pending steps that can never become ready because a dependency
        permanently failed (exhausted its own retry budget)."""
        blocked = []
        for step in self.steps.values():
            if step.status != "pending":
                continue
            deps = [self.steps.get(dep_id) for dep_id in step.depends_on]
            if any(dep is not None and dep.status == "failed" for dep in deps):
                blocked.append(step)
        return blocked

    def is_complete(self) -> bool:
        """True once every step is either validated or has permanently
        failed/is permanently blocked — i.e. there's no more work the
        scheduler could possibly do, success or not."""
        for step in self.steps.values():
            if step.status in ("validated", "failed"):
                continue
            if step in self.blocked_steps():
                continue
            return False
        return True

    def all_validated(self) -> bool:
        return bool(self.steps) and all(s.status == "validated" for s in self.steps.values())

    def summary_counts(self) -> dict:
        counts = {"pending": 0, "running": 0, "validated": 0, "failed": 0}
        for step in self.steps.values():
            counts[step.status] = counts.get(step.status, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "steps": {sid: s.to_dict() for sid, s in self.steps.items()},
            "summary": self.summary_counts(),
        }


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
    protected_files: list[str]   = field(default_factory=list)

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
            "vanilla": [
                "import react", "import vue", "import angular",
                "from react", "from vue", "from angular",
                "create-react-app", "create react app",
                "npx create-react-app", "npm create vite",
                "vue-cli", "vue create", "ng new",
                "jsx", "tsx",
            ],
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
        if self.protected_files:
            lines.append("READ ONLY: " + ", ".join(self.protected_files))
        return "\n".join(lines) if lines else "(no constraints extracted)"

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "must_have": self.must_have,
            "must_not":  self.must_not,
            "files":     self.files,
            "protected_files": self.protected_files,
        }


@dataclass
class RunState:
    # ── Input ─────────────────────────────────────────────────────────────────
    user_input:  str = ""
    history:     str = ""
    mission: Any = None
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)
    context_confidence: float = 0.0
    # Context is an explicit user-controlled part of a run.  "full" means
    # inspect and read the repository before planning; "targeted" lets the
    # discovery controller choose only task-relevant files.
    context_scope: str = "targeted"
    # When True, the planner unconditionally classifies the task as "read"
    # (read-only: inspect code, answer questions, never write). Set by main.py
    # when the user explicitly types the /read prefix command.
    force_read: bool = False
    context_response: str = ""
    context_attempts: int = 0
    reflections: list[dict[str, Any]] = field(default_factory=list)
    validation_classification: str = ""
    validation_recommendation: str = ""

    # ── Task requirements (populated by Improver.create_plan) ─────────────────
    requirements: TaskRequirements = field(default_factory=TaskRequirements)

    # ── Plan ──────────────────────────────────────────────────────────────────
    plan:        str = ""
    plan_steps:  list[str] = field(default_factory=list)
    # DAG plan for the wave-parallel multi-agent execution path (see PlanGraph
    # above). ADDITIVE — None/empty for every task that goes through the
    # existing sequential Improver.next_step() loop. Only populated when
    # Improver.create_plan() decides a task is atomic-decomposable (see
    # core/improver.py). agent.py checks `if state.plan_graph and
    # state.plan_graph.steps:` before taking the wave-execution branch;
    # anything else falls through to the untouched sequential path.
    plan_graph: PlanGraph | None = None
    # Plan review runs before asking the user for confirmation.  Keeping its
    # decision in state makes retries observable and prevents an unbounded
    # plan-regeneration loop when the validator keeps rejecting a plan.
    plan_validation_attempts: int = 0
    plan_validation_score: float | None = None
    plan_validation_notes: str = ""
    plan_validation_feedback: str = ""

    # ── Execution ─────────────────────────────────────────────────────────────
    iteration:       int = 0
    max_iterations:  int = 8
    tool_results:    list[ToolResult] = field(default_factory=list)
    files_written:   set[str] = field(default_factory=set)
    # Tracks which specific step text has already been content-first-written
    # per normalized path. Keyed separately from files_written because a
    # single file legitimately receives MULTIPLE distinct steps in one plan
    # (e.g. "Write index.html with hero section", then a later step "Write
    # index.html with navigation"). Deduping on path alone silently dropped
    # every step after the first one targeting the same file.
    written_steps:   dict[str, set[str]] = field(default_factory=dict)
    step_attempts:   dict[str, int] = field(default_factory=dict)
    # Per-step repair attempts counter to avoid repeated repair loops
    repair_attempts:  dict[str, int] = field(default_factory=dict)

    # FIX (bug 1 — "keeps trying the same thing that already failed"):
    # step_attempts only ever counted HOW MANY TIMES a step was retried, not
    # WHICH STRATEGY was used on each attempt. That's the actual root cause
    # of Codi repeating a failed approach: next_step()'s repeated-failure
    # branch called improve() with nothing but the raw error string, so a
    # small model given near-identical context twice frequently produces
    # near-identical output twice. This maps step text -> ordered list of
    # distinct strategy names already attempted for that step, so both the
    # executor (deterministically) and the improver's correction prompt
    # (as an explicit instruction) can refuse to repeat a strategy that
    # already failed instead of hoping the LLM notices on its own.
    step_strategies: dict[str, list[str]] = field(default_factory=dict)

    # NEW (gap #2 — "sharper repair prompts"): step_strategies only ever
    # gets populated by core/executor.py's edit-strategy names
    # (edit_first_textmatch, line_range_edit, ...). A test-failure repair
    # loop (validator._test_execution_check -> improve()'s "Test failures"
    # block) never wrote to that dict, so every retry of a failing test saw
    # the exact same prompt shape with zero memory of what the PREVIOUS
    # correction attempt already told the coder to do. This is a parallel,
    # lightweight tracker: step text -> ordered list of the actual
    # correction strings previously sent to the coder for that step,
    # regardless of failure category (test failure, repeated tool error,
    # anything else routed through Improver.improve()). Capped short
    # (see record_repair_attempt) since this only needs to answer "have we
    # already tried this," not serve as a full audit log.
    repair_history: dict[str, list[str]] = field(default_factory=dict)

    project_manifest: dict[str, Any] = field(default_factory=lambda: {"package": None, "files_created": {}})
    plan_confirmed:  bool = True

    # ── LLM backend health ──────────────────────────────────────────────────
    # Incremented by agent.py each time Improver.next_step()/.improve()
    # reports that the underlying LLM call itself failed (connection error,
    # timeout, backend unreachable) rather than the model returning a real
    # (possibly empty) response. Used to distinguish "the planner decided
    # there's nothing left to do" from "the backend never answered" — these
    # were previously conflated, causing runs to silently report success
    # when the configured LLM backend was simply down.
    llm_backend_errors: int = 0

    # ── Validation ────────────────────────────────────────────────────────────
    validation_passed: bool = False
    validation_notes:  str  = ""
    validation_repair_instruction: str = ""
    clarification_prompt: str = ""
    validation_findings: list[dict[str, Any]] = field(default_factory=list)
    validation_requires_correction: bool = True

    # ── Final output ──────────────────────────────────────────────────────────
    final_output: str = ""
    status:       str = "start"   # start | running | awaiting_plan_revision | complete | failed

    # ── Raw LLM JSON exchanges (for debugging) ────────────────────────────────
    llm_exchanges: list[dict] = field(default_factory=list)

    # ── Completed steps (ground truth, not iteration count) ───────────────────
    completed_steps: list[str] = field(default_factory=list)
    current_step: str = ""
    # The ORIGINAL entry from plan_steps that the current execution attempt
    # is trying to satisfy. This is intentionally separate from
    # `current_step` / the "step" text handed to the executor: next_step()
    # can rewrite that text into a validator-repair instruction or a
    # reasoned correction (see core/improver.py next_step()), and if
    # mark_step_complete() were called with that REWRITTEN text instead of
    # the original plan_steps entry, a successful retry would never match
    # anything in plan_steps — the step stays "not completed" forever and
    # the run burns every remaining iteration on a task that actually
    # finished. See agent.py's execution loop for where this is consumed.
    target_plan_step: str = ""

    # ── Helpers ───────────────────────────────────────────────────────────────

    def add_tool_result(self, tool: str, status: str, output: str):
        self.tool_results.append(ToolResult(tool=tool, status=status, output=output))
        self._compress_tool_history_if_needed()

    @staticmethod
    def _normalize_path(path: str | os.PathLike[str]) -> str:
        if path is None:
            return ""
        raw = str(path)
        if os.path.isabs(raw):
            candidate = raw
        else:
            # Resolve relative paths the same way tools/local/file_tools.py
            # does — against CODI_WORKING_DIR, not the process's actual OS
            # cwd. main.py's `cd` command only updates the CODI_WORKING_DIR
            # env var and never calls os.chdir(), so os.path.abspath() here
            # would silently resolve against a stale directory and desync
            # the "already written" bookkeeping from where files are
            # actually written on disk.
            working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
            candidate = os.path.join(working_dir, raw)
        candidate = os.path.abspath(candidate)
        return os.path.normcase(candidate)

    def mark_written(self, path: str | os.PathLike[str]) -> str:
        normalized = self._normalize_path(path)
        if normalized:
            self.files_written.add(normalized)
        return normalized

    def already_written(self, path: str | os.PathLike[str]) -> bool:
        normalized = self._normalize_path(path)
        return bool(normalized and normalized in self.files_written)

    def already_handled_step(self, path: str | os.PathLike[str], step: str) -> bool:
        """
        True only if THIS EXACT step text has already been content-first
        executed for THIS path. This is the check that should gate the
        duplicate-write skip — using already_written() (path-only) there
        was the bug: a plan with several distinct steps for the same file
        ("...with hero section", "...with navigation", "...with features")
        had every step after the first silently no-op'd as a "duplicate",
        so only the first section's content ever made it into the file.
        """
        normalized = self._normalize_path(path)
        if not normalized:
            return False
        return (step or "") in self.written_steps.get(normalized, set())

    def mark_step_handled(self, path: str | os.PathLike[str], step: str) -> str:
        normalized = self._normalize_path(path)
        if normalized:
            self.written_steps.setdefault(normalized, set()).add(step or "")
        return normalized

    # FIX (bug 1): record which strategy was used for a given plan step so
    # repeated attempts don't blindly repeat a strategy that already failed.
    # Keyed on the ORIGINAL step text (same key used by step_attempts /
    # completed_steps), not on the rewritten correction text — otherwise
    # every retry would get a fresh key and this would track nothing.
    def record_step_strategy(self, step: str, strategy: str) -> None:
        if not step or not strategy:
            return
        tried = self.step_strategies.setdefault(step, [])
        if strategy not in tried:
            tried.append(strategy)

    def tried_strategies(self, step: str) -> list[str]:
        return list(self.step_strategies.get(step, []))

    # NEW (gap #2): record the actual correction text sent to the coder for
    # a given step, regardless of failure category. Capped at the 4 most
    # recent entries per step — this only needs to tell the next repair
    # prompt "here's what was already tried," not retain unbounded history.
    def record_repair_attempt(self, step: str, correction: str) -> None:
        if not step or not correction:
            return
        history = self.repair_history.setdefault(step, [])
        if correction not in history:
            history.append(correction)
            if len(history) > 4:
                del history[0]

    def prior_repair_attempts(self, step: str) -> list[str]:
        return list(self.repair_history.get(step, []))

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

    def mark_step_complete(self, step: str):
        if step and step not in self.completed_steps:
            self.completed_steps.append(step)

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