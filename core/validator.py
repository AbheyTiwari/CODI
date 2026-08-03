# core/validator.py
# ─────────────────────────────────────────────────────────────────────────────
# Validates execution results.
# Deterministic checks run first (no LLM cost).
# LLM semantic check only runs if deterministic checks pass.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import glob
import subprocess
import shutil
import sys
import ast
from langchain_core.messages import HumanMessage

from context_trimmer import trim_tool_output
from dispatcher import Dispatcher
from llm_factory import get_validator_llm, _FallbackLLM
from logger import log
from state.temp_db import RunState, PlanStep, PlanGraph
from core.validation_utils import build_framework_contamination_errors
import re

_STOPWORDS = {
    "a", "an", "the", "to", "with", "for", "and", "or", "of", "in", "on",
    "add", "adds", "adding", "create", "creates", "creating", "edit", "edits",
    "editing", "make", "makes", "making", "update", "updates", "updating",
    "include", "includes", "including", "write", "writes", "writing",
    "file", "using", "so", "that", "it", "this",
}

_IMPLEMENTATION_TERMS = {
    "add", "change", "create", "edit", "fix", "implement", "insert",
    "modify", "remove", "replace", "style", "update", "write",
}


def _extract_step_keywords(step: str) -> list[str]:
    """Pull 1-3 concrete nouns/adjectives out of a step description."""
    words = re.findall(r"[a-zA-Z]{4,}", step.lower())
    keywords = [w for w in words if w not in _STOPWORDS]
    return keywords[-3:] if keywords else []


def _detect_step_target_file(step: str) -> str | None:
    match = re.search(
        r"([A-Za-z0-9_./\\-]+\.(?:html|css|js|ts|jsx|tsx|py|md|json|txt))",
        step,
    )
    return match.group(1).strip("'\"` ") if match else None


def _clean_step_text(step: str) -> str:
    """Strip filenames and path references from the step text to prevent
    false-positive verb matches (e.g., 'change.md' matching 'change', or
    'style.css' matching 'style')."""
    # Remove things like change.md, styles.css
    cleaned = re.sub(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,5}\b", " ", step or "")
    # Strip punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return cleaned


def _step_requires_mutation(step: str) -> bool:
    cleaned = _clean_step_text(step)
    return bool(set(re.findall(r"[a-zA-Z]+", cleaned.lower())) & _IMPLEMENTATION_TERMS)

# Validate prompt — tight JSON-only output expected.
_VALIDATE_PROMPT = """
You are the validator subagent in the dispatcher workflow.
Task: {task}
Requirements:
{requirements}
Plan progress: {plan_progress}
Plan steps:
{plan_steps}
Tool results:
{tool_results}

Verify whether the current step is complete and correct based on the tool outcomes.
If the step passes, report success. If it fails, explain the specific issue and hand it back to the improver for repair.
Respond ONLY with JSON — no fences, no prose:
{{"passed":true,"notes":""}}
OR:
{{"passed":false,"notes":"specific reason it failed or what is missing"}}

JSON only:"""


_VALIDATE_PROMPT = """
You are the validator subagent in the dispatcher workflow.
Task: {task}
Requirements:
{requirements}
Plan progress: {plan_progress}
Plan steps:
{plan_steps}
Tool results:
{tool_results}

Complete source of every file modified during this run:
{changed_sources}

Verify the code against the task, requirements, plan, and complete source. Do not
approve code merely because it parses. If it fails, name the path and line when
possible and give one minimal surgical repair instruction for the coder.
Respond ONLY with JSON:
{{"passed":true,"notes":"","repair_instruction":"","findings":[]}}
OR:
{{"passed":false,"notes":"specific failure","repair_instruction":"one exact surgical action for the coder","findings":[{{"path":"relative/path","line":12,"severity":"error","problem":"what is wrong","repair":"minimal change"}}]}}
"""


# This gate reviews a file immediately after it is written. The final
# task-level review remains useful for cross-file behaviour, but is too late
# to decide whether the executor may advance to the next implementation step.
_FILE_WRITE_VALIDATE_PROMPT = """
You are the file-write validation gate for a coding agent.

Original user request:
{task}

Current plan step:
{step}

Target file (read from disk after the write): {path}
Complete current content of that file:
--- FILE: {path} ---
{source}
--- END FILE: {path} ---

Decide whether the just-written file correctly implements the part of the
ORIGINAL user request assigned to the current plan step. Review the actual
file content, not the tool's success message. Do not approve merely because
the file parses or contains related words. If anything required for this step
is absent, incorrect, or placed in the wrong file, reject it and state exactly
what the coder must change. A rejection prevents the agent from advancing to
the next plan step.

Respond ONLY with JSON:
{{"passed":true,"notes":"","repair_instruction":"","findings":[]}}
OR:
{{"passed":false,"notes":"specific missing or incorrect behavior","repair_instruction":"one exact surgical action for the coder","findings":[{{"path":"relative/path","line":12,"severity":"error","problem":"what is wrong","repair":"minimal change"}}]}}
"""


_PLAN_VALIDATE_PROMPT = """
You are the plan validation gate for a coding agent.

Original user request:
{task}

Plan risk: {risk}
Planner confidence: {confidence:.2f}

The following is the exact current content of plan.md:
--- FILE: plan.md ---
{plan_content}
--- END FILE: plan.md ---

Assess whether this plan fully and safely implements the original request.
Check coverage, sequence, target files, dependencies, and verification. Score
the plan from 0 to 10. Give concrete corrections that the planner can apply;
do not score based only on formatting.

Respond ONLY with JSON:
{{"score":8.5,"notes":"brief assessment","suggested_edits":"specific corrections for the planner"}}
"""

_PLAN_REVIEW_MAX_ATTEMPTS = 3
# A sub-0.90 confidence score is meaningful: CODI should surface the
# uncertainty rather than silently treating it as approval. For a low-risk
# plan, however, it should be shown to the user for an informed decision,
# not regenerated in an identical three-attempt loop.
_HIGH_CONFIDENCE = 0.90


# ── Wave-execution step validation prompt ───────────────────────────────────
# Distinct from _FILE_WRITE_VALIDATE_PROMPT above: that one validates against
# state.current_step (the flat sequential loop's single active step). This
# validates a PlanStep from a PlanGraph (the atomic, wave-parallel path) —
# scoped tighter (one function/unit, not "the current step" broadly) and
# additionally told what symbols the step was supposed to `provides`, so the
# model can check the promised symbol actually exists, not just that
# "something plausible" was written.
_STEP_INTENT_VALIDATE_PROMPT = """
You are the atomic-step validation gate for a multi-agent coding pipeline.
Each step below implements ONE small unit (a function, class, or route) —
review it as a self-contained unit, not the whole file's other content.

Original user request (for overall context only):
{task}

This atomic step's instruction:
{step_text}

This step was expected to introduce these symbols: {provides}

Target file (current on-disk content): {path}
--- FILE: {path} ---
{source}
--- END FILE: {path} ---

Decide whether this step's instruction was correctly and completely carried
out in the file above. Confirm the expected symbols actually exist with
sensible implementations — not just declared as a stub or placeholder unless
the step explicitly asked for a stub. Do not approve based on the file merely
parsing or containing related words. If anything is missing, wrong, or only
partially done, reject and give one exact, minimal, surgical correction.

Respond ONLY with JSON:
{{"passed":true,"notes":"","repair_instruction":"","findings":[]}}
OR:
{{"passed":false,"notes":"specific missing or incorrect behavior","repair_instruction":"one exact surgical action for the coder","findings":[{{"path":"relative/path","line":12,"severity":"error","problem":"what is wrong","repair":"minimal change"}}]}}
"""


class Validator:
    def __init__(self):
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            try:
                self.llm = get_validator_llm()
            except Exception as e:
                log("validator_llm_error", {"error": str(e)[:200]})
                self.llm = _FallbackLLM("refiner llm unavailable")
        return self.llm

    @staticmethod
    def _latest_successful_write(state: RunState, since: int = 0) -> tuple[str, str] | None:
        """Return the path and on-disk content for the most recent file write."""
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        for result in reversed(state.tool_results[since:]):
            if result.tool not in {"create_file", "write_file", "edit_file", "apply_patch"} or result.status != "ok":
                continue
            try:
                payload = json.loads(result.output)
                path = payload.get("file_modified") or payload.get("path")
            except (TypeError, ValueError, json.JSONDecodeError):
                path = None
            if not path:
                continue
            absolute = str(path) if os.path.isabs(str(path)) else os.path.join(working_dir, str(path))
            try:
                with open(absolute, "r", encoding="utf-8", errors="replace") as handle:
                    source = handle.read()
                relative = os.path.relpath(absolute, working_dir).replace("\\", "/")
                return relative, source
            except OSError:
                return str(path), ""
        return None

    def validate_current_write(self, state: RunState, since: int = 0) -> bool | None:
        """Validate the latest write before allowing the plan to advance.

        Returns None when this executor action did not write a file.
        """
        latest = self._latest_successful_write(state, since=since)
        if latest is None:
            return None
        path, source = latest
        if not source:
            reason = f"Could not read written file '{path}' for semantic validation."
            self._fail(state, reason)
            log("validation_decision", {"layer": "per_write_source", "passed": False, "reason": reason})
            return False

        prompt = _FILE_WRITE_VALIDATE_PROMPT.format(
            task=state.user_input,
            step=state.current_step or state.target_plan_step or "(unspecified step)",
            path=path,
            source=source,
        )
        try:
            raw_response = self._get_llm().invoke([HumanMessage(content=prompt)]).content
            parsed = Dispatcher.parse_llm_json(raw_response)
        except Exception as exc:
            reason = f"Per-write LLM validation failed: {exc}"
            self._fail(state, reason)
            log("validation_decision", {"layer": "per_write_llm", "passed": False, "reason": reason[:200]})
            return False

        if not isinstance(parsed, dict):
            reason = "Per-write LLM validation returned invalid JSON."
            self._fail(state, reason)
            log("validation_decision", {"layer": "per_write_llm", "passed": False, "reason": reason})
            return False

        passed = bool(parsed.get("passed", False))
        notes = str(parsed.get("notes", "")).strip()
        repair = str(parsed.get("repair_instruction", "")).strip()
        findings = parsed.get("findings", [])
        state.validation_passed = passed
        state.validation_notes = notes or ("File write validated." if passed else "File write did not meet the request.")
        state.validation_requires_correction = not passed
        state.validation_classification = "success" if passed else "per_write_failure"
        state.validation_recommendation = "continue" if passed else "repair"
        state.validation_repair_instruction = repair if not passed else ""
        state.validation_findings = findings if isinstance(findings, list) else []
        log("validation_decision", {
            "layer": "per_write_llm", "passed": passed, "path": path,
            "notes": trim_tool_output(state.validation_notes, max_tokens=20),
        })
        return passed

    def validate_plan_file(self, state: RunState, plan_path: str, risk: str, confidence: float) -> dict:
        """Score plan.md and apply the pre-confirmation approval policy."""
        try:
            with open(plan_path, "r", encoding="utf-8", errors="replace") as handle:
                plan_content = handle.read()
        except OSError as exc:
            result = {
                "approved": False, "score": 0.0,
                "notes": f"Could not read plan.md for validation: {exc}",
                "suggested_edits": "Regenerate plan.md so it can be reviewed.",
            }
            state.plan_validation_score = result["score"]
            state.plan_validation_notes = result["notes"]
            return result

        prompt = _PLAN_VALIDATE_PROMPT.format(
            task=state.user_input,
            risk=risk or "unknown",
            confidence=float(confidence or 0.0),
            plan_content=plan_content,
        )
        try:
            raw_response = self._get_llm().invoke([HumanMessage(content=prompt)]).content
            parsed = Dispatcher.parse_llm_json(raw_response)
        except Exception as exc:
            parsed = None
            raw_response = ""
            failure = f"Plan validator LLM failed: {exc}"
        else:
            failure = ""

        if not isinstance(parsed, dict):
            result = {
                "approved": False, "score": 0.0,
                "notes": failure or "Plan validator returned invalid JSON.",
                "suggested_edits": "Return a complete plan that directly covers the original request.",
            }
        else:
            raw_score = parsed.get("score", 0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                match = re.search(r"(?:10|[0-9](?:\.\d+)?)", str(raw_score))
                score = float(match.group(0)) if match else 0.0
            score = max(0.0, min(10.0, score))
            notes = str(parsed.get("notes", "")).strip()
            suggested_value = parsed.get(
                "suggested_edits", parsed.get("suggestion", parsed.get("repair_instruction", ""))
            )
            # Some small local models return a structured object despite the
            # JSON schema asking for text. Preserve it as readable JSON so the
            # replanner receives the full feedback instead of Python's lossy
            # representation of a nested dictionary.
            if isinstance(suggested_value, (dict, list)):
                suggested_edits = json.dumps(suggested_value, ensure_ascii=False, indent=2)
            else:
                suggested_edits = str(suggested_value or "").strip()
            normalized_risk = (risk or "").lower()
            evidence_confidence = float(confidence or 0.0)
            requires_user_review = False
            if score >= 9:
                approved = True
                policy = "high_score"
            elif score >= 7 and normalized_risk in {"low", "medium"}:
                approved = evidence_confidence >= _HIGH_CONFIDENCE
                requires_user_review = not approved
                policy = (
                    f"medium_score_{normalized_risk}_risk_high_confidence"
                    if approved else f"medium_score_{normalized_risk}_risk_requires_user_review"
                )
            else:
                approved = False
                policy = "score_or_evidence_requires_revision"
            result = {
                "approved": approved, "score": score, "notes": notes,
                "suggested_edits": suggested_edits or "Improve plan coverage and make each implementation step concrete.",
                "policy": policy,
                "requires_user_review": requires_user_review,
            }

        state.plan_validation_score = result["score"]
        state.plan_validation_notes = result["notes"]
        state.plan_validation_feedback = result["suggested_edits"]
        log("validation_decision", {
            "layer": "plan_llm", "approved": result["approved"],
            "score": result["score"], "risk": risk,
            "confidence": round(float(confidence or 0.0), 2),
            "notes": trim_tool_output(result["notes"], max_tokens=20),
        })
        return result

    # ── Wave-execution step validation (atomic-step gate) ──────────────────
    # Validates ONE PlanStep from the wave-parallel/multi-agent path (see
    # core/executor.py execute_wave / core/temp_db.py PlanGraph). Does NOT
    # touch state.validation_passed / state.validation_notes / any of the
    # other singular RunState validation fields — those belong to the flat
    # sequential loop's single "current step" and would be meaningless (or
    # actively wrong, given multiple steps can be validating concurrently
    # across threads) if shared across several PlanSteps at once. Returns
    # its own self-contained result dict instead — same pattern
    # validate_plan_file() above already uses for exactly this reason.
    #
    # Three checks, cheapest/deterministic first, same ordering discipline
    # as validate() below (deterministic before LLM cost):
    #   1. AST syntax check — in-process ast.parse, Python-only (matches
    #      _python_quality_check's scope above). No subprocess spin-up per
    #      step; wave validation runs once per step across potentially many
    #      concurrent steps; a subprocess per step would multiply badly.
    #   2. Dependency-contract check — for every step this one depends on,
    #      confirm the symbols that step PROMISED (provides) are actually
    #      present via AST in the file they were supposed to land in. This
    #      is what "if function2 depends on function1, check that is
    #      maintained" (the original ask) means concretely: it's not enough
    #      that function1's step reported success, its promised symbol must
    #      still be there NOW, at the moment function2 is being checked —
    #      catches a later step or a merge accidentally clobbering it.
    #   3. Intent-match check (LLM) — only reached if 1 and 2 both pass,
    #      same "don't spend the LLM call on something deterministic
    #      already caught" discipline as the rest of this file.
    #
    # A step fails validation on the FIRST check that fails — no point
    # running the expensive LLM check against code that doesn't even parse.

    @staticmethod
    def _resolve_step_path(path: str) -> str:
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        return path if os.path.isabs(path) else os.path.join(working_dir, path)

    def _step_ast_syntax_check(self, plan_step: PlanStep) -> str:
        """Return an error string if plan_step.target_file is Python and
        fails to parse; "" otherwise (including for non-Python targets or
        a target_file that doesn't exist yet — an existence problem is a
        different failure mode, surfaced by _step_dependency_contract_check
        or the intent-match check instead, not this one)."""
        path = plan_step.target_file
        if not path or not path.lower().endswith(".py"):
            return ""
        absolute = self._resolve_step_path(path)
        try:
            with open(absolute, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except OSError:
            return ""  # file not found is not a syntax failure — different concern
        try:
            ast.parse(source)
        except SyntaxError as e:
            return f"AST syntax check failed for {path}: {e.msg} at line {e.lineno}"
        return ""

    def _step_dependency_contract_check(self, plan_step: PlanStep, graph: PlanGraph) -> str:
        """Return an error string if any upstream dependency's promised
        `provides` symbol is missing from its target file right now. Only
        meaningful for Python target files (AST-based, same scope as the
        syntax check above) — non-Python dependencies are skipped here
        (best-effort scope, matches this module's existing Python-centric
        deterministic checks; the LLM intent-match check is the catch-all
        for everything AST can't inspect)."""
        missing = []
        for dep_id in plan_step.depends_on:
            dep = graph.get(dep_id)
            if dep is None or not dep.provides or not dep.target_file:
                continue
            if not dep.target_file.lower().endswith(".py"):
                continue
            absolute = self._resolve_step_path(dep.target_file)
            try:
                with open(absolute, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
                tree = ast.parse(source)
            except (OSError, SyntaxError):
                # Can't confirm the contract right now — report it as a
                # contract failure rather than silently passing, since the
                # whole point of this check is "downstream must be able to
                # trust upstream actually delivered what it promised."
                missing.extend(
                    f"{sym} (from {dep.id}, in {dep.target_file} — file unreadable or invalid)"
                    for sym in dep.provides
                )
                continue

            defined = {
                node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            }
            for symbol in dep.provides:
                if symbol not in defined:
                    missing.append(f"{symbol} (from {dep.id}, expected in {dep.target_file})")

        if missing:
            return (
                f"Dependency contract violated for step {plan_step.id}: "
                f"the following symbols this step depends on are missing — {', '.join(missing)}."
            )
        return ""

    def _step_intent_match_check(self, plan_step: PlanStep, state: RunState) -> dict:
        """LLM check: did this atomic step's file actually implement what
        the step text asked for. Returns the raw parsed {"passed", "notes",
        "repair_instruction", "findings"} dict (or a synthesized failure
        dict on any error) rather than a bare bool, since callers need the
        repair_instruction/findings detail for the wave scheduler's retry
        prompt, not just pass/fail."""
        path = plan_step.target_file
        absolute = self._resolve_step_path(path) if path else None
        source = ""
        if absolute:
            try:
                with open(absolute, "r", encoding="utf-8", errors="replace") as f:
                    source = f.read()
            except OSError:
                pass

        if not source:
            return {
                "passed": False,
                "notes": f"Could not read target file '{path}' to validate step {plan_step.id}.",
                "repair_instruction": f"Ensure {path} is written before validation.",
                "findings": [],
            }

        prompt = _STEP_INTENT_VALIDATE_PROMPT.format(
            task=state.user_input,
            step_text=plan_step.text,
            provides=", ".join(plan_step.provides) if plan_step.provides else "(none declared)",
            path=path,
            source=source,
        )
        try:
            raw_response = self._get_llm().invoke([HumanMessage(content=prompt)]).content
            parsed = Dispatcher.parse_llm_json(raw_response)
        except Exception as exc:
            return {
                "passed": False,
                "notes": f"Step intent-match LLM call failed: {exc}",
                "repair_instruction": "",
                "findings": [],
            }

        if not isinstance(parsed, dict):
            return {
                "passed": False,
                "notes": "Step intent-match validation returned invalid JSON.",
                "repair_instruction": "",
                "findings": [],
            }

        return {
            "passed": bool(parsed.get("passed", False)),
            "notes": str(parsed.get("notes", "")).strip(),
            "repair_instruction": str(parsed.get("repair_instruction", "")).strip(),
            "findings": parsed.get("findings", []) if isinstance(parsed.get("findings"), list) else [],
        }

    def validate_plan_step(self, plan_step: PlanStep, graph: PlanGraph, state: RunState) -> dict:
        """
        Run the full atomic-step validation gate: AST syntax, then
        dependency-contract, then (only if both pass) the LLM intent-match
        check. This is the method core/executor.py's wave loop (via
        agent.py, piece 6) calls once per "running" PlanStep after a wave
        finishes, to decide whether it can move to "validated" (unblocking
        anything depending on it) or must go back for repair.

        Returns:
            {"passed": bool, "notes": str, "repair_instruction": str,
             "findings": list, "layer": str}
        `layer` identifies which check produced the result (ast_syntax /
        dependency_contract / intent_match) — useful for the wave
        scheduler's retry prompt to know what kind of failure it's dealing
        with, mirroring how _fail()'s classification is used elsewhere in
        this file for the sequential path.
        """
        syntax_error = self._step_ast_syntax_check(plan_step)
        if syntax_error:
            log("validation_decision", {
                "layer": "step_ast_syntax", "passed": False,
                "step_id": plan_step.id, "reason": syntax_error[:200],
            })
            return {
                "passed": False, "notes": syntax_error,
                "repair_instruction": f"Fix the syntax error and rewrite {plan_step.target_file}.",
                "findings": [], "layer": "ast_syntax",
            }

        contract_error = self._step_dependency_contract_check(plan_step, graph)
        if contract_error:
            log("validation_decision", {
                "layer": "step_dependency_contract", "passed": False,
                "step_id": plan_step.id, "reason": contract_error[:200],
            })
            return {
                "passed": False, "notes": contract_error,
                "repair_instruction": (
                    "A dependency this step relies on is missing its promised symbol. "
                    "Do not proceed until the upstream step is corrected."
                ),
                "findings": [], "layer": "dependency_contract",
            }

        intent_result = self._step_intent_match_check(plan_step, state)
        log("validation_decision", {
            "layer": "step_intent_match", "passed": intent_result["passed"],
            "step_id": plan_step.id,
            "notes": trim_tool_output(intent_result["notes"], max_tokens=20),
        })
        intent_result["layer"] = "intent_match"
        return intent_result

    def validate(self, state: RunState) -> bool:
        """
        Run all checks. Returns True if task is considered complete.
        Always sets state.validation_passed and state.validation_notes.
        """
        from context_trimmer import trim_tool_output

        # ── Hard cap ──────────────────────────────────────────────────────────
        if state.exceeds_max():
            self._pass(state, "Max iterations reached.")
            log("validation_decision", {
                "layer": "max_iterations",
                "passed": True,
                "notes": "Max iterations reached.",
            })
            return True

         # ── noop / done signal from Dispatcher means the CURRENT STEP is done.
        # It does NOT mean the whole task is done — that's a separate question,
        # answered by whether plan steps remain (checked below, same as
        # plan_progress). Conflating the two here was causing tasks to end
        # early any time a mid-plan step happened to resolve as a noop.
        last = state.tool_results[-1] if state.tool_results else None
        if last and last.tool == "dispatcher" and last.output in ("noop", "done"):
            current_step = getattr(state, "current_step", "") or ""
            if _step_requires_mutation(current_step):
                reason = "Implementation step returned noop without a verified file modification."
                self._fail(state, reason)
                log("validation_decision", {"layer": "noop_rejected", "passed": False, "reason": reason})
                return False
            keywords = _extract_step_keywords(current_step)
            target_file = _detect_step_target_file(current_step)

            if keywords and target_file:
                resolved = target_file
                if not os.path.isabs(resolved):
                    resolved = os.path.join(
                        os.environ.get("CODI_WORKING_DIR", os.getcwd()), resolved
                    )
                try:
                    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                        file_content = f.read().lower()
                    found = any(kw in file_content for kw in keywords)
                except Exception:
                    found = True  # can't read the file — don't block on I/O failure

                if not found:
                    reason = (
                        f"Step claimed completion via noop but keyword(s) "
                        f"{keywords} not found in {target_file}."
                    )
                    self._fail(state, reason)
                    log("validation_decision", {
                        "layer": "noop_content_check",
                        "passed": False,
                        "reason": reason,
                    })
                    return False
            elif keywords and not target_file:
                log("validation_noop_unverifiable", {
                    "step": current_step[:120],
                    "reason": "no target file detected, cannot verify noop",
                })

            # The noop is legitimate for THIS step. Now check whether the
            # overall plan still has unfinished steps — same ground truth
            # the plan_progress guard uses further down.
            remaining = [s for s in state.plan_steps if s not in state.completed_steps]
            if remaining:
                reason = f"Step completed via noop; {len(remaining)} plan step(s) still remaining."
                self._fail(state, reason, requires_correction=False)
                log("validation_decision", {
                    "layer": "noop_signal",
                    "passed": False,
                    "signal": last.output,
                    "remaining_steps": len(remaining),
                })
                return False

            self._pass(state, "Executor signalled completion.")
            log("validation_decision", {
                "layer": "noop_signal",
                "passed": True,
                "signal": last.output,
            })
            return True

        # ── No tools ran at all ───────────────────────────────────────────────
        if not state.tool_results:
            self._fail(state, "No tools were executed.")
            log("validation_decision", {
                "layer": "no_tools",
                "passed": False,
                "reason": "No tools were executed.",
            })
            return False

        # ── Deterministic checks ──────────────────────────────────────────────
        generation_reason = self._generation_completion_check(state)
        if generation_reason:
            self._fail(state, generation_reason)
            log("validation_decision", {
                "layer": "code_generation_completion",
                "passed": False,
                "reason": trim_tool_output(generation_reason, max_tokens=15),
            })
            return False

        fail_reason = self._deterministic_checks(state)
        if fail_reason:
            self._fail(state, fail_reason)
            log("validation_decision", {
                "layer": "deterministic",
                "passed": False,
                "reason": trim_tool_output(fail_reason, max_tokens=15),
            })
            return False

        java_compile_reason = self._java_compile_check(state)
        if java_compile_reason:
            self._fail(state, java_compile_reason)
            log("validation_decision", {
                "layer": "java_compile",
                "passed": False,
                "reason": trim_tool_output(java_compile_reason, max_tokens=15),
            })
            return False

        python_quality_reason = self._python_quality_check(state)
        if python_quality_reason:
            self._fail(state, python_quality_reason)
            log("validation_decision", {
                "layer": "python_quality",
                "passed": False,
                "reason": trim_tool_output(python_quality_reason, max_tokens=15),
            })
            return False

        # ── Framework contamination checks on generated/modified content ───────────
        contamination_reason = self._framework_contamination_check(state)
        if contamination_reason:
            self._fail(state, contamination_reason)
            log("validation_decision", {
                "layer": "contamination",
                "passed": False,
                "reason": trim_tool_output(contamination_reason, max_tokens=15),
            })
            return False

        # ── Real test execution ─────────────────────────────────────────────
        # Runs after syntax/lint/contamination (cheap, deterministic, already
        # established ordering) and before the LLM semantic check. This is
        # the layer that used to be entirely missing: an LLM saying "looks
        # correct" is not the same thing as pytest actually passing. No-ops
        # cleanly for projects with no pytest / no tests so it never
        # penalizes work that has no test suite to run against.
        test_failure_reason = self._test_execution_check(state)
        if test_failure_reason:
            self._fail(state, test_failure_reason)
            log("validation_decision", {
                "layer": "test_execution",
                "passed": False,
                "reason": trim_tool_output(test_failure_reason, max_tokens=15),
            })
            return False

        static_site_reason = self._static_site_check(state)
        if static_site_reason:
            self._fail(state, static_site_reason)
            log("validation_decision", {
                "layer": "static_site",
                "passed": False,
                "reason": trim_tool_output(static_site_reason, max_tokens=15),
            })
            return False

        # ── Stall detection ───────────────────────────────────────────────────
        if self._is_stalled(state):
            self._pass(state, "No progress in last 4 iterations — stopping.")
            log("validation_decision", {
                "layer": "stall_detection",
                "passed": True,
                "reason": "No progress in last 4 iterations",
            })
            return True

        # ── Plan progress guard ────────────────────────────────────────────────
        # ── Plan progress guard ────────────────────────────────────────────────
        # Only block overall completion if there are steps NOT YET marked
        # complete. A step succeeding does not mean the whole task is done —
        # but it also must not be reported as a failure. Step-level success
        # is already recorded via state.completed_steps by agent.py, BEFORE
        # this function runs. This guard only decides "is everything done."
        remaining = [s for s in state.plan_steps if s not in state.completed_steps]
        if remaining:
            reason = f"{len(remaining)} plan step(s) not yet completed."
            self._fail(state, reason, requires_correction=False)
            log("validation_decision", {
                "layer": "plan_progress",
                "passed": False,
                "reason": reason,
                "iteration": state.iteration,
                "remaining_steps": len(remaining),
            })
            return False

        # ── Structural validation gate (run only when plan steps appear complete)
        structural_reason = self._structural_validation_check(state)
        if structural_reason:
            self._fail(state, structural_reason)
            log("validation_decision", {
                "layer": "structural",
                "passed": False,
                "reason": trim_tool_output(structural_reason, max_tokens=15),
            })
            return False

        # ── LLM semantic check ────────────────────────────────────────────────
        return self._llm_check(state)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pass(self, state: RunState, notes: str):
        state.validation_passed = True
        state.validation_notes  = notes
        state.validation_requires_correction = False
        state.validation_classification = "success"
        state.validation_recommendation = "continue"
        state.validation_repair_instruction = ""
        state.validation_findings = []

    def _fail(self, state: RunState, notes: str, requires_correction: bool = True):
        state.validation_passed = False
        state.validation_notes  = notes
        state.validation_requires_correction = requires_correction
        lowered = notes.lower()
        if any(token in lowered for token in ("no tools", "missing context", "file does not exist", "not found")):
            classification, recommendation = "missing_context", "gather_more_context"
        elif any(token in lowered for token in ("syntax", "compile", "compilation", "importerror")):
            classification, recommendation = "compilation_failure", "repair"
        elif any(token in lowered for token in ("test", "assertion", "pytest")):
            classification, recommendation = "test_failure", "repair"
        elif any(token in lowered for token in ("tool error", "timeout", "permission")):
            classification, recommendation = "tool_failure", "retry"
        elif any(token in lowered for token in ("dependency", "module not found", "package")):
            classification, recommendation = "external_dependency", "gather_more_context"
        else:
            classification, recommendation = "incorrect_assumption", "gather_more_context"
        state.validation_classification = classification
        state.validation_recommendation = recommendation

    @staticmethod
    def _changed_source_context(state: RunState) -> tuple[str, str]:
        """Read complete modified files for the independent semantic review."""
        from config import VALIDATOR_CODE_CONTEXT_TOKENS

        paths = list(state.files_written)
        for result in state.tool_results:
            if result.tool not in {"create_file", "write_file", "edit_file"}:
                continue
            try:
                payload = json.loads(result.output)
                path = payload.get("file_modified") or payload.get("path")
                if path:
                    paths.append(str(path))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        blocks: list[str] = []
        used = 0
        maximum = max(4_000, int(VALIDATOR_CODE_CONTEXT_TOKENS * 3.5))
        omitted: list[str] = []
        for path in dict.fromkeys(paths):
            absolute = path if os.path.isabs(path) else os.path.join(working_dir, path)
            try:
                with open(absolute, "r", encoding="utf-8", errors="replace") as handle:
                    source = handle.read()
                relative = os.path.relpath(absolute, working_dir).replace("\\", "/")
            except OSError:
                continue
            block = f"\n--- FILE: {relative} ---\n{source}\n--- END FILE: {relative} ---\n"
            if used + len(block) > maximum:
                omitted.append(relative)
                continue
            blocks.append(block)
            used += len(block)
        if omitted:
            return "\n".join(blocks), "Validator source budget exceeded; unreviewed modified files: " + ", ".join(omitted)
        return "\n".join(blocks) or "(no modified source files available)", ""

    def _python_quality_check(self, state: RunState) -> str:
        """Compile changed Python files and run Ruff when it is available."""
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        paths: list[str] = []
        for result in state.tool_results:
            if result.tool not in {"create_file", "write_file", "edit_file"} or result.status != "ok":
                continue
            try:
                payload = json.loads(result.output)
                path = payload.get("file_modified") or payload.get("path")
            except (TypeError, ValueError, json.JSONDecodeError):
                path = None
            if path and str(path).lower().endswith(".py"):
                paths.append(str(path))

        ruff = shutil.which("ruff")
        for path in dict.fromkeys(paths):
            absolute = path if os.path.isabs(path) else os.path.join(working_dir, path)
            compile_result = subprocess.run(
                [sys.executable, "-m", "py_compile", absolute],
                cwd=working_dir, capture_output=True, text=True, timeout=20,
            )
            if compile_result.returncode:
                return f"Python syntax check failed for {path}: {(compile_result.stderr or compile_result.stdout).strip()[:500]}"
            if ruff:
                lint_result = subprocess.run(
                    [ruff, "check", "--output-format", "concise", absolute],
                    cwd=working_dir, capture_output=True, text=True, timeout=20,
                )
                if lint_result.returncode:
                    return f"Ruff lint failed for {path}: {(lint_result.stdout or lint_result.stderr).strip()[:500]}"
        return ""

    # ── Real test execution layer ──────────────────────────────────────────
    # Runs BEFORE the LLM semantic check and AFTER syntax/lint/contamination —
    # same "cheap deterministic checks first" ordering the rest of this file
    # already follows. This closes the biggest correctness gap in the
    # pipeline: previously nothing ever actually ran the project's tests,
    # so "validation passed" could mean nothing more than "the LLM said so".
    #
    # Design choices, deliberately conservative:
    #   - Only runs if pytest is importable/available AND the project has at
    #     least one file that looks like a test (test_*.py / *_test.py /
    #     a tests/ directory). Projects with no tests are not penalized —
    #     this layer is a no-op for them, not a hard failure.
    #   - Scoped to files related to what was just touched when possible
    #     (same file/module stem), falling back to the full test suite only
    #     when nothing narrower can be determined. Keeps iteration fast for
    #     small edits in larger projects.
    #   - Timeout-bounded so a hanging test cannot stall the agent loop
    #     indefinitely — a timeout is reported as a failure with a clear
    #     reason, not silently swallowed.
    def _test_execution_check(self, state: RunState) -> str:
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())

        pytest_available = shutil.which("pytest") is not None
        if not pytest_available:
            try:
                import pytest  # noqa: F401
                pytest_available = True
            except ImportError:
                pytest_available = False
        if not pytest_available:
            return ""  # nothing to run — do not fail a project with no test runner

        has_tests = bool(glob.glob(os.path.join(working_dir, "test_*.py"))) \
            or bool(glob.glob(os.path.join(working_dir, "**", "test_*.py"), recursive=True)) \
            or bool(glob.glob(os.path.join(working_dir, "**", "*_test.py"), recursive=True)) \
            or os.path.isdir(os.path.join(working_dir, "tests"))
        if not has_tests:
            return ""  # nothing to validate against — not a failure

        # Narrow to tests related to files this run actually touched, when
        # we can determine that cheaply; otherwise run the whole suite.
        touched_stems = set()
        for result in state.tool_results:
            if result.tool not in ("create_file", "write_file", "edit_file", "apply_patch"):
                continue
            if result.status != "ok":
                continue
            try:
                payload = json.loads(result.output)
                path = payload.get("file_modified") or payload.get("path")
            except (TypeError, ValueError, json.JSONDecodeError):
                path = None
            if path and str(path).lower().endswith(".py"):
                touched_stems.add(os.path.splitext(os.path.basename(str(path)))[0])

        target_args = []
        if touched_stems:
            candidates = []
            for stem in touched_stems:
                candidates += glob.glob(os.path.join(working_dir, f"test_{stem}.py"))
                candidates += glob.glob(os.path.join(working_dir, "**", f"test_{stem}.py"), recursive=True)
                candidates += glob.glob(os.path.join(working_dir, "**", f"{stem}_test.py"), recursive=True)
            candidates = sorted(set(candidates))
            if candidates:
                target_args = candidates

        cmd = [sys.executable, "-m", "pytest", "-q", "--no-header"]
        cmd += target_args if target_args else ["."]

        try:
            proc = subprocess.run(
                cmd, cwd=working_dir, capture_output=True, text=True, timeout=90,
            )
        except subprocess.TimeoutExpired:
            return "Test run timed out after 90s — check for a hang or infinite loop in the changed code."
        except Exception as e:
            log("validator_test_run_error", {"error": str(e)[:200]})
            return ""  # don't fail validation just because pytest itself couldn't launch

        if proc.returncode == 0:
            return ""

        tail = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
        try:
            tail = trim_tool_output(tail, max_tokens=500)
        except Exception:
            pass
        scope = ", ".join(os.path.relpath(p, working_dir) for p in target_args) if target_args else "full suite"
        return f"Test failures ({scope}):\n{tail}"

    def _static_site_check(self, state: RunState) -> str:
        """HTTP-check changed HTML pages and their referenced local assets."""
        html_paths: list[str] = []
        for result in state.tool_results:
            if result.tool not in {"create_file", "write_file", "edit_file", "apply_patch"} or result.status != "ok":
                continue
            try:
                payload = json.loads(result.output)
                path = payload.get("file_modified") or payload.get("path")
            except (TypeError, ValueError, json.JSONDecodeError):
                path = None
            if path and str(path).lower().endswith((".html", ".htm")):
                html_paths.append(str(path))
        if not html_paths:
            return ""

        from tools.local.static_server_tools import verify_static_site

        for path in dict.fromkeys(html_paths):
            try:
                payload = json.loads(verify_static_site({"path": path}))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                return f"Static site validation returned invalid data for {path}: {exc}"
            if not payload.get("success"):
                return f"Static site validation failed for {path}: {payload.get('error', 'unknown error')}"
        return ""

    def _generation_completion_check(self, state: RunState) -> str:
        """Fail fast when content-first generation did not provide its end marker."""
        for result in reversed(state.tool_results):
            if result.tool not in ("create_file", "write_file", "edit_file"):
                continue

            output = result.output or ""
            if "CODI_FILE_WRITE_COMPLETE missing" in output:
                return output[:300]

            if not isinstance(output, str) or not output.strip().startswith("{"):
                continue

            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                continue

            if "code_generation_complete" not in payload:
                continue

            if not payload.get("code_generation_complete"):
                sentinel = payload.get("completion_sentinel") or "completion sentinel"
                return (
                    f"Generated file is missing {sentinel}; ask the coder to regenerate only "
                    "the remaining portion and append it before validating again."
                )

            return ""

        return ""

    def _deterministic_checks(self, state: RunState) -> str:
        """
        Walk tool results newest-first.
        Returns an error string on the first definitive failure.
        Returns "" if the most recent write/create succeeded.
        """
        for result in reversed(state.tool_results):
            tool   = result.tool
            status = result.status
            output = result.output

            # A successful file write is the clearest success signal
            if tool in ("create_file", "write_file", "edit_file") and status == "ok":
                return ""

            # Executor / dispatcher errors — tool never ran
            if tool in ("coder", "dispatcher") and status == "error":
                # If it's just a parse/json fail, not a hard error, let it retry
                if "not valid JSON" in output:
                    return output[:200]
                return output[:200]

            # Syntax rejection from write/edit
            if "WRITE REJECTED" in output and "SyntaxError" in output:
                return f"Syntax error in written file: {output[:200]}"
            if "WARNING: SyntaxError" in output:
                return f"Syntax warning: {output[:200]}"

            # Tool itself returned an error string
            if tool in ("create_file", "write_file", "edit_file") and output.startswith("ERROR"):
                return output[:200]

            # Tool not found — schema mismatch, no point retrying without correction
            if output.startswith("Tool not found:"):
                return output[:200]

        return ""

    def _java_compile_check(self, state: RunState) -> str:
        """Run Maven compile when file tools touched Java sources."""
        java_touched = False
        for result in state.tool_results:
            if result.tool not in ("create_file", "write_file", "edit_file"):
                continue
            output = result.output or ""
            if ".java" in output.lower():
                java_touched = True
                break

        if not java_touched:
            return ""

        for result in reversed(state.tool_results):
            if result.tool not in ("create_file", "write_file", "edit_file") or result.status != "ok":
                continue
            output = result.output or ""
            if not isinstance(output, str):
                continue
            if not output.strip().startswith("{"):
                continue
            try:
                payload = json.loads(output)
            except json.JSONDecodeError:
                continue
            path = payload.get("file_modified") or payload.get("path")
            if not path:
                continue
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
                from tools.local.file_tools import _java_structural_check
                warning = _java_structural_check(path, content)
                if warning:
                    return warning

        working_dir = os.environ.get("CODI_WORKING_DIR")
        if not working_dir or not os.path.exists(os.path.join(working_dir, "pom.xml")):
            return ""

        try:
            proc = subprocess.run(
                ["mvn", "-q", "compile"],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except Exception as e:
            log("validation_warning", {
                "layer": "java_compile",
                "warning": str(e)[:200],
            })
            return ""

        if proc.returncode == 0:
            return ""

        combined = (proc.stdout or "") + (proc.stderr or "")
        tail = combined[-2000:]
        try:
            tail = trim_tool_output(tail, max_tokens=575)
        except Exception:
            tail = tail[-2000:]
        return "JAVA COMPILE FAILED:\n" + tail

    def _framework_contamination_check(self, state: RunState) -> str:
        """Detect forbidden framework content in recent write/edit results."""
        requirements = getattr(state, "requirements", None)
        if not requirements:
            return ""

        for result in reversed(state.tool_results):
            if result.tool not in ("create_file", "write_file", "edit_file"):
                continue

            output = result.output or ""
            file_path = None
            if isinstance(output, str) and output.strip().startswith("{"):
                try:
                    payload = json.loads(output)
                    file_path = payload.get("file_modified") or payload.get("path")
                except json.JSONDecodeError:
                    file_path = None

            if not file_path and isinstance(output, str):
                if "Written" in output:
                    file_path = output.split(" ")[-1].strip()

            if not file_path:
                continue

            if not os.path.isabs(file_path):
                file_path = os.path.join(os.getcwd(), file_path)

            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    file_content = f.read()
            except Exception:
                continue

            errors = build_framework_contamination_errors(file_content, requirements, path=file_path)
            if errors:
                return errors[0]

        return ""

    def _failure_signature(self, output: str) -> str:
        """Return a normalized signature for an error output."""
        text = (output or "").strip().lower()
        if "framework contamination" in text:
            return "framework_contamination"
        if "syntaxerror" in text or "syntax error" in text:
            return "syntax_error"
        if "tool not found" in text:
            return "tool_not_found"
        if "not valid json" in text:
            return "invalid_json"
        if text:
            return text.splitlines()[0][:120]
        return "empty_error"

    def _structural_validation_check(self, state: RunState) -> str:
        """Additional deterministic structural checks for framework-specific projects.

        Only run this gate when the plan steps are effectively complete to avoid
        premature failures during multi-step plans.
        """
        try:
            if not state.requirements:
                return ""
            # Only run structural gate when we've progressed through plan steps
            if state.plan_steps and state.iteration < len(state.plan_steps):
                return ""

            req = state.requirements
            # React-specific checks: require a package manifest and either
            # a react dependency or a jsx/tsx entry file
            if req.framework and req.framework.lower() == "react":
                working_dir = os.environ.get("CODI_WORKING_DIR") or os.getcwd()
                pkg_path = os.path.join(working_dir, "package.json")
                if not os.path.exists(pkg_path):
                    return "React project missing package.json manifest."
                try:
                    with open(pkg_path, "r", encoding="utf-8", errors="replace") as fh:
                        pkg = json.load(fh)
                except Exception:
                    return "Could not read package.json for React project."

                deps = {}
                for k in ("dependencies", "devDependencies", "peerDependencies"):
                    if isinstance(pkg.get(k), dict):
                        deps.update(pkg.get(k))

                has_react_dep = any(n.lower().startswith("react") or n.lower().startswith("react-dom") for n in deps.keys())
                # Also detect common React toolchains via scripts (create-react-app, vite, next)
                scripts = pkg.get("scripts", {}) if isinstance(pkg.get("scripts"), dict) else {}
                script_text = " ".join(scripts.values()) if scripts else ""
                has_react_tooling = any(k in script_text.lower() for k in ("react-scripts", "vite", "next", "create-react-app"))
                # Look for common React entry files
                found_entry = False
                for root, dirs, files in os.walk(working_dir):
                    for f in files:
                        if f.lower().endswith((".jsx", ".tsx", ".js", ".ts")):
                            # shallow heuristic: presence of .jsx/.tsx suggests React code
                            if f.lower().endswith((".jsx", ".tsx")):
                                found_entry = True
                                break
                    if found_entry:
                        break

                if not (has_react_dep or has_react_tooling) and not found_entry:
                    return (
                        "React project appears incomplete: no React dependency in package.json "
                        "and no .jsx/.tsx entry files detected."
                    )

            # For other frameworks, reuse existing contamination checks rather
            # than inventing new heuristics here.
            return ""
        except Exception as e:
            log("structural_validation_error", {"error": str(e)[:200]})
            return ""

    def _is_stalled(self, state: RunState) -> bool:
        """True when the last 4 consecutive results share the same failure signature."""
        if state.iteration < 4:
            return False
        recent = state.tool_results[-4:]
        if len(recent) < 4 or not all(r.status == "error" for r in recent):
            return False
        signatures = {self._failure_signature(r.output) for r in recent}
        return len(signatures) == 1

    def _llm_check(self, state: RunState) -> bool:
        """Ask the LLM if the task is semantically complete."""
        from context_trimmer import trim_tool_output
        
        changed_sources, source_error = self._changed_source_context(state)
        if source_error:
            self._fail(state, source_error, requires_correction=False)
            log("validation_decision", {"layer": "source_context", "passed": False, "reason": source_error})
            return False

        prompt = _VALIDATE_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            plan_progress=f"{min(state.iteration, len(state.plan_steps))}/{len(state.plan_steps)}",
            plan_steps="\n".join(state.plan_steps) if state.plan_steps else "(no plan steps)",
            tool_results="\n".join(
                trim_tool_output(o, max_tokens=120)
                for o in state.recent_tool_outputs(5)
            ),
            changed_sources=changed_sources,
        )
        try:
            resp   = self._get_llm().invoke([HumanMessage(content=prompt)])
            raw_response = resp.content
            parsed = Dispatcher.parse_llm_json(raw_response)
            if isinstance(parsed, dict):
                passed = bool(parsed.get("passed", False))
                notes  = str(parsed.get("notes", ""))
                state.validation_passed = passed
                state.validation_notes  = notes
                repair = str(parsed.get("repair_instruction", "")).strip()
                findings = parsed.get("findings", [])
                state.validation_repair_instruction = repair if not passed else ""
                state.validation_findings = findings if isinstance(findings, list) else []
                
                log("validation_decision", {
                    "layer": "llm_semantic",
                    "passed": passed,
                    "notes": trim_tool_output(notes, max_tokens=20),
                    "prompt": trim_tool_output(prompt, max_tokens=40),
                    "response": trim_tool_output(raw_response, max_tokens=30),
                })
                return passed
        except Exception as e:
            log("validation_decision", {
                "layer": "llm_semantic",
                "error": str(e)[:200],
            })
        return False

        # Can't validate → assume done to prevent infinite loop
        self._pass(state, "Validation check failed — assuming complete.")
        log("validation_decision", {
            "layer": "llm_semantic",
            "fallback": True,
            "reason": "Validation check failed",
        })
        return True
