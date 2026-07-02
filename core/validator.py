# core/validator.py
# ─────────────────────────────────────────────────────────────────────────────
# Validates execution results.
# Deterministic checks run first (no LLM cost).
# LLM semantic check only runs if deterministic checks pass.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import subprocess
from langchain_core.messages import HumanMessage

from context_trimmer import trim_tool_output
from dispatcher import Dispatcher
from llm_factory import get_refiner_llm, _FallbackLLM
from logger import log
from state.temp_db import RunState
from core.validation_utils import build_framework_contamination_errors


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


class Validator:
    def __init__(self):
        self.llm = None

    def _get_llm(self):
        if self.llm is None:
            try:
                self.llm = get_refiner_llm()
            except Exception as e:
                log("validator_llm_error", {"error": str(e)[:200]})
                self.llm = _FallbackLLM("refiner llm unavailable")
        return self.llm

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

        # ── noop / done signal from Dispatcher means Executor decided it's done
        last = state.tool_results[-1] if state.tool_results else None
        if last and last.tool == "dispatcher" and last.output in ("noop", "done"):
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
        if state.plan_steps and state.iteration < len(state.plan_steps):
            reason = "Task has remaining plan steps."
            self._fail(state, reason, requires_correction=False)
            log("validation_decision", {
                "layer": "plan_progress",
                "passed": False,
                "reason": reason,
                "iteration": state.iteration,
                "plan_steps": len(state.plan_steps),
            })
            return False

        # ── LLM semantic check ────────────────────────────────────────────────
        return self._llm_check(state)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pass(self, state: RunState, notes: str):
        state.validation_passed = True
        state.validation_notes  = notes
        state.validation_requires_correction = False

    def _fail(self, state: RunState, notes: str, requires_correction: bool = True):
        state.validation_passed = False
        state.validation_notes  = notes
        state.validation_requires_correction = requires_correction

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
        
        prompt = _VALIDATE_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            plan_progress=f"{min(state.iteration, len(state.plan_steps))}/{len(state.plan_steps)}",
            plan_steps="\n".join(state.plan_steps) if state.plan_steps else "(no plan steps)",
            tool_results="\n".join(
                trim_tool_output(o, max_tokens=120)
                for o in state.recent_tool_outputs(5)
            ),
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
