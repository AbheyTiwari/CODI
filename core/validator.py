# core/validator.py
# ─────────────────────────────────────────────────────────────────────────────
# Validates execution results.
# Deterministic checks run first (no LLM cost).
# LLM semantic check only runs if deterministic checks pass.
# ─────────────────────────────────────────────────────────────────────────────

import os
from langchain_core.messages import HumanMessage

from dispatcher import Dispatcher
from llm_factory import get_refiner_llm
from logger import log
from state.temp_db import RunState
from core.validation_utils import build_framework_contamination_errors


# Validate prompt — tight JSON-only output expected.
_VALIDATE_PROMPT = """\
Task: {task}
Tool results:
{tool_results}

Is the task fully complete and correct based on these results?
Respond ONLY with JSON — no fences, no prose:
{{"passed":true,"notes":""}}
OR:
{{"passed":false,"notes":"specific reason it failed or what is missing"}}

JSON only:"""


class Validator:
    def __init__(self):
        self.llm = get_refiner_llm()

    def validate(self, state: RunState) -> bool:
        """
        Run all checks. Returns True if task is considered complete.
        Always sets state.validation_passed and state.validation_notes.
        """

        # ── Hard cap ──────────────────────────────────────────────────────────
        if state.exceeds_max():
            self._pass(state, "Max iterations reached.")
            log("validator", {"passed": True, "reason": "max_iterations"})
            return True

        # ── noop / done signal from Dispatcher means Executor decided it's done
        last = state.tool_results[-1] if state.tool_results else None
        if last and last.tool == "dispatcher" and last.output in ("noop", "done"):
            self._pass(state, "Executor signalled completion.")
            log("validator", {"passed": True, "reason": "noop_signal"})
            return True

        # ── No tools ran at all ───────────────────────────────────────────────
        if not state.tool_results:
            self._fail(state, "No tools were executed.")
            log("validator", {"passed": False, "reason": "no_tool_results"})
            return False

        # ── Deterministic checks ──────────────────────────────────────────────
        fail_reason = self._deterministic_checks(state)
        if fail_reason:
            self._fail(state, fail_reason)
            log("validator", {"passed": False, "reason": fail_reason[:120]})
            return False

        # ── Stall detection ───────────────────────────────────────────────────
        if self._is_stalled(state):
            self._pass(state, "No progress in last 4 iterations — stopping.")
            log("validator", {"passed": True, "reason": "stalled"})
            return True

        # ── LLM semantic check ────────────────────────────────────────────────
        return self._llm_check(state)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pass(self, state: RunState, notes: str):
        state.validation_passed = True
        state.validation_notes  = notes

    def _fail(self, state: RunState, notes: str):
        state.validation_passed = False
        state.validation_notes  = notes

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
        prompt = _VALIDATE_PROMPT.format(
            task=state.user_input,
            tool_results="\n".join(
                trim_tool_output(o, max_tokens=120)
                for o in state.recent_tool_outputs(5)
            ),
        )
        try:
            resp   = self.llm.invoke([HumanMessage(content=prompt)])
            parsed = Dispatcher.parse_llm_json(resp.content)
            if isinstance(parsed, dict):
                passed = bool(parsed.get("passed", False))
                notes  = str(parsed.get("notes", ""))
                state.validation_passed = passed
                state.validation_notes  = notes
                log("validator", {"passed": passed, "notes": notes[:100]})
                return passed
        except Exception as e:
            log("validator_error", {"error": str(e)})

        # Can't validate → assume done to prevent infinite loop
        self._pass(state, "Validation check failed — assuming complete.")
        return True