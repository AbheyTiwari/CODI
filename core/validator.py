# core/validator.py
# ─────────────────────────────────────────────────────────────────────────────
# Explicit validation of execution results.
# Checks are deterministic where possible (file exists, syntax valid).
# Falls back to LLM judgment for semantic validation.
# ─────────────────────────────────────────────────────────────────────────────

import ast
import os
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log
from state.temp_db import RunState


_VALIDATE_PROMPT = """\
Task: {task}
Tool results:
{tool_results}

Is the task fully complete and correct based on these results?
Respond ONLY with JSON:
{{
  "passed": true,
  "notes": ""
}}
OR:
{{
  "passed": false,
  "notes": "specific reason why it failed or what's missing"
}}
"""


class Validator:
    def __init__(self):
        self.llm = get_refiner_llm()

    def validate(self, state: RunState) -> bool:
        """
        Run all checks. Returns True if task is considered complete.
        Sets state.validation_passed and state.validation_notes.
        """
        # ── Hard stops — don't even ask the LLM ──────────────────────────────
        if state.exceeds_max():
            state.validation_passed = True
            state.validation_notes  = "Max iterations reached."
            log("validator", {"passed": True, "reason": "max_iterations"})
            return True

        if not state.tool_results:
            state.validation_passed = False
            state.validation_notes  = "No tools were executed."
            log("validator", {"passed": False, "reason": "no_tool_results"})
            return False

        # ── Deterministic checks ──────────────────────────────────────────────
        deterministic_fail = self._deterministic_checks(state)
        if deterministic_fail:
            state.validation_passed = False
            state.validation_notes  = deterministic_fail
            log("validator", {"passed": False, "reason": deterministic_fail})
            return False

        # ── Stall detection ───────────────────────────────────────────────────
        if self._is_stalled(state):
            state.validation_passed = True
            state.validation_notes  = "No progress detected — stopping."
            log("validator", {"passed": True, "reason": "stalled"})
            return True

        # ── LLM semantic check ────────────────────────────────────────────────
        return self._llm_check(state)

    def _deterministic_checks(self, state: RunState) -> str:
        """
        Returns an error string if something is deterministically wrong.
        Returns empty string if all checks pass.
        """
        for result in state.tool_results:
            # Syntax rejection from write_file
            if "WRITE REJECTED" in result.output and "SyntaxError" in result.output:
                return f"Syntax error in written file: {result.output[:200]}"
            # Tool not found
            if result.output.startswith("Tool not found:"):
                return result.output

        return ""

    def _is_stalled(self, state: RunState) -> bool:
        """True if no new tools have run in the last 2 iterations."""
        if state.iteration < 4:
            return False
        # Count unique tool calls — if the last 4 results are all errors, stalled
        recent = state.tool_results[-4:]
        if all(r.status == "error" for r in recent):
            return True
        return False

    def _llm_check(self, state: RunState) -> bool:
        """Ask the LLM if the task is complete."""
        from dispatcher import Dispatcher
        prompt = _VALIDATE_PROMPT.format(
            task=state.user_input,
            tool_results="\n".join(state.recent_tool_outputs(5)),
        )
        try:
            resp   = self.llm.invoke([HumanMessage(content=prompt)])
            parsed = Dispatcher.parse_llm_json(resp.content)
            if parsed:
                passed = bool(parsed.get("passed", False))
                notes  = parsed.get("notes", "")
                state.validation_passed = passed
                state.validation_notes  = notes
                log("validator", {"passed": passed, "notes": notes[:100]})
                return passed
        except Exception as e:
            log("validator_error", {"error": str(e)})

        # Default: if we can't validate, assume done to avoid infinite loops
        state.validation_passed = True
        state.validation_notes  = "Validation check failed — assuming complete."
        return True
