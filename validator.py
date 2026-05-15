# core/validator.py
# ─────────────────────────────────────────────────────────────────────────────
# Validates execution results.
#
# ORDER OF CHECKS (cheapest → most expensive):
#   1. Hard cap / noop signal
#   2. File-based checks  — open actual files, run regex + AST (FREE, fast)
#   3. Deterministic tool-result checks
#   4. Stall detection
#   5. LLM semantic check — last resort, qualitative only
#
# The validator NEVER trusts agent summaries or tool output strings alone.
# It reads the actual files that were written.
# ─────────────────────────────────────────────────────────────────────────────

import ast
import os
import re

from dispatcher import Dispatcher
from langchain_core.messages import HumanMessage
from llm_factory import get_refiner_llm
from logger import log
from state.temp_db import RunState


# ── LLM validation prompt — qualitative check only, after all deterministic checks pass ──
_VALIDATE_PROMPT = """\
Task: {task}

Requirements that MUST be satisfied:
{requirements}

Actual file contents written:
{file_contents}

Tool results summary:
{tool_results}

Check EACH requirement explicitly against the actual file contents above.
A file being written is NOT sufficient — the content must match ALL requirements.

Respond ONLY with JSON — no fences, no prose:
{{"passed":true,"notes":""}}
OR:
{{"passed":false,"notes":"list exactly which requirements are missing or wrong"}}

JSON only:"""


class Validator:
    def __init__(self):
        self.llm = get_refiner_llm()

    def validate(self, state: RunState) -> bool:
        """
        Run all checks in order. Returns True only if task is genuinely complete.
        Always sets state.validation_passed and state.validation_notes.
        """

        # ── 1. Hard cap ───────────────────────────────────────────────────────
        if state.exceeds_max():
            self._pass(state, "Max iterations reached.")
            log("validator", {"passed": True, "reason": "max_iterations"})
            return True

        # ── 1b. noop / done signal ────────────────────────────────────────────
        last = state.tool_results[-1] if state.tool_results else None
        if last and last.tool == "dispatcher" and last.output in ("noop", "done"):
            self._pass(state, "Executor signalled completion.")
            log("validator", {"passed": True, "reason": "noop_signal"})
            return True

        # ── 1c. No tools ran ──────────────────────────────────────────────────
        if not state.tool_results:
            self._fail(state, "No tools were executed.")
            log("validator", {"passed": False, "reason": "no_tool_results"})
            return False

        # ── 2. FILE-BASED CHECKS — read actual files, never trust summaries ───
        file_fail = self._file_based_checks(state)
        if file_fail:
            self._fail(state, file_fail)
            log("validator", {"passed": False, "reason": file_fail[:120]})
            return False

        # ── 3. Deterministic tool-result checks ───────────────────────────────
        tool_fail = self._tool_result_checks(state)
        if tool_fail:
            self._fail(state, tool_fail)
            log("validator", {"passed": False, "reason": tool_fail[:120]})
            return False

        # ── 4. Stall detection ────────────────────────────────────────────────
        if self._is_stalled(state):
            self._pass(state, "No progress in last 4 iterations — stopping.")
            log("validator", {"passed": True, "reason": "stalled"})
            return True

        # ── 5. LLM semantic check — qualitative, last resort ─────────────────
        return self._llm_check(state)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pass(self, state: RunState, notes: str):
        state.validation_passed = True
        state.validation_notes  = notes

    def _fail(self, state: RunState, notes: str):
        state.validation_passed = False
        state.validation_notes  = notes

    # ── File-based checks ─────────────────────────────────────────────────────

    def _file_based_checks(self, state: RunState) -> str:
        """
        Find every file written this run. Open and inspect each one.
        Returns an error string if any file fails — empty string if all pass.

        This is the check that catches Flask-in-FastAPI after the file is on disk.
        Does NOT rely on the model's summary of what it wrote.
        """
        written_files = self._get_written_files(state)
        if not written_files:
            return ""   # no files written yet — tool_result_checks handles this

        forbidden_patterns = state.requirements.framework_lock()
        errors = []

        for path in written_files:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                errors.append(f"Could not read {path}: {e}")
                continue

            # ── Hard forbidden pattern check (regex, case-insensitive) ────────
            for pattern in forbidden_patterns:
                if re.search(re.escape(pattern), content, re.IGNORECASE):
                    errors.append(
                        f"FORBIDDEN: '{pattern}' found in {os.path.basename(path)}. "
                        f"This is a {state.requirements.framework} project. "
                        f"Remove ALL {pattern} references and rewrite using "
                        f"{state.requirements.framework} only."
                    )

            # ── must_not string check ─────────────────────────────────────────
            for constraint in state.requirements.must_not:
                # Convert natural language constraint to searchable patterns
                # e.g. "Flask framework components" -> search for "flask"
                search_term = constraint.lower().split()[0]  # first word is usually the tech
                if len(search_term) > 3 and re.search(search_term, content, re.IGNORECASE):
                    errors.append(
                        f"must_not violated: '{constraint}' detected in "
                        f"{os.path.basename(path)}."
                    )

            # ── Python-specific: AST import check ────────────────────────────
            if path.endswith(".py") and content.strip():
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            module = ""
                            if isinstance(node, ast.ImportFrom) and node.module:
                                module = node.module.lower()
                            elif isinstance(node, ast.Import):
                                module = " ".join(a.name.lower() for a in node.names)

                            for pattern in forbidden_patterns:
                                # Extract module name from pattern like "from flask"
                                pat_module = pattern.replace("from ", "").replace("import ", "").strip().lower()
                                if pat_module and pat_module in module:
                                    errors.append(
                                        f"AST check: forbidden import '{module}' in "
                                        f"{os.path.basename(path)}. "
                                        f"Replace with {state.requirements.framework} equivalent."
                                    )
                except SyntaxError:
                    pass  # syntax errors caught elsewhere

        if errors:
            return " | ".join(errors)
        return ""

    def _get_written_files(self, state: RunState) -> list[str]:
        """
        Extract actual file paths from successful write/create/edit tool results.
        Parses the structured JSON output from file_tools.
        """
        import json
        written = []
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())

        for result in state.tool_results:
            if result.tool not in ("write_file", "create_file", "edit_file"):
                continue
            if result.status != "ok":
                continue

            # Try parsing structured JSON output (new format)
            try:
                data = json.loads(result.output)
                if isinstance(data, dict) and data.get("success"):
                    path = data.get("file_modified", "")
                    if path:
                        if not os.path.isabs(path):
                            path = os.path.join(working_dir, path)
                        written.append(path)
                        continue
            except (json.JSONDecodeError, TypeError):
                pass

            # Fallback: parse old-format string "Written N chars to PATH"
            match = re.search(r"(?:Written .+ to|Edited)\s+(.+?)(?:\s*\(|$)", result.output)
            if match:
                path = match.group(1).strip()
                if not os.path.isabs(path):
                    path = os.path.join(working_dir, path)
                written.append(path)

        return list(dict.fromkeys(written))  # deduplicate, preserve order

    # ── Tool result checks ────────────────────────────────────────────────────

    def _tool_result_checks(self, state: RunState) -> str:
        """
        Walk tool results newest-first for clear execution failures.
        Does NOT treat a successful file write as task completion.
        """
        for result in reversed(state.tool_results):
            tool   = result.tool
            status = result.status
            output = result.output

            if tool in ("coder", "dispatcher") and status == "error":
                if "not valid JSON" in output:
                    return output[:200]
                return output[:200]

            if "WRITE REJECTED" in output and "SyntaxError" in output:
                return f"Syntax error in written file: {output[:200]}"
            if "WARNING: SyntaxError" in output:
                return f"Syntax warning: {output[:200]}"

            if tool in ("create_file", "write_file", "edit_file") and output.startswith("ERROR"):
                return output[:200]

            if output.startswith("Tool not found:"):
                return output[:200]

        return ""

    def _is_stalled(self, state: RunState) -> bool:
        """True if the last 4 consecutive results are all errors."""
        if state.iteration < 4:
            return False
        recent = state.tool_results[-4:]
        return all(r.status == "error" for r in recent)

    # ── LLM semantic check ────────────────────────────────────────────────────

    def _llm_check(self, state: RunState) -> bool:
        """
        Qualitative check — runs AFTER all deterministic checks pass.
        Passes actual file contents to the LLM, not just tool output summaries.
        """
        # Read actual file contents for the LLM to inspect
        written_files = self._get_written_files(state)
        file_contents_block = ""
        for path in written_files[:3]:   # cap at 3 files to stay within token budget
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        snippet = f.read(1500)   # first 1500 chars per file
                    file_contents_block += f"\n--- {os.path.basename(path)} ---\n{snippet}\n"
                except Exception:
                    pass

        if not file_contents_block:
            file_contents_block = "(no files found on disk)"

        prompt = _VALIDATE_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            file_contents=file_contents_block,
            tool_results="\n".join(state.recent_tool_outputs(4)),
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

        # Cannot validate → fail safe. Do NOT assume success.
        # Better to retry once more than to declare a broken output done.
        self._fail(state, "Validation check failed — retrying.")
        return False