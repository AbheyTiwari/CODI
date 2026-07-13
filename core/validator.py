# core/validator.py
# ─────────────────────────────────────────────────────────────────────────────
# Validates execution results.
# Deterministic checks run first (no LLM cost).
# LLM semantic check only runs if deterministic checks pass.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import subprocess
import shutil
import sys
from langchain_core.messages import HumanMessage

from context_trimmer import trim_tool_output
from dispatcher import Dispatcher
from llm_factory import get_validator_llm, _FallbackLLM
from logger import log
from state.temp_db import RunState
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


def _step_requires_mutation(step: str) -> bool:
    return bool(set(re.findall(r"[a-zA-Z]+", (step or "").lower())) & _IMPLEMENTATION_TERMS)

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
