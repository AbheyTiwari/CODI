# core/improver.py
# ─────────────────────────────────────────────────────────────────────────────
# The Improver is the orchestrating LLM (fast/cheap).
#
# Responsibilities:
#   1. Read context from the codebase
#   2. Build an execution plan
#   3. Decide which step to send to the Coder next
#   4. Generate corrections when validation fails
#   5. Produce the final user-facing summary
#
# All system-level prompt strings live in core/prompts.py.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import ast
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log
from core.prompts import planner_system_prompt, correction_system_prompt
from dispatcher import Dispatcher, wrap_prompt_data
from state.temp_db import RunState, TaskRequirements
from tools.registry import ToolRegistry


# ── Orchestrator system message ───────────────────────────────────────────────
_ORCHESTRATOR_SYSTEM = """\
You are the Improver — the orchestrator of a coding agent called Codi.
You plan, coordinate, and decide. You never call tools yourself.
Always respond in valid JSON (no prose, no markdown fences).
Working directory: {working_dir}"""


# ── Requirements extraction prompt ───────────────────────────────────────────
_REQUIREMENTS_PROMPT = """\
Extract the task requirements from this user request.

User request: {task}

Respond ONLY with JSON — no fences, no prose:
{{
  "framework": "fastapi" or "flask" or "react" or "vanilla" or "django" or null,
  "must_have": ["list of concrete things that must exist in the output"],
  "must_not":  ["list of things explicitly forbidden or that would contaminate the stack"],
  "files":     ["list of files that must be created or modified"]
}}

Rules:
- framework: only set if a specific framework is named or clearly implied
- must_have: be specific — "FastAPI GET / endpoint" not just "endpoint"
- must_not: always include competing frameworks if framework is set
- files: include all files needed to complete the task
- Keep each list item under 60 chars

JSON only:"""


# ── Plan prompt ───────────────────────────────────────────────────────────────
_PLAN_PROMPT = """\
Task: {task}

Requirements:
{requirements}

Available tools: {tools}

Codebase context:
{context}

Produce an execution plan. Respond ONLY with JSON — no fences, no prose:
{{"plan":"one sentence summary","steps":["Step 1: ...","Step 2: ..."]}}

Rules:
- Maximum 5 steps. Each step is a plain string — NOT an object.
- Reconcile every file name with the evidence before using it. If an intended
  file is absent, make the step explicitly create it; do not say "edit".
- Include a verification step only after all implementation steps.
- ONLY reference files that actually appear in the codebase context above.
  Do NOT invent a filename that is a typo or guess (e.g. do not write
  "scripts.js" if the context shows "script.js") — copy the exact filename
  as it appears in the context.
- If task mentions [BOILERPLATE CREATED: file1, file2], those files exist.
  Plan EDIT steps only — do NOT plan to create them again.
- For simple single-file tasks, ONE step is enough.
- Be specific: name the file, name the tool, name the content.
- Every implementation step MUST name at least one target file. Never use a
  vague step such as "Add CSS" or "Implement JavaScript"; say exactly where
  it will be added. A read-only action is not an implementation step.
- Do not plan browser navigation, screenshots, or `file:` URLs for local
  files. Use source reads and the semantic validator unless the user supplied
  a running http:// or https:// application URL.
- For a unit-test request, the named source file is READ ONLY. Plan a new or
  existing test file (for example `test_main.py`) and tests that import the
  source; never write, recreate, or replace the source file.

Example (simple):
{{"plan":"Create a greeting HTML page","steps":["Write hello.html with full HTML greeting Versha and Shubham"]}}

Example (edit existing):
{{"plan":"Add content to boilerplate files","steps":["Edit index.html to add hero section and greeting","Edit styles.css to style the hero section"]}}

JSON only:"""


# ── Next step prompt ──────────────────────────────────────────────────────────
_NEXT_STEP_PROMPT = """\
Task: {task}
Requirements:
{requirements}
Plan: {plan}

All plan steps:
{plan_steps}

Steps completed so far: {done_steps}
Tool results so far:
{tool_results}

Return the next uncompleted step from the plan above.
Copy the step text EXACTLY as written in the plan.
Respond ONLY with JSON — no fences:
{{"step":"exact step text","done":false}}

If ALL steps are done, respond:
{{"step":"","done":true}}

JSON only:"""


# ── Final summary prompt ──────────────────────────────────────────────────────
_FINAL_PROMPT = """\
Task: {task}
Tool results:
{tool_results}

Summarize what was accomplished in 2-3 sentences.
Name the exact files created or changed.
Be direct. No caveats. No suggestions.
Plain text only (not JSON):"""


def _as_text(value) -> str:
    """Safely coerce any value to a non-None string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _extract_file_refs(text: str) -> list[str]:
    pattern = r"(?<![\w.-])([A-Za-z0-9_./\\-]+\.(?:py|java|html|css|js|ts|jsx|tsx|json|md|txt|xml|yml|yaml|sh|sql|svg))"
    return _unique([m.group(1).strip("'\"` ") for m in re.finditer(pattern, text or "")])


_VERIFICATION_WORDS = ("verify", "validate", "test", "check", "screenshot")
_IMPLEMENTATION_WORDS = ("add", "change", "create", "edit", "fix", "implement", "insert", "modify", "remove", "replace", "style", "update", "write")


def _normalize_plan_steps(steps: list[str], requirements: TaskRequirements) -> list[str]:
    """Make every implementation step file-targeted before execution begins."""
    normalized: list[str] = []
    known_files = _unique(requirements.files)
    for raw_step in steps:
        step = _as_text(raw_step).strip()
        if not step:
            continue
        lowered = step.lower()
        # Browser MCP cannot open file:// URLs. Local HTML is verified from
        # source unless the user explicitly supplied a running web URL.
        if any(term in lowered for term in ("browser", "reload the page", "screenshot")):
            target = _extract_file_refs(step) or known_files
            if target:
                step = f"Read {target[0]} and verify its HTML structure and requested content"
                lowered = step.lower()
        is_verification = any(word in lowered for word in _VERIFICATION_WORDS)
        needs_file = any(re.search(rf"\b{word}\b", lowered) for word in _IMPLEMENTATION_WORDS)
        if needs_file and not is_verification and not _extract_file_refs(step):
            if len(known_files) == 1:
                step = f"{step} in {known_files[0]}"
            else:
                # Leave an explicit blocker in the plan rather than letting a
                # weak model silently choose noop for an ungrounded task.
                step = f"Inspect project and identify the target file before: {step}"
        normalized.append(step)
    return normalized


def _deterministic_requirements(task: str) -> TaskRequirements:
    lowered = (task or "").lower()
    framework = None
    # Pick an explicitly requested stack before scanning framework names.
    # A naive substring scan classified "no React" as React and caused the
    # coder to generate exactly the framework the user prohibited.
    if any(term in lowered for term in (
        "vanilla javascript", "vanilla js", "plain javascript", "plain html",
        "html5", "no framework", "no frameworks",
    )):
        framework = "vanilla"
    else:
        for candidate in ("fastapi", "flask", "django", "react"):
            if candidate in lowered and not re.search(rf"\b(no|without)\s+{candidate}\b", lowered):
                framework = candidate
                break

    files = _extract_file_refs(task)
    reqs = TaskRequirements(framework=framework, files=files)
    if framework:
        reqs.must_have.append(f"{framework} implementation")
        reqs.must_not.extend(reqs.framework_lock())
    return reqs


def _is_unit_test_task(task: str) -> bool:
    lowered = (task or "").lower()
    return "unit test" in lowered or "unittest" in lowered or "pytest" in lowered or "test case" in lowered


def _is_broad_every_function_request(task: str) -> bool:
    lowered = (task or "").lower()
    return _is_unit_test_task(task) and any(phrase in lowered for phrase in (
        "each function", "every function", "all functions", "each of functions",
    ))


def _function_by_function_test_steps() -> list[str]:
    """Use Python AST so a small model receives one function-sized task at a time."""
    root = os.environ.get("CODI_WORKING_DIR", os.getcwd())
    steps: list[str] = []
    skipped = {".git", ".venv", "venv", "node_modules", "__pycache__", ".codi"}
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in skipped]
        for filename in sorted(files):
            if not filename.endswith(".py") or filename.startswith("test_"):
                continue
            absolute = os.path.join(directory, filename)
            try:
                with open(absolute, "r", encoding="utf-8", errors="replace") as handle:
                    tree = ast.parse(handle.read())
            except (OSError, SyntaxError):
                continue
            source = os.path.relpath(absolute, root).replace("\\", "/")
            target = f"test_{os.path.splitext(filename)[0]}.py"
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    steps.append(
                        f"Create or update {target} with unit tests only for function {node.name} in {source}"
                    )
    if steps:
        steps.append("Run the focused Python tests and report any failing function")
    return steps


def _test_file_for(source_file: str, project_files: list[str]) -> str:
    stem, extension = os.path.splitext(os.path.basename(source_file))
    filename = f"test_{stem}{extension or '.py'}"
    normalized = [str(path).replace("\\", "/") for path in project_files]
    for directory in ("tests", "test"):
        if any(path.startswith(f"{directory}/") for path in normalized):
            return f"{directory}/{filename}"
    return filename


def _enforce_test_intent(steps: list[str], requirements: TaskRequirements) -> list[str]:
    """Keep test generation additive: source is read-only, test file is the target."""
    if not requirements.protected_files or not requirements.files:
        return steps
    source_file = requirements.protected_files[0]
    test_file = requirements.files[0]
    adjusted: list[str] = []
    test_written = False
    for step in steps:
        lowered = step.lower()
        if any(word in lowered for word in ("run ", "verify", "validate", "pytest", "unittest")):
            adjusted.append(f"Run the focused tests in {test_file} against {source_file}")
            continue
        verb = "Edit" if test_written else "Create"
        adjusted.append(f"{verb} {test_file} with focused unit tests for behavior in {source_file}")
        test_written = True
    return adjusted


def _files_from_tool_results(state: RunState) -> list[str]:
    """
    Extract the list of files that were actually, successfully modified.

    CRITICAL FIX: this previously iterated ALL tool_results regardless of
    status, so a failed edit_file call (e.g. "ERROR editing scripts.js:
    file does not exist") still had its path text-matched by
    _extract_file_refs and reported to the user as "Changed" — even though
    nothing was written. The summary must only ever reflect real,
    successful writes.
    """
    files = []
    for result in state.tool_results:
        if result.tool not in ("create_file", "write_file", "edit_file"):
            continue
        if result.status != "ok":
            continue  # <-- the fix: skip failed attempts entirely

        output = result.output or ""
        if output.strip().startswith("{"):
            try:
                payload = json.loads(output)
                if payload.get("success") is False:
                    continue
                path = payload.get("file_modified") or payload.get("path")
                if path:
                    files.append(str(path))
                    continue
            except json.JSONDecodeError:
                pass
        files.extend(_extract_file_refs(output))
    return _unique(files)


class Improver:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.llm      = get_refiner_llm()

    def _orchestrator_sys(self) -> SystemMessage:
        return SystemMessage(content=_ORCHESTRATOR_SYSTEM.format(
            working_dir=os.environ.get("CODI_WORKING_DIR", os.getcwd())
        ))

    def _call(self, prompt: str, system: SystemMessage | None = None) -> str:
        sys_msg = system or self._orchestrator_sys()
        try:
            resp = self.llm.invoke([sys_msg, HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as e:
            log("improver_error", {"error": str(e)})
            return ""

    # ── Phase 1: Read context ─────────────────────────────────────────────────

    def read_context(self, state: RunState) -> str:
        """
        Gather initial project context by running list_files + search_codebase.
        If boilerplate files were just created (tagged in user_input), read them
        so the Coder LLM can produce valid edit_file old/new pairs.
        """
        import re
        from context_trimmer import trim_tool_output
        dispatcher = Dispatcher(self.registry)

        tools_to_run = [
            {"name": "list_files",      "args": {}},
            {"name": "search_codebase", "args": {"query": state.user_input[:200]}},
        ]

        boilerplate_match = re.search(
            r'\[BOILERPLATE CREATED:\s*([^\]]+)\]', state.user_input
        )
        if boilerplate_match:
            files = [f.strip() for f in boilerplate_match.group(1).split(',')]
            for f in files:
                tools_to_run.append({"name": "read_file", "args": {"path": f}})

        result = dispatcher.dispatch({"action": "tool_call", "tools": tools_to_run})

        context_parts = []
        files_inspected = []
        total_context_chars = 0
        max_files = 5
        max_chars = 4000
        for r in result.get("results", []):
            tool_name = r["tool"]
            status = r["status"]
            output = r["output"]
            output_len = len(output)

            log("context_gathered_tool", {
                "tool": tool_name,
                "status": status,
                "output_len": output_len,
                "output_sample": trim_tool_output(output, max_tokens=10) if status == "ok" else output[:150],
            })

            if status == "ok":
                should_add_context = True
                if tool_name == "read_file":
                    path_value = (r.get("args") or {}).get("path")
                    if len(files_inspected) >= max_files or total_context_chars + len(output) > max_chars:
                        log("context_capped", {"path": path_value or "unknown", "reason": "read_context_limit"})
                        should_add_context = False
                    else:
                        files_inspected.append(path_value)
                elif total_context_chars + len(output) > max_chars:
                    log("context_capped", {"tool": tool_name, "reason": "read_context_limit"})
                    should_add_context = False

                if should_add_context:
                    wrapped_output = wrap_prompt_data(output, path=(r.get("args") or {}).get("path"))
                    context_parts.append(f"[{tool_name}]\n{wrapped_output}")
                    total_context_chars += len(wrapped_output)
            state.add_tool_result(tool_name, status, output)

        log("context_gathered", {
            "files_inspected": len(files_inspected),
            "total_context_chars": total_context_chars,
            "tools_run": len([r for r in result.get("results", []) if r["status"] == "ok"]),
            "files_list": files_inspected[:10],
        })

        return "\n\n".join(context_parts)

    # ── Phase 2: Create plan ──────────────────────────────────────────────────

    def _extract_requirements(self, state: RunState):
        state.requirements = _deterministic_requirements(state.user_input)
        if _is_unit_test_task(state.user_input):
            source_files = [path for path in _extract_file_refs(state.user_input) if path.lower().endswith(".py")]
            if source_files:
                source_file = source_files[0]
                project_files = state.knowledge.project.get("files", []) if state.knowledge else []
                state.requirements.protected_files = [source_file]
                state.requirements.files = [_test_file_for(source_file, project_files)]
        log("improver_requirements", {
            "source": "deterministic",
            "requirements": state.requirements.to_dict(),
        })

    def create_plan(self, state: RunState, context: str) -> dict:
        from dispatcher import Dispatcher

        self._extract_requirements(state)

        if _is_broad_every_function_request(state.user_input):
            steps = _function_by_function_test_steps()
            if steps:
                state.plan = "Generate and verify unit tests one function at a time."
                state.plan_steps = steps
                log("plan_created", {"plan_source": "function_inventory", "steps": len(steps)})
                return {"plan": state.plan, "steps": steps}

        prompt = _PLAN_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            tools=", ".join(self.registry.list_names()),
            context=wrap_prompt_data(context[:6000]),
        )

        raw = self._call(prompt, system=SystemMessage(content=planner_system_prompt()))
        state.record_llm("improver_plan_raw", raw)

        from context_trimmer import trim_tool_output
        parsed = Dispatcher.parse_llm_json(raw)
        if isinstance(parsed, dict) and "steps" in parsed:
            state.plan = _as_text(parsed.get("plan", ""))
            raw_steps  = parsed.get("steps", [])
            if isinstance(raw_steps, list):
                state.plan_steps = _enforce_test_intent(_normalize_plan_steps(
                    [_as_text(s) for s in raw_steps if _as_text(s)], state.requirements
                ), state.requirements)
            else:
                s = _as_text(raw_steps)
                state.plan_steps = _enforce_test_intent(_normalize_plan_steps([s] if s else [], state.requirements), state.requirements)
            log("plan_created", {
                "plan_source": "json",
                "steps": len(state.plan_steps),
                "plan": trim_tool_output(state.plan, max_tokens=20),
                "step_samples": [trim_tool_output(s, max_tokens=15) for s in state.plan_steps[:3]],
            })
            return parsed

        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        steps = []
        for line in lines:
            line = line.lstrip("```").strip()
            if line.lower().startswith(("json", "{")):
                continue
            line = line.lstrip("0123456789.-) ").strip()
            if line:
                steps.append(line)
        state.plan       = raw[:200]
        state.plan_steps = _enforce_test_intent(_normalize_plan_steps(steps[:5], state.requirements), state.requirements)

        log("plan_created", {
            "plan_source": "fallback",
            "steps": len(state.plan_steps),
            "plan": trim_tool_output(state.plan, max_tokens=20),
            "step_samples": [trim_tool_output(s, max_tokens=15) for s in state.plan_steps[:3]],
            "raw_sample": trim_tool_output(raw, max_tokens=30),
        })
        return {"plan": state.plan, "steps": state.plan_steps}

    # ── Phase 3: Decide next step ─────────────────────────────────────────────

    def next_step(self, state: RunState) -> dict:
        """Return {"step": str, "done": bool} for the current iteration."""
        from dispatcher import Dispatcher
        from context_trimmer import trim_tool_output
        done_count = len(state.completed_steps)

        if state.validation_repair_instruction:
            repair = state.validation_repair_instruction
            state.validation_repair_instruction = ""
            log("step_selected", {
                "step": trim_tool_output(repair, max_tokens=30),
                "source": "validator_repair",
                "iteration": state.iteration,
                "done": False,
            })
            return {"step": repair, "done": False}

        if state.plan_steps and "[CORRECTION]" not in (state.plan or ""):
            selected_step = next(
                (step for step in state.plan_steps if step not in state.completed_steps),
                None,
            )
            if selected_step:
                log("step_selected", {
                    "step": trim_tool_output(selected_step, max_tokens=20),
                    "matched_plan": True,
                    "iteration": state.iteration,
                    "done": False,
                    "plan_steps_remaining": max(0, len(state.plan_steps) - done_count),
                    "source": "first_incomplete_plan_step",
                })
                return {"step": selected_step, "done": False}

        prompt = _NEXT_STEP_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            plan=state.plan,
            plan_steps="\n".join(state.plan_steps) if state.plan_steps else "(no steps)",
            done_steps=f"{done_count} of {len(state.plan_steps)}",
            tool_results=wrap_prompt_data("\n".join(state.recent_tool_outputs(5))),
        )
        raw = self._call(prompt)
        state.record_llm("improver_next_step", raw)

        parsed = Dispatcher.parse_llm_json(raw)
        selected_step = None
        matched_plan = False
        done = False

        if isinstance(parsed, dict):
            selected_step = _as_text(parsed.get("step", ""))
            done = bool(parsed.get("done", False))
        elif parsed:
            selected_step = _as_text(parsed)
        else:
            selected_step = _as_text(raw)[:300]

        if selected_step and state.plan_steps:
            matched_plan = any(selected_step.strip() == ps.strip() for ps in state.plan_steps)

        log("step_selected", {
            "step": trim_tool_output(selected_step, max_tokens=20),
            "matched_plan": matched_plan,
            "iteration": state.iteration,
            "done": done,
            "plan_steps_remaining": max(0, len([s for s in state.plan_steps if s]) - done_count),
        })

        return {"step": selected_step, "done": done}

    # ── Phase 4: Generate correction after validation failure ─────────────────

    def improve(self, state: RunState) -> str:
        from dispatcher import Dispatcher
        notes = state.validation_notes or "unknown failure"
        reqs  = state.requirements
        forbidden = reqs.framework_lock()

        lines = [
            f"PREVIOUS ATTEMPT FAILED. Reason: {notes}",
            "",
            "REQUIREMENTS (non-negotiable):",
            reqs.as_prompt_block(),
            "",
        ]

        if forbidden and any(p.lower() in notes.lower() for p in forbidden):
            lines += [
                "CRITICAL: You used a forbidden framework.",
                "You MUST:",
                f"  1. DELETE every line containing: {', '.join(forbidden)}",
                f"  2. Rewrite the ENTIRE file using {reqs.framework} ONLY.",
                "  3. Do NOT import anything from the forbidden frameworks.",
                f"Any output still containing forbidden imports is automatic failure.",
            ]

        if "does not exist" in notes.lower():
            lines += [
                "CRITICAL: The referenced file does not exist. Do NOT keep",
                "retrying the same filename. Either it was typo'd — check the",
                "codebase context for the exact real filename — or it genuinely",
                "needs to be created first with write_file/create_file.",
            ]

        lines += [
            "",
            "Last tool outputs for context:",
            *state.recent_tool_outputs(3),
            "",
            "Describe the SPECIFIC fix in one sentence.",
            'Respond ONLY with JSON: {"correction":"exact fix instruction"}',
            "JSON only:",
        ]

        raw = self._call(
            "\n".join(lines),
            system=SystemMessage(content=correction_system_prompt())
        )
        state.record_llm("improver_improve", raw)

        parsed = Dispatcher.parse_llm_json(raw)
        if isinstance(parsed, dict):
            correction = _as_text(parsed.get("correction", raw))
        elif parsed:
            correction = _as_text(parsed)
        else:
            correction = f"FAILED: {notes}. Fix required: {reqs.as_prompt_block()}"

        log("improver_correction", {"correction": correction[:200]})
        return correction

    # ── Phase 5: Final summary ────────────────────────────────────────────────

    def summarize(self, state: RunState) -> str:
        """Produce the final user-facing output without an LLM round trip."""
        if not state.tool_results:
            return "Task completed with no tool executions."

        files = _files_from_tool_results(state)
        successes = len([r for r in state.tool_results if r.status == "ok"])
        failures = len([r for r in state.tool_results if r.status == "error"])

        stopped_at_limit = state.exceeds_max() and not state.validation_passed
        if stopped_at_limit or state.status == "failed":
            output = "Stopped before completion."
            if stopped_at_limit:
                output += " Reached the maximum number of iterations."
            if state.validation_notes:
                output += f" Last validation issue: {state.validation_notes}"
        elif files:
            output = "Done. Changed: " + ", ".join(files[:8]) + "."
        elif successes:
            output = f"Done. Completed {successes} tool action(s)."
        else:
            output = "Nothing was successfully changed."

        if failures:
            output += f" {failures} tool action(s) reported errors."

        log("improver_summary", {"source": "deterministic", "output": output[:200]})
        return output
