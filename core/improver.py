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
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log
from core.prompts import planner_system_prompt, correction_system_prompt
from dispatcher import Dispatcher, wrap_prompt_data
from state.temp_db import RunState, TaskRequirements
from tools.registry import ToolRegistry


# ── Orchestrator system message ───────────────────────────────────────────────
# Lightweight — rules for the orchestrator role only, not tool-calling rules.
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
- If task mentions [BOILERPLATE CREATED: file1, file2], those files exist.
  Plan EDIT steps only — do NOT plan to create them again.
- For simple single-file tasks, ONE step is enough.
- Be specific: name the file, name the tool, name the content.

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


def _deterministic_requirements(task: str) -> TaskRequirements:
    lowered = (task or "").lower()
    framework = None
    for candidate in ("fastapi", "flask", "django", "react"):
        if candidate in lowered:
            framework = candidate
            break
    if not framework and any(term in lowered for term in ("vanilla js", "plain html", "no framework")):
        framework = "vanilla"

    files = _extract_file_refs(task)
    reqs = TaskRequirements(framework=framework, files=files)
    if framework:
        reqs.must_have.append(f"{framework} implementation")
        reqs.must_not.extend(reqs.framework_lock())
    return reqs


def _files_from_tool_results(state: RunState) -> list[str]:
    files = []
    for result in state.tool_results:
        if result.tool not in ("create_file", "write_file", "edit_file"):
            continue
        output = result.output or ""
        if output.strip().startswith("{"):
            try:
                payload = json.loads(output)
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
            
            # Log each tool call
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
        
        # Log final context summary
        log("context_gathered", {
            "files_inspected": len(files_inspected),
            "total_context_chars": total_context_chars,
            "tools_run": len([r for r in result.get("results", []) if r["status"] == "ok"]),
            "files_list": files_inspected[:10],
        })

        return "\n\n".join(context_parts)

    # ── Phase 2: Create plan ──────────────────────────────────────────────────

    def _extract_requirements(self, state: RunState):
        """
        Extract structured requirements without an LLM round trip.
        The planner still sees the full task, so this only anchors obvious
        framework and file constraints for validators/executors.
        """
        state.requirements = _deterministic_requirements(state.user_input)
        log("improver_requirements", {
            "source": "deterministic",
            "requirements": state.requirements.to_dict(),
        })

    def create_plan(self, state: RunState, context: str) -> dict:
        """
        Extract requirements first, then ask the LLM to produce a plan.
        Requirements are injected into the plan prompt so every step
        is grounded against them from the start.
        Falls back gracefully if the LLM outputs plain text instead of JSON.
        """
        from dispatcher import Dispatcher

        # Extract requirements before planning — they anchor every subsequent step
        self._extract_requirements(state)

        prompt = _PLAN_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            tools=", ".join(self.registry.list_names()),
            context=wrap_prompt_data(context[:1200]),
        )

        # Use planner_system_prompt (from prompts.py) for this call —
        # it has tighter JSON-only constraints than the orchestrator system.
        raw = self._call(prompt, system=SystemMessage(content=planner_system_prompt()))
        state.record_llm("improver_plan_raw", raw)

        from context_trimmer import trim_tool_output
        parsed = Dispatcher.parse_llm_json(raw)
        if isinstance(parsed, dict) and "steps" in parsed:
            state.plan = _as_text(parsed.get("plan", ""))
            raw_steps  = parsed.get("steps", [])
            if isinstance(raw_steps, list):
                state.plan_steps = [_as_text(s) for s in raw_steps if _as_text(s)]
            else:
                s = _as_text(raw_steps)
                state.plan_steps = [s] if s else []
            log("plan_created", {
                "plan_source": "json",
                "steps": len(state.plan_steps),
                "plan": trim_tool_output(state.plan, max_tokens=20),
                "step_samples": [trim_tool_output(s, max_tokens=15) for s in state.plan_steps[:3]],
            })
            return parsed

        # Fallback: LLM output wasn't JSON — extract non-empty lines as steps
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        # Strip markdown fences and leading numbers/bullets
        steps = []
        for line in lines:
            line = line.lstrip("```").strip()
            if line.lower().startswith(("json", "{")):
                continue
            line = line.lstrip("0123456789.-) ").strip()
            if line:
                steps.append(line)
        state.plan       = raw[:200]
        state.plan_steps = steps[:5]
        
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
        done_count = max(0, state.iteration - 1)

        if state.plan_steps and "[CORRECTION]" not in (state.plan or ""):
            index = state.iteration - 1
            if 0 <= index < len(state.plan_steps):
                selected_step = state.plan_steps[index]
                log("step_selected", {
                    "step": trim_tool_output(selected_step, max_tokens=20),
                    "matched_plan": True,
                    "iteration": state.iteration,
                    "done": False,
                    "plan_steps_remaining": max(0, len(state.plan_steps) - done_count),
                    "source": "plan_index",
                })
                return {"step": selected_step, "done": False}

        prompt = _NEXT_STEP_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            plan=state.plan,
            plan_steps="\n".join(state.plan_steps) if state.plan_steps else "(no steps)",
            done_steps=f"{done_count} of {len(state.plan_steps)}",
            tool_results=wrap_prompt_data(trim_tool_output(state.context_snapshot(max_recent=3), max_tokens=900)),
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
            # Total parse failure — return raw text as the step, keep going
            selected_step = _as_text(raw)[:300]
        
        # Check if selected step exactly matches a plan step
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
        """
        Called when validation fails.
        Returns a correction string injected into the plan for the next Coder call.

        KEY CHANGE: the correction is adaptive — it names the SPECIFIC failure
        (e.g. "Flask detected") and gives explicit DELETE + REPLACE instructions.
        Vague corrections like "use FastAPI instead" caused the infinite retry loop.
        """
        from dispatcher import Dispatcher
        notes = state.validation_notes or "unknown failure"
        reqs  = state.requirements
        forbidden = reqs.framework_lock()

        # Build an adaptive, specific correction based on what failed
        lines = [
            f"PREVIOUS ATTEMPT FAILED. Reason: {notes}",
            "",
            "REQUIREMENTS (non-negotiable):",
            reqs.as_prompt_block(),
            "",
        ]

        # If it is a framework contamination failure, be brutally explicit
        if forbidden and any(p.lower() in notes.lower() for p in forbidden):
            lines += [
                "CRITICAL: You used a forbidden framework.",
                "You MUST:",
                f"  1. DELETE every line containing: {', '.join(forbidden)}",
                f"  2. Rewrite the ENTIRE file using {reqs.framework} ONLY.",
                "  3. Do NOT import anything from the forbidden frameworks.",
                f"Any output still containing forbidden imports is automatic failure.",
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
            # Fallback: construct correction directly — don't rely on LLM if it's struggling
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

        if files:
            output = "Done. Changed: " + ", ".join(files[:8]) + "."
        elif successes:
            output = f"Done. Completed {successes} tool action(s)."
        else:
            output = "Done."

        if failures:
            output += f" {failures} tool action(s) reported errors."

        log("improver_summary", {"source": "deterministic", "output": output[:200]})
        return output
