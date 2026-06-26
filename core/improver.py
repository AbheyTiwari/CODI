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
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log
from core.prompts import planner_system_prompt, correction_system_prompt
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
        from dispatcher import Dispatcher
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
        for r in result.get("results", []):
            if r["status"] == "ok":
                context_parts.append(f"[{r['tool']}]\n{r['output'][:600]}")
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return "\n\n".join(context_parts)

    # ── Phase 2: Create plan ──────────────────────────────────────────────────

    def _extract_requirements(self, state: RunState):
        """
        Ask the LLM to extract structured requirements from the user input.
        Populates state.requirements — called once at the start of create_plan.
        Falls back silently if parsing fails (requirements stay empty defaults).
        """
        from dispatcher import Dispatcher
        prompt = _REQUIREMENTS_PROMPT.format(task=state.user_input)
        raw = self._call(prompt, system=SystemMessage(content=planner_system_prompt()))
        parsed = Dispatcher.parse_llm_json(raw)

        if isinstance(parsed, dict):
            state.requirements = TaskRequirements(
                framework = parsed.get("framework") or None,
                must_have = [str(x) for x in parsed.get("must_have", []) if x],
                must_not  = [str(x) for x in parsed.get("must_not",  []) if x],
                files     = [str(x) for x in parsed.get("files",     []) if x],
            )
            log("improver_requirements", {"requirements": state.requirements.to_dict()})
        else:
            log("improver_requirements_fail", {"raw": raw[:200]})

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
            context=context[:1200],
        )

        # Use planner_system_prompt (from prompts.py) for this call —
        # it has tighter JSON-only constraints than the orchestrator system.
        raw = self._call(prompt, system=SystemMessage(content=planner_system_prompt()))
        state.record_llm("improver_plan_raw", raw)

        parsed = Dispatcher.parse_llm_json(raw)
        if isinstance(parsed, dict) and "steps" in parsed:
            state.plan = _as_text(parsed.get("plan", ""))
            raw_steps  = parsed.get("steps", [])
            if isinstance(raw_steps, list):
                state.plan_steps = [_as_text(s) for s in raw_steps if _as_text(s)]
            else:
                s = _as_text(raw_steps)
                state.plan_steps = [s] if s else []
            log("improver_plan", {"steps": len(state.plan_steps), "plan": state.plan[:120]})
            return parsed

        # Fallback: LLM output wasn't JSON — extract non-empty lines as steps
        log("improver_plan_fallback", {"raw": raw[:200]})
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
        return {"plan": state.plan, "steps": state.plan_steps}

    # ── Phase 3: Decide next step ─────────────────────────────────────────────

    def next_step(self, state: RunState) -> dict:
        """Return {"step": str, "done": bool} for the current iteration."""
        from dispatcher import Dispatcher
        done_count = max(0, state.iteration - 1)
        prompt = _NEXT_STEP_PROMPT.format(
            task=state.user_input,
            plan=state.plan,
            plan_steps="\n".join(state.plan_steps) if state.plan_steps else "(no steps)",
            done_steps=f"{done_count} of {len(state.plan_steps)}",
            tool_results="\n".join(state.recent_tool_outputs(5)),
        )
        raw = self._call(prompt)
        state.record_llm("improver_next_step", raw)

        parsed = Dispatcher.parse_llm_json(raw)
        if isinstance(parsed, dict):
            return {
                "step": _as_text(parsed.get("step", "")),
                "done": bool(parsed.get("done", False)),
            }
        # parse returned something non-dict (rare) — treat as a step string
        if parsed:
            return {"step": _as_text(parsed), "done": False}
        # Total parse failure — return raw text as the step, keep going
        return {"step": _as_text(raw)[:300], "done": False}

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
        """Produce the final user-facing output. Only place prose is allowed."""
        if not state.tool_results:
            return "Task completed with no tool executions."

        prompt = _FINAL_PROMPT.format(
            task=state.user_input,
            tool_results="\n".join(state.recent_tool_outputs(8)),
        )
        raw = self._call(prompt)
        state.record_llm("improver_summary", raw)
        log("improver_summary", {"output": raw[:200]})
        return raw or "Done."