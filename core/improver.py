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
from context_trimmer import trim_tool_output
from llm_factory import get_refiner_llm
from logger import log
from core.prompts import planner_system_prompt, correction_system_prompt
from state.temp_db import RunState
from tools.registry import ToolRegistry


# ── Orchestrator system message ───────────────────────────────────────────────
# Lightweight — rules for the orchestrator role only, not tool-calling rules.
_ORCHESTRATOR_SYSTEM = """\
You are the Improver — the orchestrator of a coding agent called Codi.
You plan, coordinate, and decide. You never call tools yourself.
Always respond in valid JSON (no prose, no markdown fences).
Working directory: {working_dir}"""


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
                context_parts.append(
                    f"[{r['tool']}]{chr(10)}{trim_tool_output(r['output'], max_tokens=120)}"
                )
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return "\n\n".join(context_parts)

    # ── Phase 2: Create plan ──────────────────────────────────────────────────

    def create_plan(self, state: RunState, context: str) -> dict:
        """
        Ask the LLM to produce {"plan": str, "steps": [str]}.
        Falls back gracefully if the LLM outputs plain text instead of JSON.
        """
        from dispatcher import Dispatcher
        prompt = _PLAN_PROMPT.format(
            task=state.user_input,
            tools=", ".join(self.registry.list_names()),
            context=trim_tool_output(context, max_tokens=1000),
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
            tool_results="\n".join(
                trim_tool_output(o, max_tokens=120)
                for o in state.recent_tool_outputs(5)
            ),
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
        Returns a correction string that gets injected into the plan for Coder.
        Uses correction_system_prompt() for tight JSON-only output.
        """
        from dispatcher import Dispatcher
        prompt = (
            f"Task: {state.user_input}\n"
            f"Validation failed: {state.validation_notes}\n"
            f"Last tool outputs:\n"
            + "\n".join(
                trim_tool_output(o, max_tokens=120)
                for o in state.recent_tool_outputs(5)
            )
            + "\n\nDescribe what the Coder should do differently to fix this.\n"
            + 'Respond ONLY with JSON: {"correction":"what to fix and how"}\n'
            + "JSON only:"
        )
        raw = self._call(prompt, system=SystemMessage(content=correction_system_prompt()))
        state.record_llm("improver_improve", raw)

        parsed = Dispatcher.parse_llm_json(raw)
        if isinstance(parsed, dict):
            return _as_text(parsed.get("correction", raw))
        if parsed:
            return _as_text(parsed)
        return _as_text(raw)

    # ── Phase 5: Final summary ────────────────────────────────────────────────

    def summarize(self, state: RunState) -> str:
        """Produce the final user-facing output. Only place prose is allowed."""
        if not state.tool_results:
            return "Task completed with no tool executions."

        prompt = _FINAL_PROMPT.format(
            task=state.user_input,
            tool_results="\n".join(
                trim_tool_output(o, max_tokens=120)
                for o in state.recent_tool_outputs(8)
            ),
        )
        raw = self._call(prompt)
        state.record_llm("improver_summary", raw)
        log("improver_summary", {"output": raw[:200]})
        return raw or "Done."