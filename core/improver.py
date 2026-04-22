# core/improver.py
# ─────────────────────────────────────────────────────────────────────────────
# The Improver is the orchestrating LLM (fast/cheap).
#
# Responsibilities:
#   1. Read context and build a plan
#   2. Decide which step to send to the Coder next
#   3. Receive execution results and decide: next step OR done
#   4. Fix/improve output when validation fails
#   5. Produce the final summary for the user
#
# It never calls tools directly. It outputs JSON or plain text.
# ─────────────────────────────────────────────────────────────────────────────

import os
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log
from state.temp_db import RunState
from tools.registry import ToolRegistry


_SYSTEM = """\
You are the Improver — the orchestrator of a coding agent called Codi.
You plan, coordinate, and decide. You never execute tools yourself.

RULES:
- Always respond in valid JSON (no prose, no markdown).
- Be concise. Do not over-explain.
- Use the tool results you are given to make decisions.
- Working directory: {working_dir}
"""

_PLAN_PROMPT = """\
Task: {task}

Available tools:
{tools}

Codebase context:
{context}

Produce a step-by-step execution plan.
Respond ONLY with JSON in this exact format:
{{
  "plan": "one-sentence summary of overall approach",
  "steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ]
}}
Maximum 5 steps. Be concrete. Name the actual tools you will use.
"""

_NEXT_STEP_PROMPT = """\
Task: {task}
Plan: {plan}
Completed steps so far: {done_steps}
Tool results so far:
{tool_results}

What is the next step to execute?
Respond ONLY with JSON:
{{
  "step": "description of what to do next",
  "done": false
}}

If the task is fully complete based on the tool results, respond:
{{
  "step": "",
  "done": true
}}
"""

_IMPROVE_PROMPT = """\
Task: {task}
Validation failed with this note: {validation_notes}
Last tool outputs:
{tool_results}

Describe what the Coder should do differently to fix this.
Respond ONLY with JSON:
{{
  "correction": "what to fix and how"
}}
"""

_FINAL_PROMPT = """\
Task: {task}
Tool results:
{tool_results}

Summarize what was done in 2-3 sentences. Be direct and specific.
Name the files created or changed. Do not add caveats or suggestions.
Respond with plain text (not JSON). This is the final user-facing output.
"""


class Improver:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.llm      = get_refiner_llm()

    def _sys(self) -> SystemMessage:
        return SystemMessage(content=_SYSTEM.format(
            working_dir=os.environ.get("CODI_WORKING_DIR", os.getcwd())
        ))

    def _call(self, prompt: str) -> str:
        try:
            resp = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as e:
            log("improver_error", {"error": str(e)})
            return ""

    # ── Phase 1: Build context from codebase ──────────────────────────────────

    def read_context(self, state: RunState) -> str:
        """
        Use read tools to gather initial context about the project.
        Returns a text summary of what was found.
        """
        from dispatcher import Dispatcher
        dispatcher = Dispatcher(self.registry)

        result = dispatcher.dispatch({
            "action": "tool_call",
            "tools": [
                {"name": "list_files",       "args": {}},
                {"name": "search_codebase",  "args": {"query": state.user_input}},
            ]
        })

        context_parts = []
        for r in result.get("results", []):
            if r["status"] == "ok":
                context_parts.append(f"[{r['tool']}]\n{r['output'][:600]}")
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return "\n\n".join(context_parts)

    # ── Phase 2: Create plan ──────────────────────────────────────────────────

    def create_plan(self, state: RunState, context: str) -> dict:
        """
        Ask the LLM to produce a JSON plan.
        Returns {"plan": str, "steps": [str]} or empty dict on failure.
        """
        from dispatcher import Dispatcher
        prompt = _PLAN_PROMPT.format(
            task=state.user_input,
            tools=self.registry.summary(),
            context=context[:1200],
        )
        raw = self._call(prompt)
        state.record_llm("improver_plan_raw", raw)

        parsed = Dispatcher.parse_llm_json(raw)
        if parsed and "steps" in parsed:
            state.plan       = parsed.get("plan", "")
            state.plan_steps = parsed.get("steps", [])
            log("improver_plan", {"steps": len(state.plan_steps), "plan": state.plan})
            return parsed

        # Fallback: treat raw output as a plain text plan
        log("improver_plan_fallback", {"raw": raw[:200]})
        state.plan       = raw
        state.plan_steps = [line.strip() for line in raw.splitlines() if line.strip()][:5]
        return {"plan": state.plan, "steps": state.plan_steps}

    # ── Phase 3: Decide next step ─────────────────────────────────────────────

    def next_step(self, state: RunState) -> dict:
        """
        Given current state, return {"step": str, "done": bool}.
        """
        from dispatcher import Dispatcher
        done_count = state.iteration
        prompt = _NEXT_STEP_PROMPT.format(
            task=state.user_input,
            plan=state.plan,
            done_steps=f"{done_count} of {len(state.plan_steps)}",
            tool_results="\n".join(state.recent_tool_outputs(5)),
        )
        raw = self._call(prompt)
        state.record_llm("improver_next_step", raw)

        parsed = Dispatcher.parse_llm_json(raw)
        if parsed:
            return parsed

        # If JSON parse fails, assume not done
        return {"step": raw, "done": False}

    # ── Phase 4: Improvement after validation failure ─────────────────────────

    def improve(self, state: RunState) -> str:
        """
        Called when validation fails. Returns a correction instruction for Coder.
        """
        from dispatcher import Dispatcher
        prompt = _IMPROVE_PROMPT.format(
            task=state.user_input,
            validation_notes=state.validation_notes,
            tool_results="\n".join(state.recent_tool_outputs(5)),
        )
        raw = self._call(prompt)
        state.record_llm("improver_improve", raw)

        parsed = Dispatcher.parse_llm_json(raw)
        if parsed:
            return parsed.get("correction", raw)
        return raw

    # ── Phase 5: Final output ─────────────────────────────────────────────────

    def summarize(self, state: RunState) -> str:
        """
        Produce the final user-facing output string.
        This is the only place prose is allowed.
        """
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
