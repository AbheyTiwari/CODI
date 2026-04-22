# core/executor.py
# ─────────────────────────────────────────────────────────────────────────────
# The Executor wraps the Coder LLM.
#
# It receives a single step description from the Improver,
# asks the Coder LLM to translate it into a JSON action bundle,
# and returns that bundle for the Dispatcher to execute.
#
# The Coder LLM ONLY outputs JSON. No prose. No explanation.
# ─────────────────────────────────────────────────────────────────────────────

import os
from langchain_core.messages import HumanMessage, SystemMessage

from dispatcher import Dispatcher
from llm_factory import get_coder_llm
from logger import log
from state.temp_db import RunState
from tools.registry import ToolRegistry


_SYSTEM = """\
You are the Coder — the execution engine of a coding agent called Codi.
You receive ONE step instruction and translate it into tool calls.

RULES:
- Respond ONLY with valid JSON. No prose. No markdown. No explanation.
- Use ONLY tools from the provided list.
- Working directory: {working_dir}
- Do not invent tool names.

Output format:
{{
  "action": "tool_call",
  "tools": [
    {{"name": "<tool_name>", "args": {{...}}}},
    {{"name": "<tool_name>", "args": {{...}}}}
  ]
}}

To signal you have nothing to do (step already done), output:
{{
  "action": "noop"
}}
"""

_STEP_PROMPT = """\
Step to execute: {step}

Available tools:
{tools}

Previous tool results (for context):
{context}

Output the JSON action bundle now.
"""


class Executor:
    def __init__(self, registry: ToolRegistry):
        self.registry   = registry
        self.dispatcher = Dispatcher(registry)
        self.llm        = get_coder_llm()

    def _sys(self) -> SystemMessage:
        return SystemMessage(content=_SYSTEM.format(
            working_dir=os.environ.get("CODI_WORKING_DIR", os.getcwd())
        ))

    def execute_step(self, step: str, state: RunState) -> dict:
        """
        Given a step description, ask the Coder LLM to produce an action bundle,
        then dispatch it. Returns the dispatcher result dict.
        """
        prompt = _STEP_PROMPT.format(
            step=step,
            tools=self.registry.summary(),
            context="\n".join(state.recent_tool_outputs(3)) or "(none yet)",
        )

        # ── Ask Coder LLM for JSON action bundle ──────────────────────────────
        try:
            resp = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            raw  = resp.content.strip()
        except Exception as e:
            log("executor_llm_error", {"step": step[:100], "error": str(e)})
            return {
                "status": "error",
                "results": [],
                "error": f"Coder LLM error: {e}",
            }

        state.record_llm("coder", raw)
        log("executor_coder_raw", {"step": step[:80], "raw": raw[:300]})

        # ── Parse JSON ────────────────────────────────────────────────────────
        action_bundle = Dispatcher.parse_llm_json(raw)

        if action_bundle is None:
            log("executor_parse_fail", {"raw": raw[:200]})
            # Try to salvage: if the raw text mentions a known tool, build noop
            return {
                "status":  "error",
                "results": [],
                "error":   f"Coder output was not valid JSON: {raw[:200]}",
            }

        # ── Dispatch ──────────────────────────────────────────────────────────
        dispatch_result = self.dispatcher.dispatch(action_bundle)

        # ── Store results in state ─────────────────────────────────────────────
        for r in dispatch_result.get("results", []):
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return dispatch_result
