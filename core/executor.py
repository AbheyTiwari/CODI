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
- The top-level action must be "tool_call" or "noop". Never output custom actions.
- For write_file with multi-line files such as HTML, CSS, or JS, prefer:
  {{"path": "index.html", "content_lines": ["<!doctype html>", "<html>", "..."]}}
- Use create_file or write_file for new files. For targeted changes, use edit_file:
  {{"path": "file.txt", "old": "existing text", "new": "replacement"}}
  or {{"path": "file.txt", "append": "text to add"}}
- When creating HTML files, READ THE STEP DESCRIPTION CAREFULLY.
  Create content that EXACTLY MATCHES the requirements in the step.
  DO NOT use generic boilerplate for different types of files.
  Example: If step says "Create portfolio with projects showcase and skills",
           Create diverse sections (projects, skills, experience) — NOT just a generic template.
  Example: If step says "Create simple starter page", create a basic template.
  Different files must have meaningfully different content.
- Do not put raw file content outside the JSON action bundle.
- Do not open a browser unless the user explicitly asks you to. Creating a file is enough for create/write tasks.
- IMPORTANT: If the step says to create a boilerplate or basic structure, use write_file with minimal content.
- IMPORTANT: If the step says to edit or add content to an existing file, use edit_file with old/new or append.

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

CRITICAL: This step contains SPECIFIC CONTENT REQUIREMENTS.
- Read the step description carefully and identify what makes this output unique.
- DO NOT reuse content from previous files.
- Create content that EXACTLY MATCHES the step requirements.
- If the step says to create a portfolio, include portfolio-specific content.
- If the step says to create a simple page, create simple content.
- Look at previous results if similar files were created — make this one different.
- IMPORTANT: If this step says to EDIT or ADD content to an existing file, use edit_file with old/new or append.
- IMPORTANT: If this step says to CREATE a boilerplate or basic structure, use write_file with minimal content.

Output the JSON action bundle now.
"""

_REPAIR_PROMPT = """\
Your previous response was not valid JSON, so no tool could run.

Original step:
{step}

Available tools:
{tools}

Invalid response:
{raw}

Return the corrected JSON action bundle only.
If using write_file for multi-line content, put the file body in content_lines as a JSON array of strings.
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

    def _repair_action_bundle(self, step: str, raw: str, state: RunState) -> dict | None:
        prompt = _REPAIR_PROMPT.format(
            step=step,
            tools=self.registry.summary(),
            raw=raw[:4000],
        )
        try:
            resp = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            repaired = resp.content.strip()
        except Exception as e:
            log("executor_repair_error", {"step": step[:100], "error": str(e)})
            return None

        state.record_llm("coder_repair", repaired)
        log("executor_repair_raw", {"step": step[:80], "raw": repaired[:300]})
        return Dispatcher.parse_llm_json(repaired)

    def execute_step(self, step: str, state: RunState) -> dict:
        """
        Given a step description, ask the Coder LLM to produce an action bundle,
        then dispatch it. Returns the dispatcher result dict.
        """
        prompt = _STEP_PROMPT.format(
            step=step,
            tools=self.registry.summary(),
            context="\n".join(state.recent_tool_outputs(8)) or "(none yet)",
        )

        # ── Ask Coder LLM for JSON action bundle ──────────────────────────────
        try:
            resp = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            raw  = resp.content.strip()
        except Exception as e:
            error = f"Coder LLM error: {e}"
            log("executor_llm_error", {"step": step[:100], "error": str(e)})
            state.add_tool_result("coder", "error", error)
            return {
                "status": "error",
                "results": [{"tool": "coder", "status": "error", "output": error}],
                "error": error,
            }

        state.record_llm("coder", raw)
        log("executor_coder_raw", {"step": step[:80], "raw": raw[:300]})

        # ── Parse JSON ────────────────────────────────────────────────────────
        action_bundle = Dispatcher.parse_llm_json(raw)
        if action_bundle is None:
            action_bundle = self._repair_action_bundle(step, raw, state)

        if action_bundle is None:
            error = f"Coder output was not valid JSON: {raw[:200]}"
            log("executor_parse_fail", {"raw": raw[:200]})
            state.add_tool_result("coder", "error", error)
            return {
                "status":  "error",
                "results": [{"tool": "coder", "status": "error", "output": error}],
                "error":   error,
            }

        # ── Dispatch ──────────────────────────────────────────────────────────
        dispatch_result = self.dispatcher.dispatch(action_bundle)

        # ── Store results in state ─────────────────────────────────────────────
        results = dispatch_result.get("results", [])
        if not results and dispatch_result.get("status") == "error":
            state.add_tool_result("dispatcher", "error", dispatch_result.get("error", "Dispatcher returned no results."))

        for r in results:
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return dispatch_result
