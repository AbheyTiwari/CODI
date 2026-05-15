# core/executor.py
# ─────────────────────────────────────────────────────────────────────────────
# The Executor wraps the Coder LLM.
#
# Receives a step description from the Improver, asks the Coder LLM to
# translate it into a JSON action bundle, then dispatches it.
#
# All prompts live in core/prompts.py — never inline here.
# The Dispatcher normalizes malformed LLM output before routing.
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
from langchain_core.messages import HumanMessage, SystemMessage

from dispatcher import Dispatcher
from llm_factory import get_coder_llm
from logger import log
from core.prompts import executor_system_prompt
from state.temp_db import RunState
from tools.registry import ToolRegistry


# ── Step prompt ───────────────────────────────────────────────────────────────
_STEP_PROMPT = """\
Step to execute: {step}

Available tools:
{tools}

Previous tool results (for context):
{context}

Output the JSON action bundle now. JSON only — no prose, no fences."""


# ── Repair prompt ─────────────────────────────────────────────────────────────
_REPAIR_PROMPT = """\
Your previous output was NOT valid JSON. No tool ran.

Original step: {step}

What you output (broken):
{raw}

Available tools:
{tools}

Output ONLY the corrected JSON. Use exactly this structure:
{{"action":"tool_call","tools":[{{"name":"TOOL_NAME","args":{{ARGS}}}}]}}

Or if nothing to do: {{"action":"noop"}}

JSON only:"""


# ── Content-first prompt ──────────────────────────────────────────────────────
# Used when the step involves writing a file with substantial content.
# The LLM outputs the file content directly (no JSON wrapper) so it can
# use its full context window for content instead of JSON escaping overhead.
_CONTENT_PROMPT = """\
You are writing the full content of a file. Output ONLY the raw file content.
No JSON. No markdown fences. No explanation. Just the file content itself.

HARD CONSTRAINTS — violating these is automatic failure:
{requirements}

File to write: {path}
Task: {step}

Context from project:
{context}

Output the complete file content now:"""


# ── Token threshold for switching to content-first mode ───────────────────────
# If a step mentions writing a file and the expected content is likely large
# (HTML/CSS with design requirements, full scripts, etc.), bypass JSON entirely.
_LARGE_CONTENT_TRIGGERS = (
    "modern", "sleek", "design", "website", "webpage", "landing", "dashboard",
    "terminal", "portal", "app", "full", "complete", "entire", "whole",
    "beautiful", "styled", "animated", "responsive", "interactive",
    "dark", "light", "theme", "glass", "gradient", "shadow", "effect",
    "layout", "component", "feature", "section", "header", "footer",
    "nav", "card", "modal", "form", "button", "style", "color", "font",
)

# Extensions that commonly produce large content
_LARGE_CONTENT_EXTS = (".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".svg", ".py")

# File write tool names
_WRITE_TOOLS = {"write_file", "create_file"}


def _detect_file_write_step(step: str) -> tuple[str | None, str | None]:
    """
    If this step is clearly a file-write operation, return (tool_name, path).
    Otherwise return (None, None).

    Detects patterns like:
      - "Write index.html with ..."
      - "Create codi.html using create_file ..."
      - "Use write_file to save styles.css ..."
    """
    step_lower = step.lower()

    # Must mention a write/create action
    write_keywords = ("write", "create", "save", "generate", "produce", "output")
    if not any(kw in step_lower for kw in write_keywords):
        return None, None

    # Extract file path — look for known extensions
    ext_pattern = r"([A-Za-z0-9_./\\-]+\.(?:html|css|js|ts|jsx|tsx|py|md|json|txt|svg|sh))"
    match = re.search(ext_pattern, step)
    if not match:
        return None, None

    path = match.group(1).strip("'\"` ")
    ext  = os.path.splitext(path)[1].lower()

    # Determine which write tool to use
    tool = "create_file" if "create" in step_lower else "write_file"

    return tool, path


def _is_large_content_step(step: str, path: str) -> bool:
    """
    True only when the step requires GENERATING a full new file from scratch.
    False for edits to existing files — those go through edit_file via JSON.

    The key distinction:
      CREATE  -> content-first (LLM writes raw content, no JSON overhead)
      EDIT    -> JSON path     (edit_file with old/new, no full rewrite needed)

    Bad signals (removed):
      - "all html files are large" - false for single-line edits
      - "step contains with" - edit steps say "replace X with Y"
      - "step > 60 chars" - almost every edit step is over 60 chars

    Good signals (kept):
      - Step explicitly creates/writes a NEW file (not edit/update/modify)
      - File extension is one that produces large content
      - Step mentions design/content keywords that imply substantial output
    """
    ext = os.path.splitext(path)[1].lower()
    step_lower = step.lower()

    if ext not in _LARGE_CONTENT_EXTS:
        return False

    # If this is an edit operation, NEVER use content-first.
    # edit_file handles these correctly via old/new replacement — no full rewrite.
    edit_keywords = ("edit", "update", "modify", "change", "fix", "replace",
                     "append", "prepend", "insert", "add to", "remove from")
    if any(kw in step_lower for kw in edit_keywords):
        return False

    # Must be a creation step to proceed
    create_keywords = ("write", "create", "generate", "build", "produce", "make")
    is_create = any(kw in step_lower for kw in create_keywords)
    if not is_create:
        return False

    # HTML creation: always content-first — even "simple.html" needs full boilerplate
    if ext == ".html":
        return True

    # Python files: use content-first when the step describes a framework app.
    # JSON path truncates at token limit, producing incomplete/wrong code.
    if ext == ".py":
        py_triggers = (
            "fastapi", "flask", "django", "app", "api", "endpoint", "route",
            "handle", "request", "response", "server", "uvicorn", "starlette",
        )
        if any(t in step_lower for t in py_triggers):
            return True

    # CSS/JS/TS creation: only if design/content keywords are present.
    # Without these it is likely a short utility file — JSON handles it fine.
    if any(trigger in step_lower for trigger in _LARGE_CONTENT_TRIGGERS):
        return True

    return False


class Executor:
    def __init__(self, registry: ToolRegistry):
        self.registry   = registry
        self.dispatcher = Dispatcher(registry)
        self.llm        = get_coder_llm()

    def _sys(self) -> SystemMessage:
        """Build the system message from the canonical prompt in prompts.py."""
        return SystemMessage(content=executor_system_prompt(
            tool_names=self.registry.list_names(),
        ))

    def _repair_action_bundle(self, step: str, raw: str, state: RunState) -> dict | None:
        """Second attempt: send the broken output back to the LLM with a tighter repair prompt."""
        prompt = _REPAIR_PROMPT.format(
            step=step,
            tools=self.registry.summary(),
            raw=raw[:2000],
        )
        try:
            resp     = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            repaired = resp.content.strip()
        except Exception as e:
            log("executor_repair_error", {"step": step[:100], "error": str(e)})
            return None

        state.record_llm("coder_repair", repaired)
        log("executor_repair_raw", {"step": step[:80], "raw": repaired[:300]})
        return Dispatcher.parse_llm_json(repaired)

    # ── Content-first strategy ────────────────────────────────────────────────

    def _execute_content_first(
        self, step: str, tool: str, path: str, state: RunState
    ) -> dict:
        """
        Bypass JSON entirely for large file writes.

        Strategy:
          1. Ask the coder LLM to output ONLY the raw file content.
          2. Take that raw output and call write_file/create_file directly.

        This means the LLM's full output window goes to content quality,
        not to JSON escaping. No truncation. No parse failures.
        """
        log("executor_content_first", {"tool": tool, "path": path, "step": step[:80]})

        context_str = "\n".join(state.recent_tool_outputs(4)) or "(none yet)"
        requirements_str = state.requirements.as_prompt_block()
        prompt = _CONTENT_PROMPT.format(
            requirements=requirements_str,
            path=path,
            step=step,
            context=context_str[:800],
        )

        try:
            resp    = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content.strip()
        except Exception as e:
            error = f"Coder LLM error (content-first): {e}"
            log("executor_content_first_error", {"error": str(e)})
            state.add_tool_result(tool, "error", error)
            return {
                "status":  "error",
                "results": [{"tool": tool, "status": "error", "output": error}],
                "error":   error,
            }

        state.record_llm("coder_content_first", content[:200])
        log("executor_content_first_raw", {"path": path, "content_len": len(content)})

        # Strip any accidental markdown fences the model adds anyway
        content = _strip_fences(content)

        if not content:
            error = "Coder returned empty content for file write."
            log("executor_content_first_empty", {"path": path})
            state.add_tool_result(tool, "error", error)
            return {
                "status":  "error",
                "results": [{"tool": tool, "status": "error", "output": error}],
                "error":   error,
            }

        # Framework contamination check before writing to disk
        forbidden = state.requirements.framework_lock()
        for pattern in forbidden:
            if pattern.lower() in content.lower():
                error = (
                    f"Framework contamination blocked: found '{pattern}' in content "
                    f"for a {state.requirements.framework} project. "
                    f"Rewrite without {pattern}."
                )
                log("executor_contamination_blocked", {"pattern": pattern, "path": path})
                state.add_tool_result(tool, "error", error)
                return {
                    "status":  "error",
                    "results": [{"tool": tool, "status": "error", "output": error}],
                    "error":   error,
                }

        # Dispatch directly — no LLM JSON round-trip
        dispatch_result = self.dispatcher.dispatch({
            "action": "tool_call",
            "tools":  [{"name": tool, "args": {"path": path, "content": content}}],
        })

        # Store results
        for r in dispatch_result.get("results", []):
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return dispatch_result

    # ── Main entry point ──────────────────────────────────────────────────────

    def execute_step(self, step: str, state: RunState) -> dict:
        """
        Translate a step description into a tool call and execute it.
        Returns the dispatcher result dict.

        For steps that involve writing large files (HTML, CSS, JS with design
        requirements), uses the content-first strategy to avoid JSON truncation.
        """

        # ── Content-first routing ─────────────────────────────────────────────
        tool, path = _detect_file_write_step(step)
        if tool and path and _is_large_content_step(step, path):
            log("executor_routing", {"strategy": "content_first", "path": path})
            return self._execute_content_first(step, tool, path, state)

        # ── Standard JSON path ────────────────────────────────────────────────
        log("executor_routing", {"strategy": "json", "step": step[:60]})

        prompt = _STEP_PROMPT.format(
            step=step,
            tools=self.registry.summary(),
            context="\n".join(state.recent_tool_outputs(8)) or "(none yet)",
        )

        # ── Ask Coder LLM ─────────────────────────────────────────────────────
        try:
            resp = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            raw  = resp.content.strip()
        except Exception as e:
            error = f"Coder LLM error: {e}"
            log("executor_llm_error", {"step": step[:100], "error": str(e)})
            state.add_tool_result("coder", "error", error)
            return {
                "status":  "error",
                "results": [{"tool": "coder", "status": "error", "output": error}],
                "error":   error,
            }

        state.record_llm("coder", raw)
        log("executor_coder_raw", {"step": step[:80], "raw": raw[:300]})

        # ── Parse ─────────────────────────────────────────────────────────────
        action_bundle = Dispatcher.parse_llm_json(raw)

        # If parse failed, try once more with the repair prompt
        if action_bundle is None:
            action_bundle = self._repair_action_bundle(step, raw, state)

        # If STILL None and this looks like a truncated file-write, switch
        # to content-first as a last resort (catches cases where the LLM
        # produced partial JSON with a file path but no closing string)
        if action_bundle is None:
            tool_fb, path_fb = _detect_file_write_step(step)
            if tool_fb and path_fb:
                log("executor_json_fallback_content_first", {
                    "step": step[:80], "path": path_fb
                })
                return self._execute_content_first(step, tool_fb, path_fb, state)

        if action_bundle is None:
            error = f"Coder output was not valid JSON: {raw[:200]}"
            log("executor_parse_fail", {"raw": raw[:200]})
            state.add_tool_result("coder", "error", error)
            return {
                "status":  "error",
                "results": [{"tool": "coder", "status": "error", "output": error}],
                "error":   error,
            }

        # ── Pre-dispatch contamination check on JSON path ─────────────────────
        # The content-first path checks before writing. The JSON path must too.
        forbidden = state.requirements.framework_lock()
        if forbidden and action_bundle.get("action") == "tool_call":
            for tc in action_bundle.get("tools", []):
                if tc.get("name") in ("write_file", "create_file", "edit_file"):
                    file_content = tc.get("args", {}).get("content", "")
                    for pattern in forbidden:
                        if pattern.lower() in file_content.lower():
                            error = (
                                f"Framework contamination blocked (JSON path): "
                                f"'{pattern}' found. This is a "
                                f"{state.requirements.framework} project. "
                                f"Rewrite without {pattern}."
                            )
                            log("executor_contamination_blocked_json", {
                                "pattern": pattern,
                                "path": tc.get("args", {}).get("path", "?")
                            })
                            state.add_tool_result("coder", "error", error)
                            return {
                                "status":  "error",
                                "results": [{"tool": "coder", "status": "error", "output": error}],
                                "error":   error,
                            }

        # ── Dispatch ──────────────────────────────────────────────────────────
        dispatch_result = self.dispatcher.dispatch(action_bundle)

        # ── Store results in state ────────────────────────────────────────────
        results = dispatch_result.get("results", [])
        if not results and dispatch_result.get("status") == "error":
            state.add_tool_result(
                "dispatcher", "error",
                dispatch_result.get("error", "Dispatcher returned no results.")
            )

        for r in results:
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return dispatch_result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences that models sometimes add even when told not to.
    Handles:
      ```html ... ```
      ```css  ... ```
      ```     ... ```
    """
    text = text.strip()
    # Match opening fence with optional language tag
    fence_re = re.compile(r"^```[a-z]*\s*\n?", re.IGNORECASE)
    if fence_re.match(text):
        text = fence_re.sub("", text, count=1)
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()