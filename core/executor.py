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
import time
from langchain_core.messages import HumanMessage, SystemMessage

from context_trimmer import trim_tool_output
from dispatcher import Dispatcher, wrap_prompt_data
from llm_factory import get_coder_llm
from logger import log
from core.prompts import executor_system_prompt
from state.temp_db import RunState
from tools.registry import ToolRegistry


# ── Step prompt ───────────────────────────────────────────────────────────────
_STEP_PROMPT = """\
Step to execute: {step}

Requirements:
{requirements}

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

File to write: {path}
Task: {step}

Requirements:
{requirements}

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
_LARGE_CONTENT_EXTS = (".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".svg", ".xml", ".java")

# File write tool names
_WRITE_TOOLS = {"write_file", "create_file"}


def _estimate_token_count(text: str) -> int:
    """Rough token estimate used when the LLM client does not expose usage metadata."""
    if not text:
        return 0
    return max(1, len(re.findall(r"\S+", text)))


def _extract_token_metrics(response: object) -> dict[str, int]:
    """Extract prompt/output/total token metrics from a LangChain response if available."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        total_tokens = usage.get("total_tokens") or (prompt_tokens + output_tokens) or 0
        return {
            "prompt_tokens": int(prompt_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(total_tokens),
        }
    return {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _log_llm_metrics(stage: str, prompt: str, response: object, start_time: float, end_time: float) -> None:
    """Emit concise performance metrics about the LLM call."""
    content = getattr(response, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    metrics = _extract_token_metrics(response)
    raw_response_len = len(content)
    raw_response_tokens = metrics.get("output_tokens") or _estimate_token_count(content)
    log(stage, {
        "prompt_build_s": round((end_time - start_time), 3),
        "prompt_tokens": metrics.get("prompt_tokens", 0),
        "generation_started": True,
        "first_token_s": round((time.time() - start_time), 3),
        "generation_finished_s": round((end_time - start_time), 3),
        "output_tokens": metrics.get("output_tokens", 0),
        "raw_response_length_characters": raw_response_len,
        "raw_response_tokens": raw_response_tokens,
    })


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
    ext_pattern = r"([A-Za-z0-9_./\\-]+\.(?:html|css|js|ts|jsx|tsx|py|md|json|txt|svg|sh|xml|java))"
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
    True if the step is likely to require more content than a 7B model
    can safely JSON-serialize without truncating.

    Heuristics:
      1. All HTML files — always large (even "simple.html" needs boilerplate)
      2. CSS/JS/TS with design keywords
      3. CSS/JS/TS step contains "with" — means caller described content
      4. Step is longer than 60 chars — enough description = enough content
    """
    ext = os.path.splitext(path)[1].lower()
    step_lower = step.lower()

    if ext not in _LARGE_CONTENT_EXTS:
        return False

    # Always use content-first for HTML — it's almost always large
    if ext == ".html":
        return True

    # For CSS/JS/TS: trigger on design keywords
    if any(trigger in step_lower for trigger in _LARGE_CONTENT_TRIGGERS):
        return True

    # "with" in the step means the caller described what goes in the file
    # e.g. "write styles.css with dark theme" — content will be substantial
    if " with " in step_lower:
        return True

    # Long step description = complex requirements = large output
    if len(step) > 60:
        return True

    return False


def _should_use_content_first(step: str, path: str) -> bool:
    """Use content-first for any clear file-write step to avoid JSON repair loops."""
    tool, detected_path = _detect_file_write_step(step)
    if not tool or not detected_path:
        return False
    if path and detected_path != path:
        return False
    return True


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

        context_str = trim_tool_output(
            "\n".join(state.recent_tool_outputs(4)) or "(none yet)",
            max_tokens=800,
        )
        prompt = _CONTENT_PROMPT.format(
            path=path,
            step=step,
            requirements=state.requirements.as_prompt_block(),
            context=wrap_prompt_data(context_str, path=path),
        )

        try:
            start_time = time.time()
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            end_time = time.time()
            content = resp.content.strip()
            _log_llm_metrics("executor_content_first_llm", prompt, resp, start_time, end_time)
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

        action_bundle = {
            "action": "tool_call",
            "tools":  [{"name": tool, "args": {"path": path, "content": content}}],
        }

        violation = self._framework_violation(state, action_bundle)
        if violation:
            log("executor_framework_violation", {
                "tool": tool,
                "path": path,
                "reason": violation,
            })
            state.add_tool_result(tool, "error", violation)
            return {
                "status": "error",
                "results": [{"tool": tool, "status": "error", "output": violation}],
                "error": violation,
            }

        # Dispatch directly — no LLM JSON round-trip
        dispatch_result = self.dispatcher.dispatch(action_bundle)
        if dispatch_result.get("signal") in ("noop", "done"):
            signal = dispatch_result.get("signal", "noop")
            state.add_tool_result("dispatcher", "ok", signal)
            log("executor_dispatch_signal", {
                "signal": signal,
                "step": step[:160],
                "action": action_bundle.get("action"),
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
        from context_trimmer import trim_tool_output

        # ── Content-first routing ─────────────────────────────────────────────
        tool, path = _detect_file_write_step(step)
        if tool and path and (_is_large_content_step(step, path) or _should_use_content_first(step, path)):
            log("tool_routing", {
                "strategy": "content_first",
                "tool": tool,
                "path": path[:80],
                "repair": False,
            })
            return self._execute_content_first(step, tool, path, state)

        # ── Standard JSON path ────────────────────────────────────────────────
        prompt = _STEP_PROMPT.format(
            step=step,
            requirements=state.requirements.as_prompt_block(),
            tools=self.registry.summary(),
            context=wrap_prompt_data(
                trim_tool_output(state.context_snapshot(max_recent=3), max_tokens=900)
                or "(none yet)"
            ),
        )

        # ── Ask Coder LLM ─────────────────────────────────────────────────────
        try:
            start_time = time.time()
            resp = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            end_time = time.time()
            raw = resp.content.strip()
            _log_llm_metrics("executor_llm_call", prompt, resp, start_time, end_time)
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

        # ── Parse ─────────────────────────────────────────────────────────────
        action_bundle = Dispatcher.parse_llm_json(raw)
        repair_needed = False

        # If parse failed, try once more with the repair prompt
        if action_bundle is None:
            action_bundle = self._repair_action_bundle(step, raw, state)
            repair_needed = True

        # If STILL None and this looks like a truncated file-write, switch
        # to content-first as a last resort (catches cases where the LLM
        # produced partial JSON with a file path but no closing string)
        if action_bundle is None:
            tool_fb, path_fb = _detect_file_write_step(step)
            if tool_fb and path_fb:
                log("tool_routing", {
                    "strategy": "json_fallback_content_first",
                    "tool": tool_fb,
                    "path": path_fb[:80],
                    "repair": repair_needed,
                })
                return self._execute_content_first(step, tool_fb, path_fb, state)

        if action_bundle is None:
            error = f"Coder output was not valid JSON: {raw[:200]}"
            log("tool_routing", {
                "strategy": "json",
                "status": "parse_fail",
                "repair": repair_needed,
            })
            state.add_tool_result("coder", "error", error)
            return {
                "status":  "error",
                "results": [{"tool": "coder", "status": "error", "output": error}],
                "error":   error,
            }

        # Extract which tools will be called
        tools_to_call = []
        if isinstance(action_bundle, dict) and "tools" in action_bundle:
            tools_to_call = [t.get("name", "unknown") for t in action_bundle.get("tools", [])]

        log("tool_routing", {
            "strategy": "json",
            "tools": tools_to_call,
            "repair": repair_needed,
            "action": action_bundle.get("action"),
        })

        action = action_bundle.get("action")
        if action in ("tool_call", "parallel") and not tools_to_call:
            error = "Executor produced a tool_call action with no tools; no requested work could run."
            log("tool_routing", {
                "strategy": "json",
                "status": "empty_tool_list",
                "step": step[:160],
                "repair": repair_needed,
                "action_bundle": str(action_bundle)[:500],
            })
            state.add_tool_result("dispatcher", "error", error)
            return {
                "status":  "error",
                "results": [{"tool": "dispatcher", "status": "error", "output": error}],
                "error":   error,
            }

        violation = self._framework_violation(state, action_bundle)
        if violation:
            log("executor_framework_violation", {
                "step": step[:80],
                "reason": violation,
            })
            state.add_tool_result("coder", "error", violation)
            return {
                "status": "error",
                "results": [{"tool": "coder", "status": "error", "output": violation}],
                "error": violation,
            }

        # ── Dispatch ──────────────────────────────────────────────────────────
        dispatch_result = self.dispatcher.dispatch(action_bundle)

        # ── Store results in state ────────────────────────────────────────────
        if dispatch_result.get("signal") in ("noop", "done"):
            signal = dispatch_result.get("signal", "noop")
            state.add_tool_result("dispatcher", "ok", signal)
            log("executor_dispatch_signal", {
                "signal": signal,
                "step": step[:160],
                "action": action_bundle.get("action"),
            })

        results = dispatch_result.get("results", [])
        if not results and dispatch_result.get("status") == "error":
            state.add_tool_result(
                "dispatcher", "error",
                dispatch_result.get("error", "Dispatcher returned no results.")
            )

        for r in results:
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return dispatch_result

    def _framework_violation(self, state: RunState, action_bundle: dict) -> str | None:
        forbidden = getattr(state, "requirements", None)
        if not forbidden:
            return None
        patterns = forbidden.framework_lock()
        if not patterns or not isinstance(action_bundle, dict):
            return None

        for tool in action_bundle.get("tools", []):
            if not isinstance(tool, dict):
                continue
            args = tool.get("args") or {}
            for value in args.values():
                if not isinstance(value, str):
                    continue
                lowered = value.lower()
                for pattern in patterns:
                    if pattern.lower() in lowered:
                        return (
                            f"Forbidden framework content detected in tool args: '{pattern}'. "
                            f"This task is locked to {state.requirements.framework}."
                        )
        return None


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
