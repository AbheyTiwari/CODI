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

import json
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
from status_stream import emit_status


# ── Step prompt ───────────────────────────────────────────────────────────────
_STEP_PROMPT = """\
Step to execute: {step}

Requirements:
{requirements}

Available tools:
{tools}

Previous tool results (for context):
{context}

Protocol: choose exactly one atomic tool action for this step. Do not emit a batch of tool calls.
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
# Used when the step involves writing a file.
# The LLM outputs markdown-wrapped file content, not a JSON tool bundle, so it
# can use its full context window for content instead of JSON escaping overhead.
_CONTENT_PROMPT = """\
You are writing the full content of a file.
Do not output JSON. Do not describe your reasoning.

Output format:
1. A single markdown code fence containing the complete file content.
2. After the closing fence, output this exact sentinel on its own line:
{sentinel}

The sentinel is not part of the file. It tells the executor that generation finished.

File to write: {path}
Task: {step}

Requirements:
{requirements}

Project manifest:
{project_manifest}

Context from project:
{context}

Output the complete file content now:"""


_CONTENT_CONTINUATION_PROMPT = """\
The previous file generation for {path} ended before the required sentinel.
Continue from the exact next character after the current partial content.
Do not repeat content that is already present. Do not output JSON. Do not explain.

When the file is complete, close any markdown fence you opened and output this exact sentinel on its own line:
{sentinel}

Task: {step}

Tail of current partial content:
{tail}

Output only the remaining file content:"""


_CONTENT_COMPLETE_SENTINEL = "CODI_FILE_WRITE_COMPLETE"
_MAX_CONTENT_CONTINUATIONS = 3


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

    # Extract file path — look for common source, config, and document extensions
    ext_pattern = r"([A-Za-z0-9_./\\-]+\.(?:py|java|html|css|js|ts|jsx|tsx|json|md|txt|xml|yml|yaml|sh|sql|c|cpp|h|hpp|go|rs|rb|php|graphql|svg))"
    match = re.search(ext_pattern, step)
    if not match:
        return None, None

    path = match.group(1).strip("'\"` ")
    ext  = os.path.splitext(path)[1].lower()

    # Determine which write tool to use
    tool = "create_file" if "create" in step_lower else "write_file"

    return tool, path


def _should_use_content_first(step: str, path: str) -> bool:
    """Use content-first for any clear file-write step to avoid JSON repair loops."""
    tool, detected_path = _detect_file_write_step(step)
    if not tool or not detected_path:
        return False
    if path and detected_path != path:
        return False
    return True


def _cli_status(message: str) -> None:
    """Show high-level progress in the CLI without exposing hidden reasoning."""
    emit_status("executor", message)


def _has_completion_sentinel(text: str) -> bool:
    return _CONTENT_COMPLETE_SENTINEL in (text or "")


def _extract_markdown_file_content(text: str) -> tuple[str, bool]:
    """
    Extract generated file content from the markdown content protocol.
    Returns (content_without_sentinel, complete).
    """
    raw = text or ""
    complete = _has_completion_sentinel(raw)
    before_sentinel = raw.split(_CONTENT_COMPLETE_SENTINEL, 1)[0] if complete else raw
    return _strip_fences(before_sentinel), complete


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

    def _repair_action_bundle(self, step: str, raw: str, state: RunState, reason: str) -> dict | None:
        """Send the broken output back to the LLM with a tighter repair prompt."""
        prompt = _REPAIR_PROMPT.format(
            step=step,
            tools=self.registry.summary(),
            raw=raw[:2000],
        )
        try:
            start_time = time.time()
            resp = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            end_time = time.time()
            repaired = resp.content.strip()
            _log_llm_metrics("executor_repair_llm", prompt, resp, start_time, end_time)
        except Exception as e:
            log("executor_repair_error", {"step": step[:100], "error": str(e)})
            return None

        state.record_llm("coder_repair", repaired)
        log("executor_repair_raw", {"step": step[:80], "raw": repaired[:300]})
        log("repair_reason", {"reason": reason, "step": step[:120]})
        log("repair_attempt", {"step": step[:120], "attempt": True})
        return Dispatcher.parse_llm_json(repaired)

    # ── Content-first strategy ────────────────────────────────────────────────

    def _execute_content_first(
        self, step: str, tool: str, path: str, state: RunState
    ) -> dict:
        """
        Bypass LLM JSON entirely for file writes.

        Strategy:
          1. Ask the coder LLM to output markdown-wrapped file content.
          2. Require an explicit completion sentinel after the content.
          3. If the sentinel is missing, ask only for the remaining content.
          4. Send the final content through a direct dispatcher write path.

        This means the LLM's full output window goes to content quality,
        not to JSON escaping.
        """
        if state.already_written(path):
            message = f"skip_duplicate_write:{path}"
            log("executor_duplicate_write", {"path": path, "step": step[:80]})
            state.add_tool_result("dispatcher", "ok", message)
            return {"status": "success", "results": [{"tool": "dispatcher", "status": "ok", "output": message}]}

        log("executor_content_first", {"tool": tool, "path": path, "step": step[:80]})
        _cli_status(f"Generating file content for {path} using markdown protocol.")

        context_str = trim_tool_output(
            "\n".join(state.recent_tool_outputs(4)) or "(none yet)",
            max_tokens=800,
        )
        project_manifest = state.project_manifest or {"package": None, "files_created": {}}
        package_hint = f"MUST use this exact package declaration: {project_manifest.get('package')}" if project_manifest.get('package') else "(no package lock)"
        manifest_text = f"package: {project_manifest.get('package') or 'none'}\nfiles_created: {json.dumps(project_manifest.get('files_created', {}), ensure_ascii=True)}\npackage_instruction: {package_hint}"
        prompt = _CONTENT_PROMPT.format(
            path=path,
            step=step,
            requirements=state.requirements.as_prompt_block(),
            project_manifest=manifest_text,
            context=wrap_prompt_data(context_str, path=path),
            sentinel=_CONTENT_COMPLETE_SENTINEL,
        )

        try:
            start_time = time.time()
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            end_time = time.time()
            raw_content = resp.content.strip()
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

        state.record_llm("coder_content_first", raw_content[:200])
        log("executor_content_first_raw", {"path": path, "content_len": len(raw_content)})

        complete = _has_completion_sentinel(raw_content)
        continuation_attempts = 0
        while not complete and continuation_attempts < _MAX_CONTENT_CONTINUATIONS:
            continuation_attempts += 1
            _cli_status(
                f"Generation for {path} is missing the completion marker; requesting continuation "
                f"{continuation_attempts}/{_MAX_CONTENT_CONTINUATIONS}."
            )
            log("executor_content_continuation", {
                "path": path,
                "attempt": continuation_attempts,
                "current_len": len(raw_content),
            })
            continuation_prompt = _CONTENT_CONTINUATION_PROMPT.format(
                path=path,
                step=step,
                sentinel=_CONTENT_COMPLETE_SENTINEL,
                tail=wrap_prompt_data(raw_content[-2000:], path=path),
            )
            try:
                start_time = time.time()
                resp = self.llm.invoke([HumanMessage(content=continuation_prompt)])
                end_time = time.time()
                continuation = resp.content.strip()
                _log_llm_metrics(
                    "executor_content_continuation_llm",
                    continuation_prompt,
                    resp,
                    start_time,
                    end_time,
                )
            except Exception as e:
                error = f"Coder LLM error during content continuation: {e}"
                log("executor_content_continuation_error", {"error": str(e), "path": path})
                state.add_tool_result(tool, "error", error)
                return {
                    "status": "error",
                    "results": [{"tool": tool, "status": "error", "output": error}],
                    "error": error,
                }

            state.record_llm("coder_content_continuation", continuation[:200])
            raw_content += continuation
            complete = _has_completion_sentinel(raw_content)

        content, complete = _extract_markdown_file_content(raw_content)
        if not complete:
            error = (
                f"{_CONTENT_COMPLETE_SENTINEL} missing after {continuation_attempts} continuation attempts; "
                "generated content may be truncated. Ask the coder to regenerate only the remaining portion."
            )
            log("executor_content_incomplete", {
                "path": path,
                "attempts": continuation_attempts,
                "content_len": len(content),
            })
            state.add_tool_result(tool, "error", error)
            return {
                "status": "error",
                "results": [{"tool": tool, "status": "error", "output": error}],
                "error": error,
            }

        if not content:
            error = "Coder returned empty content for file write."
            log("executor_content_first_empty", {"path": path})
            state.add_tool_result(tool, "error", error)
            return {
                "status":  "error",
                "results": [{"tool": tool, "status": "error", "output": error}],
                "error":   error,
            }

        _cli_status(f"Completion marker found for {path}; writing {len(content)} characters.")
        action_bundle = {"tools": [{"name": tool, "args": {"path": path, "content": content}}]}

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
        dispatch_result = self.dispatcher.dispatch_file_write(
            tool,
            path,
            content,
            {
                "code_generation_complete": True,
                "completion_sentinel": _CONTENT_COMPLETE_SENTINEL,
                "continuation_attempts": continuation_attempts,
            },
        )
        state.mark_written(path)
        if isinstance(state.project_manifest, dict):
            state.project_manifest.setdefault("files_created", {})[path] = content.splitlines()[0][:100] if content.splitlines() else "generated file"
        if dispatch_result.get("signal") in ("noop", "done"):
            signal = dispatch_result.get("signal", "noop")
            state.add_tool_result("dispatcher", "ok", signal)
            log("executor_dispatch_signal", {
                "signal": signal,
                "step": step[:160],
                "action": "direct_file_write",
            })

        # Store results
        for r in dispatch_result.get("results", []):
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return dispatch_result

    def _validate_atomic_contract(self, step: str, action_bundle: dict, raw: str, state: RunState) -> tuple[bool, str | None, list[dict]]:
        """Validate that the LLM returned exactly one atomic tool action affecting one file."""
        if not isinstance(action_bundle, dict):
            log("executor_contract_check", {"step": step[:120], "passed": False, "reason": "invalid_bundle"})
            return False, "Response was not a valid action bundle.", []

        action = action_bundle.get("action")
        if action in ("noop", "done", "control"):
            log("executor_contract_check", {"step": step[:120], "passed": True, "reason": "noop_signal"})
            return True, None, []

        tools = action_bundle.get("tools", [])
        if not isinstance(tools, list):
            log("executor_contract_check", {"step": step[:120], "passed": False, "reason": "tools_not_list"})
            return False, "Response did not contain a valid tools list.", []

        tool_count = len(tools)
        log("executor_contract_check", {"step": step[:120], "passed": tool_count == 1, "tool_count": tool_count, "reason": "tool_count_check"})
        if tool_count != 1:
            return False, f"Expected exactly one tool call but received {tool_count}.", tools

        tool = tools[0]
        if not isinstance(tool, dict):
            log("executor_contract_check", {"step": step[:120], "passed": False, "reason": "tool_not_object"})
            return False, "Tool entry was malformed.", tools

        name = tool.get("name")
        args = tool.get("args") or {}
        if not isinstance(name, str) or not name:
            log("executor_contract_check", {"step": step[:120], "passed": False, "reason": "missing_tool_name"})
            return False, "Tool name was missing.", tools

        if not isinstance(args, dict):
            log("executor_contract_check", {"step": step[:120], "passed": False, "reason": "args_not_object"})
            return False, "Tool arguments were malformed.", tools

        if name in _WRITE_TOOLS:
            path = args.get("path")
            if not path:
                log("executor_contract_check", {"step": step[:120], "passed": False, "reason": "missing_path"})
                return False, "The tool call did not include a file path.", tools

        if action not in ("tool_call", "parallel") and action != "noop":
            log("executor_contract_check", {"step": step[:120], "passed": False, "reason": "unsupported_action"})
            return False, f"Unsupported action: {action}", tools

        return True, None, tools

    def _split_child_steps(self, step: str) -> list[str]:
        """Split a broad planner step into child steps when it clearly targets multiple files."""
        lowered = step.lower()
        if not any(keyword in lowered for keyword in ("implement", "create", "build", "add", "write")):
            return [step]

        if " and " not in lowered and "," not in lowered:
            return [step]

        base = step.strip()
        parts = [p.strip() for p in re.split(r"\s*(?:,|and)\s*", base) if p.strip()]
        if len(parts) <= 1:
            return [step]

        child_steps = []
        for part in parts:
            if part.lower().endswith(("class", "record", "model", "service", "repository", "exception")):
                child_steps.append(f"Create {part}")
            else:
                child_steps.append(f"Create {part}")
        return child_steps

    # ── Main entry point ──────────────────────────────────────────────────────

    def execute_step(self, step: str, state: RunState) -> dict:
        """
        Translate a step description into a tool call and execute it.
        Returns the dispatcher result dict.

        For steps that involve writing large files (HTML, CSS, JS with design
        requirements), uses the content-first strategy to avoid JSON truncation.
        """
        from context_trimmer import trim_tool_output
        _cli_status(f"Executing step: {step[:120]}")

        child_steps = self._split_child_steps(step)
        if len(child_steps) > 1:
            log("child_step_created", {"parent_step": step[:160], "children": child_steps})
            for child_step in child_steps:
                self.execute_step(child_step, state)
            return {
                "status": "success",
                "results": [],
                "child_steps": child_steps,
            }

        # ── Content-first routing ─────────────────────────────────────────────
        tool, path = _detect_file_write_step(step)
        if tool and path:
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
        for attempt in range(2):
            try:
                _cli_status("Asking coder to choose the next tool action.")
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

            # ── Parse ─────────────────────────────────────────────────────────
            action_bundle = Dispatcher.parse_llm_json(raw)
            repair_needed = False

            # If parse failed, try once more with the repair prompt
            if action_bundle is None:
                action_bundle = self._repair_action_bundle(step, raw, state, "malformed_json")
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

            log("tool_count", {"step": step[:160], "count": len(tools_to_call), "repair": repair_needed})
            log("tool_routing", {
                "strategy": "json",
                "tools": tools_to_call,
                "repair": repair_needed,
                "action": action_bundle.get("action"),
            })

            contract_ok, contract_error, _ = self._validate_atomic_contract(step, action_bundle, raw, state)
            if not contract_ok:
                error = f"Executor rejected non-atomic response: {contract_error}"
                log("executor_contract_check", {"step": step[:160], "passed": False, "reason": contract_error})
                state.add_tool_result("dispatcher", "error", error)
                return {
                    "status": "error",
                    "results": [{"tool": "dispatcher", "status": "error", "output": error}],
                    "error": error,
                }

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

            if action == "noop" and self._should_reject_noop(step, state):
                attempt_num = state.step_attempts.get(step, 0) + 1
                state.step_attempts[step] = attempt_num
                if attempt_num >= 2:
                    error = f"Executor rejected noop for implementation step after {attempt_num} attempts: {step}"
                    log("executor_invalid_noop", {"step": step[:160], "attempt": attempt_num})
                    state.add_tool_result("dispatcher", "error", error)
                    return {"status": "error", "results": [{"tool": "dispatcher", "status": "error", "output": error}], "error": error}
                error = "Executor rejected noop for implementation step; the coder must produce a tool_call for this step."
                log("executor_invalid_noop", {"step": step[:160], "attempt": attempt_num})
                state.add_tool_result("dispatcher", "error", error)
                continue

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

            # ── Dispatch ──────────────────────────────────────────────────────
            _cli_status(f"Dispatching tool action: {', '.join(tools_to_call) or 'none'}")
            dispatch_result = self.dispatcher.dispatch(action_bundle)

            # ── Store results in state ────────────────────────────────────────
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

        error = f"Executor rejected noop for implementation step after retries: {step}"
        state.add_tool_result("dispatcher", "error", error)
        return {"status": "error", "results": [{"tool": "dispatcher", "status": "error", "output": error}], "error": error}

    def _should_reject_noop(self, step: str, state: RunState) -> bool:
        lowered = (step or "").lower()
        if not any(keyword in lowered for keyword in ("write", "create", "generate", "implement", "build")):
            return False
        return state.step_attempts.get(step, 0) < 2

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
