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


# ── Edit-first prompts (REPLACE semantics) ─────────────────────────────────────
# Used when the step is a targeted replace-style edit to an existing file.
# Injects the ACTUAL current file content (read deterministically via the
# read_file tool, not pulled from stale/trimmed tool_results) so the model
# can copy "old" verbatim instead of inventing text that was never there.
_EDIT_PROMPT = """\
You must edit an EXISTING file by specifying an exact old/new text pair.

File to edit: {path}

CURRENT FULL CONTENT OF THE FILE (copy "old" from here EXACTLY — same
whitespace, same indentation, same line breaks. Do not paraphrase or
reformat it):
{file_content}

Task: {step}

Requirements:
{requirements}

Output ONLY this JSON — no prose, no fences:
{{"action":"tool_call","tools":[{{"name":"edit_file","args":{{"path":"{path}","old":"EXACT TEXT COPIED FROM ABOVE","new":"REPLACEMENT TEXT"}}}}]}}

Rules:
- "old" MUST be a contiguous substring that appears character-for-character in the file content shown above.
- Keep "old" as SHORT as possible while still being unique — a few lines is usually enough. Do not paste the whole file.
- If you are adding something new (not replacing), pick a short unique anchor line as "old" and include that same anchor line at the start or end of "new".

JSON only:"""

_EDIT_REPAIR_PROMPT = """\
Your previous edit_file call FAILED because "old" was not found in the file,
even after whitespace normalization. This means you did not copy it exactly.

File to edit: {path}

CURRENT FULL CONTENT OF THE FILE (this is the ONLY valid source for "old"):
{file_content}

Your previous (failed) "old" value was:
{failed_old}

Task: {step}

Look at the file content above character by character and pick a short
substring that ACTUALLY EXISTS in it. Output ONLY this JSON:
{{"action":"tool_call","tools":[{{"name":"edit_file","args":{{"path":"{path}","old":"EXACT TEXT COPIED FROM FILE ABOVE","new":"REPLACEMENT TEXT"}}}}]}}

JSON only:"""


# ── Additive-append prompt (no replace anchor needed) ──────────────────────────
# Used for tasks that ADD new capability (new function, new listener, new
# block) rather than modify existing text. There's no reliable "old" anchor
# for these — forcing old/new replace semantics on them is what causes
# repeated "text not found" failures on tasks like "add dark mode toggle".
# append has no search step at all, so it structurally cannot fail that way.
_ADDITIVE_APPEND_PROMPT = """\
You are adding new functionality to an existing file. Output ONLY the new
code to append to the end of the file. Do not repeat existing code. No
prose, no markdown fences — just the new code block.

File: {path}

CURRENT FULL CONTENT (for context — do not repeat this, just add what's new):
{file_content}

Task: {step}

Requirements:
{requirements}

Output only the new code to append:"""


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
_LARGE_CONTENT_EXTS = (".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".svg")

# File write tool names
_WRITE_TOOLS = {"write_file", "create_file"}

# Keywords that signal a targeted edit rather than a full file (re)write.
# When the target path already exists on disk, these force edit_file so a
# request like "add a hero image" can't degrade into a full-file rewrite.
_EDIT_KEYWORDS = ("edit", "update", "add", "modify", "insert", "append", "change", "fix", "remove")

# Additive vs. replace sub-classification (both are subsets of edit-intent).
# Additive tasks have no reliable "old" text to anchor a replace on — they
# introduce something that doesn't exist yet. Replace-specific verbs win
# when both appear (e.g. "change the add-to-cart button" is a replace).
_ADDITIVE_KEYWORDS = ("add", "insert", "implement", "introduce")
_REPLACE_KEYWORDS = ("change", "replace", "update", "fix", "modify", "remove", "rename")


def _detect_file_write_step(step: str) -> tuple[str | None, str | None]:
    """
    If this step is clearly a file write/edit operation, return (tool_name, path).
    Otherwise return (None, None).

    Detects patterns like:
      - "Write index.html with ..."
      - "Create codi.html using create_file ..."
      - "Use write_file to save styles.css ..."
      - "Edit index.html to add a hero image"     -> edit_file (if file exists)
      - "Add a hero image to index.html"          -> edit_file (if file exists)

    Edit-intent language on a file that already exists on disk always takes
    priority over create/write detection. This is what stops "add a hero
    image" from being routed to a full-file write/rewrite.
    """
    step_lower = step.lower()

    # Extract file path — look for known extensions
    ext_pattern = r"([A-Za-z0-9_./\\-]+\.(?:html|css|js|ts|jsx|tsx|py|md|json|txt|svg|sh))"
    match = re.search(ext_pattern, step)
    if not match:
        return None, None

    path = match.group(1).strip("'\"` ")

    # ── Edit intent + file already exists on disk → force edit_file ──────────
    if any(kw in step_lower for kw in _EDIT_KEYWORDS):
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        abs_path = path if os.path.isabs(path) else os.path.join(working_dir, path)
        if os.path.exists(abs_path):
            return "edit_file", path

    # ── Fall back to create/write detection ───────────────────────────────────
    write_keywords = ("write", "create", "save", "generate", "produce", "output")
    if not any(kw in step_lower for kw in write_keywords):
        return None, None

    # Determine which write tool to use
    tool = "create_file" if "create" in step_lower else "write_file"

    return tool, path


def _is_additive_edit(step: str) -> bool:
    """
    True if the step is asking to ADD new capability rather than modify
    existing text. Additive steps have no reliable 'old' anchor to replace —
    forcing old/new replace semantics on them causes the model to invent
    text that was never in the file, producing repeated "text not found"
    failures (e.g. "add dark/light mode toggle" — there's no existing code
    to anchor a replace on).

    A replace-specific verb co-occurring with an additive verb wins, since
    that usually means an existing thing is being swapped out
    (e.g. "change the add-to-cart button color" is a replace, not additive).
    """
    step_lower = step.lower()
    has_additive = any(kw in step_lower for kw in _ADDITIVE_KEYWORDS)
    has_replace = any(kw in step_lower for kw in _REPLACE_KEYWORDS)
    return has_additive and not has_replace


def _is_large_content_step(step: str, path: str, tool: str | None = None) -> bool:
    """
    True if the step is likely to require more content than a 7B model
    can safely JSON-serialize without truncating.

    IMPORTANT: edit_file must NEVER go through content-first mode. Content-
    first only knows how to dump a full replacement file — routing an edit
    through it is exactly how "add a hero image" turned into a full
    index.html rewrite. So if the detected tool is edit_file, this always
    returns False regardless of the other heuristics below. (In practice
    edit_file steps are now routed to _execute_edit_first before this
    function is ever consulted — this guard stays as defense in depth.)

    Heuristics (write_file/create_file only):
      1. All HTML files — always large (even "simple.html" needs boilerplate)
      2. CSS/JS/TS with design keywords
      3. CSS/JS/TS step contains "with" — means caller described content
      4. Step is longer than 60 chars — enough description = enough content
    """
    if tool == "edit_file":
        return False

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

    # ── Content-first strategy (write_file / create_file only) ────────────────

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

        NOTE: This path must only ever be entered with tool in
        {"write_file", "create_file"}. _is_large_content_step guarantees this
        by always returning False when tool == "edit_file", and edit_file
        steps are routed to _execute_edit_first before this is ever reached.
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

    # ── Edit-first strategy (edit_file — replace and additive) ─────────────────

    def _execute_edit_first(self, step: str, path: str, state: RunState) -> dict:
        """
        Dedicated flow for edit_file. Reads the real file content deterministically
        (never trusts the LLM to have it right from stale/trimmed tool_results).

        Branches by intent:
          - Additive steps ("add", "insert", "implement", "introduce" without a
            replace verb) go straight to _execute_additive_append — there is no
            reliable "old" anchor for genuinely new code, so replace semantics
            just cause the model to invent text that was never in the file.
          - Replace steps go through _EDIT_PROMPT with the real file content
            injected, so the model can copy "old" verbatim. One automatic
            repair retry if the first old/new pair doesn't match. If that
            repair ALSO fails with "text not found", falls back to additive
            append rather than exhausting all iterations on a dead end.
        """
        read_handler = self.registry.get("read_file")
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        abs_path = path if os.path.isabs(path) else os.path.join(working_dir, path)

        try:
            file_content = read_handler({"path": path}) if read_handler else ""
        except Exception as e:
            file_content = ""
            log("executor_edit_first_read_error", {"path": path, "error": str(e)})

        if not file_content or file_content.startswith("ERROR"):
            error = f"edit_file requested but could not read current content of {path}: {file_content}"
            log("executor_edit_first_no_content", {"path": path})
            state.add_tool_result("edit_file", "error", error)
            return {
                "status": "error",
                "results": [{"tool": "edit_file", "status": "error", "output": error}],
                "error": error,
            }

        # ── Additive tasks skip replace semantics entirely ─────────────────────
        if _is_additive_edit(step):
            log("executor_edit_additive_route", {"path": path, "step": step[:120]})
            return self._execute_additive_append(step, path, file_content, state)

        prompt = _EDIT_PROMPT.format(
            path=path,
            file_content=wrap_prompt_data(file_content, path=path),
            step=step,
            requirements=state.requirements.as_prompt_block(),
        )

        result = self._run_edit_attempt(prompt, path, state, label="coder_edit_first")

        # ── One repair retry if the old/new pair didn't match the real file ────
        if self._edit_failed_text_not_found(result):
            failed_args = self._last_edit_args(result)
            failed_old = (failed_args or {}).get("old", "")[:300]

            repair_prompt = _EDIT_REPAIR_PROMPT.format(
                path=path,
                file_content=wrap_prompt_data(file_content, path=path),
                failed_old=failed_old,
                step=step,
            )
            log("executor_edit_repair", {"path": path, "failed_old": failed_old[:120]})
            result = self._run_edit_attempt(repair_prompt, path, state, label="coder_edit_repair")

        # ── Last resort: replace failed twice → fall back to guaranteed-anchor
        # append instead of burning the rest of the iteration budget on a
        # replace strategy that has already failed twice. append cannot
        # produce "text not found" since it has no search step at all.
        if self._edit_failed_text_not_found(result):
            log("executor_edit_fallback_append", {"path": path, "step": step[:120]})
            return self._execute_additive_append(step, path, file_content, state)

        return result

    def _execute_additive_append(
        self, step: str, path: str, file_content: str, state: RunState
    ) -> dict:
        """
        For additive edits: ask the model for ONLY the new code to add
        (no old/new pair needed), then append it via edit_file's append
        arg. The anchor problem disappears entirely because append doesn't
        search for anything — it can never produce 'text not found'.
        """
        prompt = _ADDITIVE_APPEND_PROMPT.format(
            path=path,
            file_content=wrap_prompt_data(file_content, path=path),
            step=step,
            requirements=state.requirements.as_prompt_block(),
        )

        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            new_code = _strip_fences(resp.content.strip())
        except Exception as e:
            error = f"Coder LLM error (additive append): {e}"
            log("executor_additive_append_error", {"error": str(e)})
            state.add_tool_result("edit_file", "error", error)
            return {
                "status": "error",
                "results": [{"tool": "edit_file", "status": "error", "output": error}],
                "error": error,
            }

        state.record_llm("coder_additive_append", new_code[:200])
        log("executor_additive_append_raw", {"path": path, "content_len": len(new_code)})

        if not new_code:
            error = "Coder returned empty content for additive append."
            log("executor_additive_append_empty", {"path": path})
            state.add_tool_result("edit_file", "error", error)
            return {
                "status": "error",
                "results": [{"tool": "edit_file", "status": "error", "output": error}],
                "error": error,
            }

        action_bundle = {
            "action": "tool_call",
            "tools": [{"name": "edit_file", "args": {"path": path, "append": new_code}}],
        }

        violation = self._framework_violation(state, action_bundle)
        if violation:
            log("executor_framework_violation", {
                "tool": "edit_file",
                "path": path,
                "reason": violation,
            })
            state.add_tool_result("edit_file", "error", violation)
            return {
                "status": "error",
                "results": [{"tool": "edit_file", "status": "error", "output": violation}],
                "error": violation,
            }

        dispatch_result = self.dispatcher.dispatch(action_bundle)
        if dispatch_result.get("signal") in ("noop", "done"):
            signal = dispatch_result.get("signal", "noop")
            state.add_tool_result("dispatcher", "ok", signal)
            log("executor_dispatch_signal", {
                "signal": signal,
                "step": step[:160],
                "action": action_bundle.get("action"),
            })

        for r in dispatch_result.get("results", []):
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return dispatch_result

    def _run_edit_attempt(self, prompt: str, path: str, state: RunState, label: str) -> dict:
        """Single LLM call + dispatch attempt for a replace-style edit_file action.
        Shared by the initial edit-first call and the one-shot repair retry."""
        try:
            resp = self.llm.invoke([self._sys(), HumanMessage(content=prompt)])
            raw = resp.content.strip()
        except Exception as e:
            error = f"Coder LLM error (edit-first): {e}"
            log("executor_edit_first_llm_error", {"error": str(e)})
            state.add_tool_result("edit_file", "error", error)
            return {
                "status": "error",
                "results": [{"tool": "edit_file", "status": "error", "output": error}],
                "error": error,
            }

        state.record_llm(label, raw)

        action_bundle = Dispatcher.parse_llm_json(raw)
        if action_bundle is None:
            action_bundle = self._repair_action_bundle(f"edit {path}", raw, state)

        if action_bundle is None:
            error = f"Coder edit-first output was not valid JSON: {raw[:200]}"
            state.add_tool_result("edit_file", "error", error)
            return {
                "status": "error",
                "results": [{"tool": "edit_file", "status": "error", "output": error}],
                "error": error,
            }

        violation = self._framework_violation(state, action_bundle)
        if violation:
            state.add_tool_result("edit_file", "error", violation)
            return {
                "status": "error",
                "results": [{"tool": "edit_file", "status": "error", "output": violation}],
                "error": violation,
            }

        dispatch_result = self.dispatcher.dispatch(action_bundle)
        for r in dispatch_result.get("results", []):
            state.add_tool_result(r["tool"], r["status"], r["output"])

        return dispatch_result

    @staticmethod
    def _edit_failed_text_not_found(result: dict) -> bool:
        for r in result.get("results", []):
            if r.get("status") == "error" and "text not found" in (r.get("output") or ""):
                return True
        return False

    @staticmethod
    def _last_edit_args(result: dict) -> dict | None:
        for r in result.get("results", []):
            if r.get("tool") == "edit_file":
                return r.get("args") or {}
        return None

    # ── Main entry point ──────────────────────────────────────────────────────

    def execute_step(self, step: str, state: RunState) -> dict:
        """
        Translate a step description into a tool call and execute it.
        Returns the dispatcher result dict.

        Routing order:
          1. edit_file steps (existing file + edit-intent language) always go
             through the edit-first flow. Within that flow, additive steps
             (add/insert/implement/introduce, no replace verb) skip replace
             semantics entirely and go straight to append; replace steps get
             the real file content injected so the model can copy "old"
             verbatim, with one repair retry and an append fallback if
             replace still can't find a match.
          2. write_file/create_file steps with large expected output
             (HTML, styled CSS/JS, etc.) go through content-first — no JSON
             wrapper, full output budget goes to file content.
          3. Everything else goes through the standard JSON tool-call path.
        """
        from context_trimmer import trim_tool_output

        # ── Edit-first routing — always for detected edit_file steps ──────────
        tool, path = _detect_file_write_step(step)
        if tool == "edit_file" and path:
            log("tool_routing", {
                "strategy": "edit_first",
                "tool": tool,
                "path": path[:80],
                "additive": _is_additive_edit(step),
            })
            return self._execute_edit_first(step, path, state)

        # ── Content-first routing (write_file/create_file only) ───────────────
        if tool and path and _is_large_content_step(step, path, tool):
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
                "\n".join(
                    trim_tool_output(o, max_tokens=120)
                    for o in state.recent_tool_outputs(8)
                ) or "(none yet)"
            ),
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

        # ── Parse ─────────────────────────────────────────────────────────────
        action_bundle = Dispatcher.parse_llm_json(raw)
        repair_needed = False

        # If parse failed, try once more with the repair prompt
        if action_bundle is None:
            action_bundle = self._repair_action_bundle(step, raw, state)
            repair_needed = True

        # If STILL None and this looks like a truncated file-write, switch
        # to content-first as a last resort — but ONLY for write/create,
        # never for edit_file (an edit step must never silently become a
        # full-file rewrite just because JSON parsing failed twice).
        if action_bundle is None:
            tool_fb, path_fb = _detect_file_write_step(step)
            if tool_fb in _WRITE_TOOLS and path_fb:
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
`````html ... ```
````css  ... ```
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