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

import difflib
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
{{"action":"tool_call","reason":"short justification","tools":[{{"name":"TOOL_NAME","args":{{ARGS}}}}]}}

Or if nothing to do: {{"action":"noop"}}

JSON only:"""


# ── Content-first prompt ──────────────────────────────────────────────────────
_CONTENT_PROMPT = """\
You are writing the full content of a file. Output ONLY the raw file content.
No JSON. No markdown fences. No explanation. Just the file content itself.

File to write: {path}
Current implementation step: {step}

Original user task (this is the source of the required features and quality):
{user_task}

Requirements:
{requirements}

Context from project:
{context}

Output the complete file content now:"""


# ── Edit-first prompts (REPLACE semantics) ─────────────────────────────────────
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
{{"action":"tool_call","reason":"short justification","tools":[{{"name":"edit_file","args":{{"path":"{path}","old":"EXACT TEXT COPIED FROM ABOVE","new":"REPLACEMENT TEXT"}}}}]}}

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
{{"action":"tool_call","reason":"short justification","tools":[{{"name":"edit_file","args":{{"path":"{path}","old":"EXACT TEXT COPIED FROM FILE ABOVE","new":"REPLACEMENT TEXT"}}}}]}}

JSON only:"""


# ── Line-range edit prompt (surgical, no exact-text reproduction needed) ──────
_LINE_EDIT_PROMPT = """\
You are surgically editing an EXISTING file using LINE NUMBERS — no need to
copy exact text.

File to edit: {path}

NUMBERED CONTENT (format: "LINE_NUMBER<TAB>code"):
{numbered_content}

Task: {step}

Requirements:
{requirements}

Output ONLY JSON using ONE of these forms — no prose, no fences:

To replace a contiguous range of lines:
{{"action":"tool_call","reason":"short justification","tools":[{{"name":"edit_file","args":{{"path":"{path}","replace_lines":{{"start":N,"end":M,"content":"new code here"}}}}}}]}}

To delete a contiguous range of lines:
{{"action":"tool_call","reason":"short justification","tools":[{{"name":"edit_file","args":{{"path":"{path}","delete_lines":{{"start":N,"end":M}}}}}}]}}

To insert new code before line N (use the line number that should come
immediately AFTER the inserted code; to insert at the very end of the file,
use line number = total_lines + 1):
{{"action":"tool_call","reason":"short justification","tools":[{{"name":"edit_file","args":{{"path":"{path}","insert_at_line":{{"line":N,"content":"new code here"}}}}}}]}}

Rules:
- Line numbers MUST come directly from the numbered content above — never guess or estimate them.
- Keep the edited range as tight as possible — only the lines that actually change.
- Preserve the indentation style of the surrounding code in "content".
- "content" should NOT include line-number prefixes — those are for your reference only, not part of the file.

JSON only:"""

_LINE_EDIT_REPAIR_PROMPT = """\
Your previous line-range edit FAILED: {failure_reason}

File to edit: {path}

CURRENT NUMBERED CONTENT (re-read carefully — your line numbers were wrong
or out of bounds):
{numbered_content}

Task: {step}

Output ONLY corrected JSON using one of the replace_lines / delete_lines /
insert_at_line forms shown before, with line numbers copied exactly from the
numbered content above.

JSON only:"""


# ── Additive-append prompt (no replace anchor needed) ──────────────────────────
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
_LARGE_CONTENT_TRIGGERS = (
    "modern", "sleek", "design", "website", "webpage", "landing", "dashboard",
    "terminal", "portal", "app", "full", "complete", "entire", "whole",
    "beautiful", "styled", "animated", "responsive", "interactive",
    "dark", "light", "theme", "glass", "gradient", "shadow", "effect",
    "layout", "component", "feature", "section", "header", "footer",
    "nav", "card", "modal", "form", "button", "style", "color", "font",
)

_LARGE_CONTENT_EXTS = (".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".svg")

_WRITE_TOOLS = {"write_file", "create_file"}

_EDIT_KEYWORDS = ("edit", "update", "add", "modify", "insert", "append", "change", "fix", "remove", "debug", "style")
_IMPLEMENTATION_VERBS = _EDIT_KEYWORDS + ("create", "implement", "replace", "write")

_ADDITIVE_KEYWORDS = ("add", "insert", "implement", "introduce")
_REPLACE_KEYWORDS = ("change", "replace", "update", "fix", "modify", "remove", "rename", "style", "debug")

# Steps that reference specific lines, blocks, or named code units are routed
# to the line-range surgical edit path instead of text-match old/new — this
# avoids requiring the coder LLM to reproduce large/whitespace-sensitive
# snippets verbatim, which was the main source of "text not found" failures.
_LINE_EDIT_KEYWORDS = (
    "line ", "lines ", "block", "function", "method", "def ", "class ",
    "section between", "between line", "from line",
)

# Directories skipped when scanning for fuzzy-match candidates
_SKIP_DIRS = {".git", "node_modules", "__pycache__", "venv", "dist", "build", "chroma_db"}


def _step_requires_mutation(step: str) -> bool:
    lowered = (step or "").lower()
    return any(re.search(rf"\b{re.escape(verb)}\b", lowered) for verb in _IMPLEMENTATION_VERBS)


def _explicit_full_rewrite_requested(user_input: str) -> bool:
    lowered = (user_input or "").lower()
    phrases = (
        "rewrite the entire",
        "replace the entire",
        "overwrite",
        "overide",       # common typo for "override"/"overwrite"
        "override",
        "regenerate the entire",
        "replace all",
        "replace any and all",
        "override any and all",
        "overide any and all",
        "delete all old",
        "remove all old",
        "start from scratch",
        "from scratch",
        "wipe and",
        "clean slate",
    )
    return any(phrase in lowered for phrase in phrases)


def _path_is_protected(path: str, state: RunState) -> bool:
    protected = getattr(state.requirements, "protected_files", []) or []
    normalized = os.path.normcase(os.path.basename(path))
    return any(normalized == os.path.normcase(os.path.basename(item)) for item in protected)


def _resolve_existing_path(path: str) -> str | None:
    """
    If `path` doesn't exist on disk, look for the closest-matching existing
    filename in the working directory tree and return that instead.

    This exists because the coder LLM routinely typos filenames it invents
    from memory of the task description rather than the actual directory
    listing (e.g. user says "scripts.js", real file is "script.js", model
    faithfully repeats the user's typo). Without this, edit_file loops on
    "file does not exist" and burns iterations instead of just fixing the
    obvious one-character mismatch.

    Returns None if the path already exists, or if no reasonably close
    match was found (caller should then treat it as a genuine missing file).
    """
    working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
    abs_path = path if os.path.isabs(path) else os.path.join(working_dir, path)

    if os.path.exists(abs_path):
        return None  # already valid, no resolution needed

    target_name = os.path.basename(path)
    target_ext = os.path.splitext(target_name)[1].lower()

    candidates = []
    try:
        for root, dirs, files in os.walk(working_dir):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for fname in files:
                if os.path.splitext(fname)[1].lower() == target_ext:
                    candidates.append(os.path.relpath(os.path.join(root, fname), working_dir))
    except Exception as e:
        log("executor_fuzzy_resolve_error", {"path": path, "error": str(e)})
        return None

    if not candidates:
        return None

    matches = difflib.get_close_matches(target_name, [os.path.basename(c) for c in candidates], n=1, cutoff=0.6)
    if not matches:
        return None

    matched_name = matches[0]
    for c in candidates:
        if os.path.basename(c) == matched_name:
            log("executor_fuzzy_resolve", {"requested": path, "resolved": c})
            return c

    return None


def _detect_file_write_step(step: str) -> tuple[str | None, str | None]:
    """
    If this step is clearly a file write/edit operation, return (tool_name, path).
    Otherwise return (None, None).

    Edit-intent language on a file that already exists on disk always takes
    priority over create/write detection. If the exact path doesn't exist,
    a fuzzy match against real files in the working dir is attempted before
    giving up — this is what stops a typo'd "scripts.js" from either
    silently falling through to the standard JSON path (where the LLM can
    still freely pick edit_file against the bad path) or looping forever
    on "file does not exist".
    """
    step_lower = step.lower()

    ext_pattern = r"([A-Za-z0-9_./\\-]+\.(?:html|css|js|ts|jsx|tsx|py|md|json|txt|svg|sh))"
    match = re.search(ext_pattern, step)
    if not match:
        return None, None

    path = match.group(1).strip("'\"` ")

    # Explicit creation wins over incidental edit words such as the "style"
    # inside a filename like styles.css.  Without this, a step such as
    # "Create new files: index.html, styles.css, script.js" tried to edit an
    # existing file instead of creating the missing one.
    if re.search(r"\b(create|write|generate|produce)\b", step_lower):
        tool = "create_file" if "create" in step_lower else "write_file"
        return tool, path

    write_keywords = ("write", "create", "save", "generate", "produce", "output")
    has_write_signal = any(kw in step_lower for kw in write_keywords)

    if any(re.search(rf"\b{re.escape(kw)}\b", step_lower) for kw in _EDIT_KEYWORDS):
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        abs_path = path if os.path.isabs(path) else os.path.join(working_dir, path)

        if os.path.exists(abs_path):
            return "edit_file", path

        resolved = _resolve_existing_path(path)
        if resolved:
            return "edit_file", resolved

        # File doesn't exist and no close match found. If the step ALSO
        # carries a write/create signal (e.g. "style script.js and
        # styles.css for the portfolio" — "style" is an edit keyword but
        # the file is brand new and the plan clearly means create it),
        # fall through to create/write detection instead of hard-failing.
        # This is what previously caused every non-first file in a
        # multi-file build plan to dead-end as "edit_file_missing" purely
        # because its step text happened to contain a generic word like
        # "style", "add", or "update" alongside the (nonexistent) filename.
        if has_write_signal:
            log("executor_edit_keyword_fallback_to_create", {
                "path": path, "step": step[:160],
            })
        else:
            # Genuinely edit-only language (no create/write signal at all)
            # against a file that doesn't exist and has no close match —
            # this really is an ambiguous/missing-target case. Signal the
            # caller explicitly so execute_step can fail fast with a clear
            # message instead of burning iterations.
            return "edit_file_missing", path

    if not has_write_signal:
        return None, None

    tool = "create_file" if "create" in step_lower else "write_file"

    return tool, path


def _is_additive_edit(step: str) -> bool:
    step_lower = step.lower()
    has_additive = any(kw in step_lower for kw in _ADDITIVE_KEYWORDS)
    has_replace = any(kw in step_lower for kw in _REPLACE_KEYWORDS)
    return has_additive and not has_replace


def _is_line_range_edit(step: str) -> bool:
    """
    True when the step explicitly references line numbers, blocks, or named
    code units (functions/classes/methods) — signals that a line-range
    surgical edit (replace_lines / delete_lines / insert_at_line) is more
    reliable than asking the coder LLM to reproduce exact text verbatim.
    """
    step_lower = step.lower()
    if not any(kw in step_lower for kw in _LINE_EDIT_KEYWORDS):
        return False
    # Require at least one of: an explicit "line" reference, or a named
    # code-unit reference — avoids over-triggering on generic words like
    # "block" used in a CSS/HTML sense ("style the hero block").
    return bool(
        re.search(r"\bline(s)?\b", step_lower)
        or re.search(r"\bfunction\b|\bmethod\b|\bclass\b|\bdef\b", step_lower)
    )


def _is_large_content_step(step: str, path: str, tool: str | None = None) -> bool:
    if tool in ("edit_file", "edit_file_missing"):
        return False

    ext = os.path.splitext(path)[1].lower()
    step_lower = step.lower()

    if ext not in _LARGE_CONTENT_EXTS:
        return False

    if ext == ".html":
        return True

    if any(trigger in step_lower for trigger in _LARGE_CONTENT_TRIGGERS):
        return True

    if " with " in step_lower:
        return True

    if len(step) > 60:
        return True

    return False


class Executor:
    def __init__(self, registry: ToolRegistry):
        self.registry   = registry
        self.dispatcher = Dispatcher(registry)
        self.llm        = get_coder_llm()

    def _sys(self) -> SystemMessage:
        return SystemMessage(content=executor_system_prompt(
            tool_names=self.registry.list_names(),
        ))

    def _log_justification(self, step: str, action_bundle: dict | None) -> None:
        """Record the executor's stated reason for its tool choice.

        Every tool_call bundle is required (by the executor system prompt) to
        carry a "reason" field. This makes the ReAct "why this tool, why not
        another" requirement auditable via /logs without introducing any new
        approval or blocking machinery — dispatch() already ignores unknown
        top-level keys, so this is purely observational.
        """
        if not isinstance(action_bundle, dict):
            return
        reason = action_bundle.get("reason")
        if reason:
            log("tool_justification", {"step": step[:160], "reason": str(reason)[:240]})
        elif action_bundle.get("action") in ("tool_call", "parallel"):
            log("tool_justification_missing", {"step": step[:160]})

    def _repair_action_bundle(self, step: str, raw: str, state: RunState) -> dict | None:
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
        log("executor_content_first", {"tool": tool, "path": path, "step": step[:80]})

        context_str = trim_tool_output(
            "\n".join(state.recent_tool_outputs(4)) or "(none yet)",
            max_tokens=800,
        )
        # Include only the named source file for a one-function test step.
        # This keeps the prompt compact while giving the model the behavior it
        # must test, instead of relying on broad project context.
        source_match = re.search(r"\bin\s+([A-Za-z0-9_./\\-]+\.py)\b", step)
        if source_match:
            source_path = source_match.group(1)
            root = os.environ.get("CODI_WORKING_DIR", os.getcwd())
            absolute_source = source_path if os.path.isabs(source_path) else os.path.join(root, source_path)
            try:
                with open(absolute_source, "r", encoding="utf-8", errors="replace") as handle:
                    context_str += f"\n\nSource under test ({source_path}):\n" + handle.read()[:16000]
            except OSError:
                pass
        prompt = _CONTENT_PROMPT.format(
            path=path,
            step=step,
            user_task=state.user_input[:12000],
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
            "reason": f"content-first write of {path} (fresh or large file, no exact-text edit possible)",
            "tools":  [{"name": tool, "args": {"path": path, "content": content}}],
        }
        if tool == "create_file":
            root = os.environ.get("CODI_WORKING_DIR", os.getcwd())
            absolute_target = path if os.path.isabs(path) else os.path.join(root, path)
            if os.path.exists(absolute_target):
                try:
                    with open(absolute_target, "r", encoding="utf-8", errors="replace") as handle:
                        existing = handle.read()
                    action_bundle["tools"] = [{
                        "name": "edit_file",
                        "args": {"path": path, "old": existing, "new": existing.rstrip() + "\n\n" + content},
                    }]
                    tool = "edit_file"
                    log("executor_test_content_append", {"path": path, "step": step[:160]})
                except OSError as exc:
                    error = f"Unable to read existing test file '{path}': {exc}"
                    state.add_tool_result("edit_file", "error", error)
                    return {"status": "error", "results": [{"tool": "edit_file", "status": "error", "output": error}], "error": error}

        self._log_justification(step, action_bundle)

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

        dispatch_result = self.dispatcher.dispatch(action_bundle, knowledge=state.knowledge)
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

    # ── Line-range edit strategy (surgical, uses read_file_numbered) ──────────

    def _execute_line_edit(self, step: str, path: str, state: RunState) -> dict:
        """
        Surgically edit a file by line number instead of exact-text matching.

        Reads the file with line numbers attached (read_file_numbered), asks
        the coder LLM to specify a replace_lines / delete_lines / insert_at_line
        operation using those real line numbers, then dispatches it. On an
        out-of-bounds / malformed line-range failure, re-reads the file (in
        case a previous attempt already partially changed it) and retries once
        with a repair prompt before giving up.
        """
        numbered_handler = self.registry.get("read_file_numbered")

        def _read_numbered() -> str:
            try:
                return numbered_handler({"path": path}) if numbered_handler else ""
            except Exception as e:
                log("executor_line_edit_read_error", {"path": path, "error": str(e)})
                return ""

        numbered = _read_numbered()
        if not numbered or numbered.startswith("ERROR"):
            error = f"line edit requested but could not read numbered content of {path}: {numbered}"
            log("executor_line_edit_no_content", {"path": path})
            state.add_tool_result("edit_file", "error", error)
            return {
                "status": "error",
                "results": [{"tool": "edit_file", "status": "error", "output": error}],
                "error": error,
            }

        prompt = _LINE_EDIT_PROMPT.format(
            path=path,
            numbered_content=wrap_prompt_data(numbered, path=path),
            step=step,
            requirements=state.requirements.as_prompt_block(),
        )

        result = self._run_edit_attempt(prompt, path, state, label="coder_line_edit")

        if self._edit_failed_line_range(result):
            failure_reason = self._last_edit_error(result) or "line range was invalid or out of bounds"
            fresh_numbered = _read_numbered()
            repair_prompt = _LINE_EDIT_REPAIR_PROMPT.format(
                failure_reason=failure_reason,
                path=path,
                numbered_content=wrap_prompt_data(fresh_numbered or numbered, path=path),
                step=step,
            )
            log("executor_line_edit_repair", {"path": path, "reason": failure_reason[:120]})
            result = self._run_edit_attempt(repair_prompt, path, state, label="coder_line_edit_repair")

        if self._edit_failed_line_range(result):
            # Fall back to the text-match edit path rather than failing outright —
            # gives the step one more realistic chance to succeed.
            log("executor_line_edit_fallback_textmatch", {"path": path, "step": step[:120]})
            return self._execute_edit_first(step, path, state, skip_line_route=True)

        return result

    # ── Edit-first strategy (edit_file — replace and additive) ─────────────────

    def _execute_edit_first(self, step: str, path: str, state: RunState, skip_line_route: bool = False) -> dict:
        if not skip_line_route and _is_line_range_edit(step):
            log("executor_edit_line_route", {"path": path, "step": step[:120]})
            return self._execute_line_edit(step, path, state)

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

        if self._edit_failed_text_not_found(result):
            log("executor_edit_fallback_append", {"path": path, "step": step[:120]})
            return self._execute_additive_append(step, path, file_content, state)

        return result

    def _execute_edit_missing_file(self, step: str, path: str, state: RunState) -> dict:
        """
        The step has edit-intent language but the referenced file doesn't
        exist and no close fuzzy match was found in the working directory.
        Fail immediately and clearly instead of letting the standard JSON
        path or repeated LLM calls loop on "file does not exist" for
        several iterations (as happened with a typo'd "scripts.js" against
        a directory that only had "script.js").
        """
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        error = (
            f"Step references '{path}' for editing, but that file does not exist "
            f"in {working_dir} and no similarly-named file was found. "
            f"If this file should be created, say so explicitly (e.g. 'create {path}')."
        )
        log("executor_edit_missing_file", {"path": path, "step": step[:120]})
        state.add_tool_result("edit_file", "error", error)
        return {
            "status": "error",
            "results": [{"tool": "edit_file", "status": "error", "output": error}],
            "error": error,
        }

    def _execute_additive_append(
        self, step: str, path: str, file_content: str, state: RunState
    ) -> dict:
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
            "reason": f"additive append to {path} (step is additive-only, no existing text to replace)",
            "tools": [{"name": "edit_file", "args": {"path": path, "append": new_code}}],
        }

        self._log_justification(step, action_bundle)

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

        self._log_justification(path, action_bundle)

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
    def _edit_failed_line_range(result: dict) -> bool:
        """True when an edit_file line-range op failed (bad/out-of-bounds range,
        missing keys, or a caught ValueError surfaced as 'ERROR editing ...')."""
        for r in result.get("results", []):
            if r.get("tool") != "edit_file" or r.get("status") != "error":
                continue
            output = r.get("output") or ""
            if any(marker in output for marker in (
                "out of bounds", "replace_lines requires", "delete_lines requires",
                "insert_at_line requires", "must be integer", "ERROR editing",
            )):
                return True
        return False

    @staticmethod
    def _last_edit_args(result: dict) -> dict | None:
        for r in result.get("results", []):
            if r.get("tool") == "edit_file":
                return r.get("args") or {}
        return None

    @staticmethod
    def _last_edit_error(result: dict) -> str | None:
        for r in result.get("results", []):
            if r.get("tool") == "edit_file" and r.get("status") == "error":
                return (r.get("output") or "")[:300]
        return None

    # ── Main entry point ──────────────────────────────────────────────────────

    def execute_step(self, step: str, state: RunState) -> dict:
        from context_trimmer import trim_tool_output

        # ── Edit routing — always for detected edit_file / edit_file_missing ───
        tool, path = _detect_file_write_step(step)

        if tool == "edit_file_missing" and path:
            log("tool_routing", {
                "strategy": "edit_missing_file",
                "path": path[:80],
            })
            return self._execute_edit_missing_file(step, path, state)

        if tool == "edit_file" and path:
            log("tool_routing", {
                "strategy": "edit_first",
                "tool": tool,
                "path": path[:80],
                "additive": _is_additive_edit(step),
                "line_range": _is_line_range_edit(step),
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

        action_bundle = Dispatcher.parse_llm_json(raw)
        repair_needed = False

        if action_bundle is None:
            action_bundle = self._repair_action_bundle(step, raw, state)
            repair_needed = True

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

        self._log_justification(step, action_bundle)

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

        # ── Guard: if the LLM chose edit_file against a path that doesn't
        # actually exist (bypassing the earlier detection), resolve or reject
        # it here too, so the standard JSON path can't reintroduce the same
        # "file does not exist" looping bug through a different door.
        for t in action_bundle.get("tools", []):
            if not isinstance(t, dict):
                continue
            tool_name = t.get("name")
            args = t.get("args") or {}
            target_path = str(args.get("path", ""))
            if tool_name in _WRITE_TOOLS and target_path:
                working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
                absolute_target = target_path if os.path.isabs(target_path) else os.path.join(working_dir, target_path)
                if _path_is_protected(target_path, state):
                    error = (
                        f"Protected source file '{target_path}' cannot be changed for this task. "
                        "Read it to understand behavior, then create or edit the requested test file instead."
                    )
                    state.add_tool_result(tool_name, "error", error)
                    log("executor_protected_file_block", {"path": target_path, "step": step[:160]})
                    return {"status": "error", "results": [{"tool": tool_name, "status": "error", "output": error}], "error": error}
                # Models often plan to create a test file without accounting
                # for a pre-existing placeholder.  Preserve the safe
                # create_file contract, but transparently turn this specific
                # recoverable case into an exact edit of the existing file.
                # This avoids spending several repair iterations on the same
                # predictable "already exists" failure.
                if tool_name == "create_file" and os.path.exists(absolute_target):
                    content = args.get("content")
                    if isinstance(content, str) and content.strip():
                        try:
                            with open(absolute_target, "r", encoding="utf-8", errors="replace") as handle:
                                existing_content = handle.read()
                        except OSError as exc:
                            error = f"Unable to read existing file '{target_path}' before editing: {exc}"
                            state.add_tool_result("edit_file", "error", error)
                            return {"status": "error", "results": [{"tool": "edit_file", "status": "error", "output": error}], "error": error}
                        t["name"] = "edit_file"
                        t["args"] = {"path": target_path, "old": existing_content, "new": content}
                        log("executor_create_existing_converted_to_edit", {"path": target_path, "step": step[:160]})
                        continue
                if os.path.exists(absolute_target) and not _explicit_full_rewrite_requested(state.user_input):
                    # Instead of dead-ending the step with a terminal error
                    # (which used to burn all repair attempts on a step that
                    # never got a chance to actually change anything), convert
                    # this into a real edit: read the current content and turn
                    # the write into an append/merge via edit_file, the same
                    # recoverable-conversion pattern already used above for
                    # create_file-against-existing. This keeps the "don't
                    # silently clobber existing files" safety property while
                    # still making forward progress on the step.
                    content = args.get("content")
                    if tool_name in _WRITE_TOOLS and isinstance(content, str) and content.strip():
                        try:
                            with open(absolute_target, "r", encoding="utf-8", errors="replace") as handle:
                                existing_content = handle.read()
                        except OSError as exc:
                            error = f"Unable to read existing file '{target_path}' before editing: {exc}"
                            state.add_tool_result("edit_file", "error", error)
                            return {"status": "error", "results": [{"tool": "edit_file", "status": "error", "output": error}], "error": error}
                        t["name"] = "edit_file"
                        t["args"] = {"path": target_path, "old": existing_content, "new": content}
                        log("executor_write_existing_converted_to_edit", {"path": target_path, "tool": tool_name, "step": step[:160]})
                        continue
                    error = (
                        f"Refusing to overwrite existing file '{target_path}'. Use read_file and edit_file for "
                        "a surgical change, or require an explicit user request to rewrite the entire file."
                    )
                    state.add_tool_result(tool_name, "error", error)
                    log("executor_existing_file_overwrite_block", {"path": target_path, "tool": tool_name, "step": step[:160]})
                    return {"status": "error", "results": [{"tool": tool_name, "status": "error", "output": error}], "error": error}
            if isinstance(t, dict) and t.get("name") in {"browser_navigate", "playwright_navigate"}:
                url = str((t.get("args") or {}).get("url", ""))
                if url.lower().startswith("file:"):
                    error = (
                        "Browser navigation to a local file URL is blocked. Read and edit the local "
                        "file directly, or use a user-supplied running http:// URL for browser validation."
                    )
                    state.add_tool_result("browser_navigate", "error", error)
                    log("executor_blocked_file_browser_url", {"url": url[:160], "step": step[:160]})
                    return {
                        "status": "error",
                        "results": [{"tool": "browser_navigate", "status": "error", "output": error}],
                        "error": error,
                    }
            if not isinstance(t, dict) or t.get("name") != "edit_file":
                continue
            args = t.get("args") or {}
            target_path = args.get("path")
            if not target_path:
                continue
            working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
            abs_target = target_path if os.path.isabs(target_path) else os.path.join(working_dir, target_path)
            if os.path.exists(abs_target):
                continue
            resolved = _resolve_existing_path(target_path)
            if resolved:
                log("executor_json_path_fuzzy_resolve", {"requested": target_path, "resolved": resolved})
                args["path"] = resolved
            else:
                error = (
                    f"edit_file target '{target_path}' does not exist and no similar "
                    f"file was found in the project directory."
                )
                log("tool_routing", {
                    "strategy": "json",
                    "status": "edit_target_missing",
                    "path": target_path,
                })
                state.add_tool_result("edit_file", "error", error)
                return {
                    "status": "error",
                    "results": [{"tool": "edit_file", "status": "error", "output": error}],
                    "error": error,
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

        dispatch_result = self.dispatcher.dispatch(action_bundle, knowledge=state.knowledge)

        if dispatch_result.get("signal") in ("noop", "done", "need_context"):
            signal = dispatch_result.get("signal", "noop")
            if signal == "noop" and _step_requires_mutation(step):
                error = (
                    "Invalid noop for an implementation step. The coder must inspect the target "
                    "and perform one verified create_file, write_file, or edit_file operation."
                )
                state.add_tool_result("dispatcher", "error", error)
                log("executor_invalid_noop", {"step": step[:160], "action": action_bundle.get("action")})
                return {
                    "status": "error",
                    "results": [{"tool": "dispatcher", "status": "error", "output": error}],
                    "error": error,
                }
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
    text = text.strip()
    fence_re = re.compile(r"^```[a-z]*\s*\n?", re.IGNORECASE)
    if fence_re.match(text):
        text = fence_re.sub("", text, count=1)
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()