# core/planner.py
# ─────────────────────────────────────────────────────────────────────────────
# Decides whether a task needs full agent execution, a direct Q&A answer,
# a read-only lookup, or a targeted single-file edit.
# Input refinement lives here too (called from main.py before agent.invoke).
# ─────────────────────────────────────────────────────────────────────────────

import difflib
import os
import re
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log
from core.quick_actions import is_direct_file_request
from state.temp_db import RunState


SIMPLE_PREFIXES = (
    "hello", "hi", "hey", "what is", "what are", "who is", "explain",
    "how do", "how does", "tell me", "what's", "whats", "thanks", "thank you",
    "yes", "no", "ok", "okay", "sure", "help", "why", "when", "where"
)

# "debug" added — without it, "debug the entire website" has no action
# trigger word, no file extension, and is under 80 chars, so route_reason
# fell through to short_no_action and got answered as plain Q&A instead of
# ever touching the project files.
ACTION_TRIGGERS = (
    "create", "write", "make", "build", "fix", "edit", "update", "delete",
    "run", "execute", "generate", "refactor", "implement", "add", "code",
    "put", "save", "html", "css", "script", "file", "folder", "index",
    "function", "class", "api", "page", "deploy", "install", "setup",
    "rename", "move", "copy", "open", "parse", "fetch", "download",
    "list", "search", "find", "show", "get", "check", "access", "browse",
    "navigate", "click", "screenshot", "scrape", "query", "lookup", "pull",
    "push", "commit", "clone", "diff", "status", "remember", "store",
    "repo", "repository", "github", "git", "debug", "troubleshoot",
    "diagnose", "inspect", "lint",
)

EXECUTION_CONTEXT_HINTS = (
    "this repo", "this repository", "this project", "current project",
    "codebase", "workspace", "current file", "these files", "my files",
    "local file", "codi.log", "the website", "the frontend", "the site",
)

# ── Intent classification vocab ────────────────────────────────────────────
# These must be defined BEFORE route_reason() since route_reason now also
# consults them (a phrase like "change the button color" has no
# ACTION_TRIGGERS word and no file extension, so without this it was
# silently classified as a plain question — see planner_route bug).
READ_VERBS = (
    "read", "show", "explain", "describe", "summarize", "summarise",
    "what does", "how does", "walk me through", "understand", "review",
    "look at", "inspect", "analyze", "analyse", "tell me about",
)

# "debug" and "troubleshoot" belong here too — debugging is fundamentally
# a fix/edit action (it implies inspecting AND then changing code), not a
# read-only lookup. Treating it as edit-intent means it correctly triggers
# execution and, downstream in classify_intent, gets routed toward "build"
# (multi-file) rather than silently answered as prose.
EDIT_VERBS = (
    "edit", "fix", "update", "change", "modify", "rename", "refactor",
    "remove", "delete", "replace", "append", "prepend", "insert", "patch",
    "debug", "troubleshoot", "diagnose",
)

BUILD_VERBS = (
    "create", "build", "generate", "scaffold", "implement", "make",
    "set up", "setup", "new project", "new app", "write a",
)

FILE_PATH_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,5}\b")


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _starts_with_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    for phrase in phrases:
        if re.match(rf"^{re.escape(phrase)}(\b|[?.!,]|$)", text):
            return True
    return False


def _phrase_hit(t: str, phrases: tuple[str, ...]) -> str | None:
    for phrase in phrases:
        if phrase in t:
            return phrase
    return None


def _fuzzy_verb_hit(words: set[str], vocab: tuple[str, ...], cutoff: float = 0.8) -> bool:
    """
    Catch common typos ("chnage" -> "change", "dowload" -> "download") that
    exact substring matching misses. Only checks single-word vocab entries —
    multi-word phrases ("what does", "new project") aren't meaningfully
    fuzzy-matchable against a single mistyped token.
    """
    single_words = [v for v in vocab if " " not in v]
    for w in words:
        if len(w) < 4:
            continue
        if difflib.get_close_matches(w, single_words, n=1, cutoff=cutoff):
            return True
    return False


def route_reason(text: str) -> tuple[bool, str]:
    """Return (needs_execution, reason) using word-aware routing."""
    t = text.lower().strip()
    words = _tokens(t)

    if not t:
        return False, "empty"

    if any(hint in t for hint in EXECUTION_CONTEXT_HINTS):
        return True, "mentions_workspace_context"

    if re.search(r"[A-Za-z0-9_./\\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|md|txt|svg|sh)\b", text):
        return True, "mentions_file_path"

    action_hits = sorted(set(ACTION_TRIGGERS).intersection(words))
    if action_hits:
        return True, f"action_trigger:{','.join(action_hits[:5])}"

    # EDIT_VERBS / BUILD_VERBS / READ_VERBS are used downstream by
    # classify_intent() for fine-grained routing, but they must ALSO count
    # as execution triggers here — otherwise a phrase like "debug the
    # entire website" or "change the button color to blue" (no
    # ACTION_TRIGGERS word, no file extension, under 80 chars) falls
    # through to "short_no_action" and gets answered as plain Q&A instead
    # of actually being executed.
    phrase_hit = (
        _phrase_hit(t, EDIT_VERBS)
        or _phrase_hit(t, BUILD_VERBS)
        or _phrase_hit(t, READ_VERBS)
    )
    if phrase_hit:
        return True, f"phrase_trigger:{phrase_hit}"

    # typo tolerance for the same vocab (chnage/dowload/etc.)
    if (
        _fuzzy_verb_hit(words, EDIT_VERBS)
        or _fuzzy_verb_hit(words, BUILD_VERBS)
        or _fuzzy_verb_hit(words, READ_VERBS)
        or _fuzzy_verb_hit(words, ACTION_TRIGGERS)
    ):
        return True, "fuzzy_verb_trigger"

    if len(t) < 80:
        return False, "short_no_action"

    if _starts_with_phrase(t, SIMPLE_PREFIXES):
        return False, "simple_question_prefix"

    return True, "long_or_ambiguous"


def is_simple_input(text: str) -> bool:
    """
    True when the input is clearly a Q&A question that needs no tool execution.
    Returns False (i.e. needs execution) if any action trigger word is found.
    """
    needs_execution, _ = route_reason(text)
    return not needs_execution


def classify_intent(text: str) -> str:
    """
    Classify a task into one of four buckets so the agent can pick the
    right-sized pipeline instead of always running the full build loop.

    "qa"    — plain question, no tools needed at all
    "read"  — inspect/explain existing code; tools may run (read/search)
              but nothing is ever written
    "edit"  — targeted change to one (or two) existing file(s); tries a
              single direct executor call before falling back to "build"
    "build" — multi-file creation / scaffolding / ambiguous — full pipeline
    """
    needs_exec, _ = route_reason(text)
    if not needs_exec:
        return "qa"

    t = (text or "").lower()
    words = _tokens(t)
    has_file       = bool(FILE_PATH_RE.search(text))
    # A read verb only counts as genuine read-intent when it appears near
    # the START of the instruction (the user's actual ask), not buried deep
    # in a long sentence describing unrelated downstream behavior. Without
    # this, a request like "...upload files and the Chatbot should be able
    # to read answer from those files" was misclassified as pure "read"
    # intent purely because the word "read" appeared 20+ words in, despite
    # the sentence's actual leading intent being "add [a feature]".
    # ACTION_TRIGGERS/EDIT_VERBS-style words anywhere still count normally;
    # this narrowing applies only to READ_VERBS since those are the ones
    # that can otherwise downgrade a build/edit task to explain-only.
    _lead_window = " ".join(t.split()[:12])
    has_read_verb  = (
        any(v in _lead_window for v in READ_VERBS)
        or _fuzzy_verb_hit(_tokens(_lead_window), READ_VERBS)
    )
    has_edit_verb  = any(v in t for v in EDIT_VERBS) or _fuzzy_verb_hit(words, EDIT_VERBS)
    has_build_verb = any(v in t for v in BUILD_VERBS) or _fuzzy_verb_hit(words, BUILD_VERBS)
    file_mentions  = len(FILE_PATH_RE.findall(text))

    # "debug the entire website" — no explicit file, but multiple project-wide
    # words ("entire", "website", "frontend") imply multi-file scope. Since
    # EDIT_VERBS now includes debug/troubleshoot/diagnose, has_edit_verb will
    # be True here; without a specific file mentioned, treat broad-scope
    # debug/troubleshoot requests as "build" (full pipeline) rather than the
    # single-file "edit" fast path, since they legitimately need to inspect
    # and potentially touch more than one file.
    broad_scope_hint = any(
        phrase in t for phrase in ("entire website", "whole site", "entire site", "whole project", "entire project")
    )
    if broad_scope_hint:
        return "build"

    # FIX: read-intent is no longer inferred from keywords. The word "read"
    # appearing in natural language (e.g. "the chatbot should be able to read
    # answer from those files") was incorrectly locking Codi into read-only
    # mode, preventing any writes. Read-intent is now ONLY activated by the
    # explicit /read command prefix (handled in Planner.classify via
    # state.force_read). When has_read_verb fires without edit/build verbs,
    # we fall through to the build classification below instead of returning
    # "read" — this lets the full pipeline decide whether the task actually
    # needs file modifications.
    #
    # OLD: if has_read_verb and not has_build_verb and not has_edit_verb:
    #          return "read"

    # NOTE: " and " is deliberately NOT a build signal on its own anymore.
    # "edit index.html and add content there" has one file and one edit
    # verb and no build verb — it should be a targeted single-file edit,
    # not a full read-context + multi-step plan (which previously caused
    # the planner to invent an unrequested edit to styles.css as well).
    if has_build_verb or file_mentions > 1:
        return "build"

    if has_edit_verb and (has_file or "this file" in t or "that function" in t):
        return "edit"

    return "build"


class Planner:
    def __init__(self):
        self.llm = get_refiner_llm()

    def _system_prompt(self) -> str:
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        return (
            f"You are Codi, an offline-first AI coding agent having a plain-text "
            f"conversation with a developer. You are NOT executing tools right now — "
            f"this is a direct question-and-answer exchange.\n"
            f"CURRENT PROJECT DIRECTORY: {working_dir}\n\n"
            f"Answer directly and concisely. Use code blocks for code. "
            f"If the question implies you should inspect or change actual project "
            f"files, say so plainly — don't pretend to have read files you haven't."
        )

    def needs_execution(self, state: RunState) -> bool:
        """True if the task requires tool execution. False for simple Q&A.

        Kept for backward compatibility — prefer classify() for new code,
        since it distinguishes read/edit/build instead of a single bucket.
        """
        result, reason = route_reason(state.user_input)
        log("planner_route", {
            "input": state.user_input[:160],
            "needs_execution": result,
            "reason": reason,
            "input_len": len(state.user_input or ""),
        })
        return result

    def classify(self, state: RunState) -> str:
        """Return one of "qa" | "read" | "edit" | "build" for state.user_input.

        The "read" intent is ONLY returned when state.force_read is True
        (set by main.py when the user types /read). Keyword-based read
        detection was removed from classify_intent() because the word
        "read" in natural language too easily downgraded real edit/build
        tasks to read-only mode.
        """
        if state.force_read:
            log("planner_classify", {
                "input": state.user_input[:160],
                "intent": "read",
                "input_len": len(state.user_input or ""),
                "reason": "force_read_via_slash_command",
            })
            return "read"

        intent = classify_intent(state.user_input)
        log("planner_classify", {
            "input": state.user_input[:160],
            "intent": intent,
            "input_len": len(state.user_input or ""),
        })
        return intent

    def direct_answer(self, state: RunState) -> str:
        """For simple Q&A that doesn't need tools. Returns plain text answer."""
        try:
            resp = self.llm.invoke([
                SystemMessage(content=self._system_prompt()),
                HumanMessage(content=state.user_input),
            ])
            answer = resp.content.strip()
            log("planner_direct", {"output": answer[:100]})
            return answer
        except Exception as e:
            log("planner_direct_error", {"error": str(e)})
            return f"Error generating response: {e}"

    def refine_input(self, raw_input: str) -> str:
        """
        Optionally rewrite the user input as a crisp 1-2 sentence instruction.
        Short inputs, questions, and direct file requests are returned unchanged.
        """
        text = raw_input.strip()

        # Never refine direct file requests — they're already precise
        if is_direct_file_request(text):
            return text

        if len(text) < 50:
            return text

        refine_triggers = (
            "create", "write", "make", "build", "fix", "edit",
            "update", "generate", "refactor", "implement", "add", "debug",
        )
        if not any(t in text.lower() for t in refine_triggers):
            return text

        prompt = (
            "Rewrite this coding task as a clear 1-2 sentence instruction "
            "for an AI agent. Preserve every concrete detail: file names, "
            "exact wording, colors, values. Do not add scope that wasn't "
            "there. No bullet points. No headers. Just the core instruction.\n\n"
            f"Task: {text}\nInstruction:"
        )
        try:
            resp    = self.llm.invoke([HumanMessage(content=prompt)])
            refined = resp.content.strip()
            # Discard if refiner bloated the prompt or returned garbage
            if len(refined) > len(text) * 2 or len(refined) < 10:
                return text
            return refined
        except Exception:
            return text