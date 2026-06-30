# core/planner.py
# ─────────────────────────────────────────────────────────────────────────────
# Decides whether a task needs full agent execution or a direct Q&A answer.
# Input refinement lives here too (called from main.py before agent.invoke).
# ─────────────────────────────────────────────────────────────────────────────

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

ACTION_TRIGGERS = (
    "create", "write", "make", "build", "fix", "edit", "update", "delete",
    "run", "execute", "generate", "refactor", "implement", "add", "code",
    "put", "save", "html", "css", "script", "file", "folder", "index",
    "function", "class", "api", "page", "deploy", "install", "setup",
    "rename", "move", "copy", "read", "open", "parse", "fetch", "download",
    "list", "search", "find", "show", "get", "check", "access", "browse",
    "navigate", "click", "screenshot", "scrape", "query", "lookup", "pull",
    "push", "commit", "clone", "diff", "status", "remember", "store",
    "repo", "repository", "github", "git",
)

EXECUTION_CONTEXT_HINTS = (
    "this repo", "this repository", "this project", "current project",
    "codebase", "workspace", "current file", "these files", "my files",
    "local file", "codi.log",
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _starts_with_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    for phrase in phrases:
        if re.match(rf"^{re.escape(phrase)}(\b|[?.!,]|$)", text):
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


class Planner:
    def __init__(self):
        self.llm = get_refiner_llm()

    def _system_prompt(self) -> str:
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        return (
            f"You are Codi, an offline-first AI coding agent.\n"
            f"CURRENT PROJECT DIRECTORY: {working_dir}\n"
            f"Answer questions directly and concisely."
        )

    def needs_execution(self, state: RunState) -> bool:
        """True if the task requires tool execution. False for simple Q&A."""
        result, reason = route_reason(state.user_input)
        log("planner_route", {
            "input": state.user_input[:160],
            "needs_execution": result,
            "reason": reason,
            "input_len": len(state.user_input or ""),
        })
        return result

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
            "update", "generate", "refactor", "implement", "add"
        )
        if not any(t in text.lower() for t in refine_triggers):
            return text

        prompt = (
            "Rewrite this coding task as a clear 1-2 sentence instruction "
            "for an AI agent. No bullet points. No headers. Just the core instruction.\n\n"
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
