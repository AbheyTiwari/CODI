# core/planner.py
# ─────────────────────────────────────────────────────────────────────────────
# Decides whether a task needs full agent execution or a direct Q&A answer.
# Input refinement lives here too (called from main.py before agent.invoke).
# ─────────────────────────────────────────────────────────────────────────────

import os
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log
from core.quick_actions import is_direct_file_request
from state.temp_db import RunState


SIMPLE_TRIGGERS = (
    "hello", "hi ", "hey ", "what is", "what are", "who is", "explain",
    "how do", "how does", "tell me", "what's", "whats", "thanks", "thank you",
    "yes", "no", "ok", "okay", "sure", "help", "why ", "when ", "where "
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


def is_simple_input(text: str) -> bool:
    """
    True when the input is clearly a Q&A question that needs no tool execution.
    Returns False (i.e. needs execution) if any action trigger word is found.
    """
    t = text.lower().strip()
    if any(w in t for w in ACTION_TRIGGERS):
        return False
    if len(t) < 80:
        return True
    if any(t.startswith(trigger) for trigger in SIMPLE_TRIGGERS):
        return True
    return False


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
        result = not is_simple_input(state.user_input)
        log("planner_route", {"input": state.user_input[:80], "needs_execution": result})
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