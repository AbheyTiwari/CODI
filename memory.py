import threading
from typing import List, Tuple
from llm_factory import get_refiner_llm
from langchain_core.messages import HumanMessage, SystemMessage


class SessionMemory:
    def __init__(self, max_turns: int = 15):
        self.max_turns = max_turns
        self._history: List[Tuple[str, str]] = []
        self._summary: str = ""
        self._lock = threading.Lock()
        self._compressing = False          # guard: only one compression at a time

    def add(self, role: str, msg: str):
        # Truncate giant tool outputs before storing
        if role == "assistant" and len(msg) > 3000:
            msg = msg[:1500] + "\n... [TRUNCATED] ...\n" + msg[-1500:]

        with self._lock:
            self._history.append((role, msg))
            should_compress = (
                len(self._history) > self.max_turns * 2
                and not self._compressing
            )
            if should_compress:
                self._compressing = True

        # Fire compression on a daemon thread so the main loop never blocks.
        # The lock is released before starting the thread — add() returns instantly.
        if should_compress:
            t = threading.Thread(target=self._compress_memory, daemon=True)
            t.start()

    def _compress_memory(self):
        """
        Summarize the oldest 8 messages and drop them from history.
        Runs on a background thread — never blocks the CLI.
        Uses a lock around all history mutations so as_text() stays consistent.
        """
        try:
            with self._lock:
                if len(self._history) < 8:
                    return
                old_messages = self._history[:8]
                self._history = self._history[8:]

            text_to_summarize = "\n".join(
                f"{r.capitalize()}: {m}" for r, m in old_messages
            )
            prompt = (
                f"Previous summary: {self._summary}\n\n"
                f"New messages:\n{text_to_summarize}\n\n"
                f"Provide a concise running summary of the conversation so far."
            )

            llm = get_refiner_llm()
            resp = llm.invoke([
                SystemMessage(content=(
                    "Compress conversation history into a dense summary. "
                    "Keep facts, constraints, and current state."
                )),
                HumanMessage(content=prompt),
            ])

            with self._lock:
                self._summary = resp.content

        except Exception:
            pass  # if compression fails, oldest messages are already dropped — that's fine

        finally:
            self._compressing = False

    def as_text(self) -> str:
        with self._lock:
            history_snapshot = list(self._history)
            summary_snapshot = self._summary

        parts = []
        if summary_snapshot:
            parts.append(f"[Historical Summary]: {summary_snapshot}\n---")
        if not history_snapshot and not summary_snapshot:
            return "No previous conversation history."
        for role, msg in history_snapshot:
            parts.append(f"{role.capitalize()}: {msg}")
        return "\n".join(parts)

    def clear(self):
        with self._lock:
            self._history.clear()
            self._summary = ""
            self._compressing = False


# Global session memory instance
session_memory = SessionMemory()