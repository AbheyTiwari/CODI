from typing import List, Tuple
from llm_factory import get_refiner_llm
from langchain_core.messages import HumanMessage, SystemMessage

class SessionMemory:
    def __init__(self, max_turns: int = 15):
        self.max_turns = max_turns
        self._history: List[Tuple[str, str]] = []
        self._summary: str = ""

    def add(self, role: str, msg: str):
        # Truncate giant tool outputs before storing
        if role == "assistant" and len(msg) > 3000:
            msg = msg[:1500] + "\n... [TRUNCATED] ...\n" + msg[-1500:]
        self._history.append((role, msg))
        if len(self._history) > self.max_turns * 2:
            self._compress_memory()

    def _compress_memory(self):
        old_messages = self._history[:8]
        self._history = self._history[8:]
        text_to_summarize = "\n".join(f"{r.capitalize()}: {m}" for r, m in old_messages)
        prompt = (
            f"Previous summary: {self._summary}\n\n"
            f"New messages:\n{text_to_summarize}\n\n"
            f"Provide a concise running summary of the conversation so far."
        )
        try:
            llm = get_refiner_llm()
            resp = llm.invoke([
                SystemMessage(content="Compress conversation history into a dense summary. Keep facts, constraints, and current state."),
                HumanMessage(content=prompt)
            ])
            self._summary = resp.content
        except Exception:
            pass  # if compression fails, drop oldest without summarizing

    def as_text(self) -> str:
        parts = []
        if self._summary:
            parts.append(f"[Historical Summary]: {self._summary}\n---")
        if not self._history and not self._summary:
            return "No previous conversation history."
        for role, msg in self._history:
            parts.append(f"{role.capitalize()}: {msg}")
        return "\n".join(parts)

    def clear(self):
        self._history.clear()
        self._summary = ""

# Global session memory instance
session_memory = SessionMemory()