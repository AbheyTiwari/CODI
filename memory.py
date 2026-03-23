from typing import List, Tuple

class SessionMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: List[Tuple[str, str]] = []

    def add(self, role: str, msg: str):
        self._history.append((role, msg))
        if len(self._history) > self.max_turns * 2: # pairs of messages roughly
            # Drop the oldest turn (two messages: user + assistant)
            # Or just keep the raw length bounded
            self._history = self._history[-self.max_turns * 2:]

    def as_messages(self):
        # Could convert to LangChain BaseMessage objects if needed, 
        # but string formatting might be enough depending on the prompt.
        return self._history

    def as_text(self) -> str:
        if not self._history:
            return "No previous conversation history."
        
        text_parts = []
        for role, msg in self._history:
            text_parts.append(f"{role.capitalize()}: {msg}")
        return "\n".join(text_parts)

    def clear(self):
        self._history.clear()

# Global session memory
session_memory = SessionMemory()
