from __future__ import annotations

from types import SimpleNamespace

from agent import _confirmed_missing_file_creation
import memory
from core.planner import Planner
from state.temp_db import RunState


class _CaptureLLM:
    def __init__(self, answer: str = "ok"):
        self.answer = answer
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.answer)


def test_direct_answer_receives_previous_conversation():
    planner = Planner.__new__(Planner)
    planner.llm = _CaptureLLM("I remember it.")
    state = RunState(
        user_input="What did you change?",
        history="User: Fix the checkout button.\nAssistant: Updated checkout.css.",
    )

    assert planner.direct_answer(state) == "I remember it."
    assert "Updated checkout.css" in planner.llm.messages[-1].content


def test_failed_memory_compression_keeps_original_messages(monkeypatch):
    session = memory.SessionMemory(max_turns=1)
    session._history = [("user", f"message {index}") for index in range(8)]

    class _UnavailableLLM:
        def invoke(self, _messages):
            raise RuntimeError("offline")

    monkeypatch.setattr(memory, "get_refiner_llm", lambda: _UnavailableLLM())
    session._compress_memory()

    assert len(session._history) == 8


def test_yes_to_missing_file_prompt_authorizes_creation_not_more_discovery():
    unknowns = [
        "Requested file is not present: index.html",
        "Requested file is not present: script.js",
    ]

    assert _confirmed_missing_file_creation("yes, create them", unknowns) == [
        "index.html", "script.js"
    ]
    assert _confirmed_missing_file_creation("create it", unknowns) == [
        "index.html", "script.js"
    ]
    assert _confirmed_missing_file_creation("crate it", unknowns) == [
        "index.html", "script.js"
    ]
    assert _confirmed_missing_file_creation("what framework are we using?", unknowns) == []
