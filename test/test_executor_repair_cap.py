from core.executor import Executor
from state.temp_db import RunState
from tools.registry import ToolRegistry


class DummyLLM:
    def invoke(self, messages):
        class R:
            content = "not valid json"
        return R()


def test_repair_attempts_capped(monkeypatch):
    registry = ToolRegistry()
    ex = Executor(registry)
    # Replace llm and repair method
    ex.llm = DummyLLM()

    def fake_repair(step, raw, state, reason):
        # Simulate repair returning None (still broken)
        return None

    monkeypatch.setattr(ex, "_repair_action_bundle", fake_repair)

    state = RunState()
    # Use a non-file-write step so content-first routing isn't used
    step = "Make a small change to README"
    result = ex.execute_step(step, state)

    # Expect an error after repair exhausted and one attempted repair recorded
    assert state.repair_attempts.get(step, 0) == 1
    assert result.get("status") == "error"
