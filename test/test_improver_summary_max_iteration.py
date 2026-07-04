from core.improver import Improver
from state.temp_db import RunState, TaskRequirements
from tools.registry import ToolRegistry


def test_summarize_max_iteration_with_validation_failure():
    registry = ToolRegistry()
    imp = Improver(registry)

    state = RunState()
    state.tool_results = [
        # one successful write
        type("R", (), {"tool": "create_file", "status": "ok", "output": '{"file_modified":"index.html"}'})(),
        # one error
        type("R", (), {"tool": "coder", "status": "error", "output": 'malformed json'})(),
    ]
    state.requirements = TaskRequirements(framework="react")
    state.iteration = state.max_iterations
    state.validation_passed = False
    state.validation_notes = "Structural error: missing package.json"

    summary = imp.summarize(state)
    assert "reached max iterations" in summary.lower() or "stopped" in summary.lower()
    assert "missing package.json" in summary.lower()
