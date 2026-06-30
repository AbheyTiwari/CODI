import json
import tempfile
import unittest
from pathlib import Path

from core.validator import Validator
from state.temp_db import RunState, TaskRequirements, ToolResult


class ValidatorRepeatedFailureTests(unittest.TestCase):
    def test_different_failure_causes_do_not_count_as_stalled(self):
        state = RunState(iteration=4)
        state.tool_results = [
            ToolResult("coder", "error", "Framework contamination blocked: flask"),
            ToolResult("coder", "error", "Tool not found: edit_file"),
            ToolResult("coder", "error", "Tool not found: edit_file"),
            ToolResult("coder", "error", "Tool not found: edit_file"),
        ]

        validator = Validator.__new__(Validator)
        self.assertFalse(validator._is_stalled(state))

    def test_same_failure_cause_repeatedly_counts_as_stalled(self):
        state = RunState(iteration=4)
        state.tool_results = [
            ToolResult("coder", "error", "Framework contamination blocked: flask"),
            ToolResult("coder", "error", "Framework contamination blocked: flask"),
            ToolResult("coder", "error", "Framework contamination blocked: flask"),
            ToolResult("coder", "error", "Framework contamination blocked: flask"),
        ]

        validator = Validator.__new__(Validator)
        self.assertTrue(validator._is_stalled(state))

    def test_framework_contamination_detects_forbidden_import_in_written_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "app.py"
            path.write_text("from flask import Flask\napp = Flask(__name__)\n", encoding="utf-8")

            state = RunState()
            state.requirements = TaskRequirements(framework="fastapi")
            state.tool_results = [
                ToolResult(
                    "write_file",
                    "ok",
                    json.dumps({"success": True, "file_modified": str(path), "bytes_written": len(path.read_text(encoding='utf-8'))}),
                )
            ]

            validator = Validator()
            result = validator.validate(state)

            self.assertFalse(result)
            self.assertIn("forbidden", state.validation_notes.lower())
            self.assertIn("flask", state.validation_notes.lower())

    def test_validator_llm_check_does_not_crash(self):
        state = RunState(user_input="create a file", iteration=2, max_iterations=8)
        state.add_tool_result("write_file", "ok", "Written 10 chars to a.py")
        validator = Validator()

        try:
            validator.validate(state)
        except Exception as exc:
            self.fail(f"Validator.validate raised an exception: {exc}")

    def test_validator_does_not_pass_with_remaining_plan_steps(self):
        state = RunState(user_input="create a file", iteration=1)
        state.plan_steps = ["Write a.py", "Write b.py", "Write c.py"]
        state.add_tool_result("write_file", "ok", "Written 10 chars to a.py")

        validator = Validator()
        result = validator.validate(state)

        self.assertFalse(result)
        self.assertEqual(state.validation_notes, "Task has remaining plan steps.")

    def test_executor_rejects_forbidden_framework_content(self):
        from core.executor import Executor

        registry = type("DummyRegistry", (), {"list_names": lambda self: [], "summary": lambda self: "",})()
        executor = Executor(registry)
        state = RunState()
        state.requirements = TaskRequirements(framework="fastapi")
        action_bundle = {
            "action": "tool_call",
            "tools": [
                {
                    "name": "write_file",
                    "args": {
                        "path": "app.py",
                        "content": "from flask import Flask\napp = Flask(__name__)\n",
                    },
                }
            ],
        }

        violation = executor._framework_violation(state, action_bundle)
        self.assertIsNotNone(violation)
        self.assertIn("Forbidden framework content", violation)
        self.assertIn("fastapi", violation.lower())


if __name__ == "__main__":
    unittest.main()
