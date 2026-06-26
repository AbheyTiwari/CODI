import unittest

from core.validator import Validator
from state.temp_db import RunState, ToolResult


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


if __name__ == "__main__":
    unittest.main()
