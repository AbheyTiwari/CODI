import json
import os
import tempfile
import unittest
from pathlib import Path

from agent import _step_succeeded
from core.executor import Executor
from core.improver import _enforce_test_intent, _normalize_plan_steps
from core.improver import Improver
from core.validator import Validator
from dispatcher import Dispatcher
from state.code_index import find_references, find_symbols, index_project
from state.temp_db import RunState, TaskRequirements, ToolResult
from tools.local.file_tools import create_file, edit_file


class CodeIndexAndSurgicalEditTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures inside the repository so restricted Windows runners
        # do not deny access to the system temp directory.
        self.tempdir = tempfile.TemporaryDirectory(dir=os.getcwd())
        self.addCleanup(self.tempdir.cleanup)
        self.previous_dir = os.environ.get("CODI_WORKING_DIR")
        os.environ["CODI_WORKING_DIR"] = self.tempdir.name
        self.addCleanup(self._restore_working_dir)

    def _restore_working_dir(self):
        if self.previous_dir is None:
            os.environ.pop("CODI_WORKING_DIR", None)
        else:
            os.environ["CODI_WORKING_DIR"] = self.previous_dir

    def test_sqlite_index_tracks_exact_python_symbol_and_references(self):
        source = Path(self.tempdir.name) / "example.py"
        source.write_text("def total(items):\n    total = len(items)\n    return total\n", encoding="utf-8")

        indexed = index_project()
        declarations = find_symbols("total", path="example.py")
        references = find_references("total", path="example.py")

        self.assertTrue(indexed["success"])
        self.assertTrue(any(item["kind"] == "function" for item in declarations))
        self.assertEqual([item["line"] for item in references], [1, 2, 3])

    def test_ambiguous_text_edit_is_rejected_without_changing_file(self):
        source = Path(self.tempdir.name) / "example.txt"
        source.write_text("value = 1\nvalue = 1\n", encoding="utf-8")

        result = edit_file({"path": "example.txt", "old": "value = 1", "new": "value = 2"})

        self.assertIn("ambiguous", result.lower())
        self.assertEqual(source.read_text(encoding="utf-8"), "value = 1\nvalue = 1\n")

    def test_create_file_never_overwrites_existing_file(self):
        source = Path(self.tempdir.name) / "main.py"
        source.write_text("original = True\n", encoding="utf-8")

        result = create_file({"path": "main.py", "content": "print('Hello, World!')"})

        self.assertIn("already exists", result)
        self.assertEqual(source.read_text(encoding="utf-8"), "original = True\n")

    def test_validator_receives_complete_modified_file_source(self):
        source = Path(self.tempdir.name) / "generated.py"
        source.write_text("def generated():\n    return 42\n", encoding="utf-8")
        state = RunState()
        state.files_written.add(str(source))

        content, error = Validator._changed_source_context(state)

        self.assertEqual(error, "")
        self.assertIn("def generated", content)
        self.assertIn("generated.py", content)

    def test_validator_repair_is_prioritized_over_replanning(self):
        state = RunState()
        state.validation_repair_instruction = "In app.py, replace only the stale null check on line 12."
        improver = Improver.__new__(Improver)

        decision = improver.next_step(state)

        self.assertFalse(decision["done"])
        self.assertIn("line 12", decision["step"])
        self.assertEqual(state.validation_repair_instruction, "")

    def test_validator_returns_surgical_repair_from_complete_source(self):
        source = Path(self.tempdir.name) / "app.py"
        source.write_text("def run():\n    return None\n", encoding="utf-8")

        class Response:
            content = json.dumps({
                "passed": False,
                "notes": "The function returns None instead of a value.",
                "repair_instruction": "In app.py, replace only `return None` with `return 42`.",
                "findings": [{"path": "app.py", "line": 2, "severity": "error", "problem": "wrong return", "repair": "return 42"}],
            })

        class LLM:
            def __init__(self):
                self.messages = []

            def invoke(self, messages):
                self.messages = messages
                return Response()

        state = RunState(user_input="Make run return 42")
        state.files_written.add(str(source))
        validator = Validator()
        validator.llm = LLM()

        passed = validator._llm_check(state)

        self.assertFalse(passed)
        self.assertIn("replace only", state.validation_repair_instruction)
        self.assertEqual(state.validation_findings[0]["line"], 2)
        self.assertIn("def run", validator.llm.messages[0].content)

    def test_read_or_noop_cannot_complete_implementation_step(self):
        state = RunState()
        state.tool_results = [ToolResult("read_file", "ok", "source")]
        self.assertFalse(_step_succeeded(state, "Add responsive rules in portfolio.html"))
        state.tool_results = [ToolResult("dispatcher", "ok", "noop")]
        self.assertFalse(_step_succeeded(state, "Implement animations in portfolio.html"))
        state.tool_results = [ToolResult("edit_file", "ok", '{"file_modified":"portfolio.html"}')]
        self.assertTrue(_step_succeeded(state, "Implement animations in portfolio.html"))

    def test_planner_normalizes_vague_single_file_steps(self):
        steps = _normalize_plan_steps(["Add JavaScript animations"], TaskRequirements(files=["portfolio.html"]))
        self.assertEqual(steps, ["Add JavaScript animations in portfolio.html"])

    def test_planner_replaces_local_browser_step_with_source_verification(self):
        steps = _normalize_plan_steps(["Open portfolio.html in the browser"], TaskRequirements(files=["portfolio.html"]))
        self.assertEqual(steps, ["Read portfolio.html and verify its HTML structure and requested content"])

    def test_unit_test_plan_protects_source_and_targets_test_file(self):
        requirements = TaskRequirements(files=["test_main.py"], protected_files=["main.py"])
        steps = _enforce_test_intent(["Write unit test for CabOptimizer class in main.py", "Run the unit test"], requirements)
        self.assertEqual(steps[0], "Create test_main.py with focused unit tests for behavior in main.py")
        self.assertEqual(steps[1], "Run the focused tests in test_main.py against main.py")

    def test_executor_rejects_noop_for_implementation(self):
        class Registry:
            def list_names(self): return []
            def summary(self): return ""

        class Response:
            content = '{"action":"noop"}'

        class LLM:
            def invoke(self, _messages): return Response()

        executor = Executor(Registry())
        executor.llm = LLM()
        result = executor.execute_step("Implement animations in portfolio.html", RunState())
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid noop", result["error"])

    def test_dispatcher_marks_embedded_mcp_error_as_failure(self):
        class Registry:
            def get(self, _name):
                return lambda _args: "### Error\nError: browserBackend.callTool: browser closed"
            def list_names(self): return ["browser_take_screenshot"]

        result = Dispatcher(Registry()).dispatch({"action": "tool_call", "tools": [{"name": "browser_take_screenshot", "args": {}}]})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["results"][0]["status"], "error")

    def test_executor_blocks_file_protocol_browser_navigation(self):
        class Registry:
            def list_names(self): return []
            def summary(self): return ""

        class Response:
            content = '{"action":"tool_call","tools":[{"name":"browser_navigate","args":{"url":"file:///tmp/portfolio.html"}}]}'

        class LLM:
            def invoke(self, _messages): return Response()

        executor = Executor(Registry())
        executor.llm = LLM()
        result = executor.execute_step("Verify portfolio.html", RunState())
        self.assertEqual(result["status"], "error")
        self.assertIn("local file URL", result["error"])

    def test_executor_blocks_test_task_write_to_protected_source(self):
        class Registry:
            def list_names(self): return []
            def summary(self): return ""

        class Response:
            content = '{"action":"tool_call","tools":[{"name":"create_file","args":{"path":"main.py","content":"print(1)"}}]}'

        class LLM:
            def invoke(self, _messages): return Response()

        state = RunState(user_input="write unit test of main.py")
        state.requirements = TaskRequirements(files=["test_main.py"], protected_files=["main.py"])
        executor = Executor(Registry())
        executor.llm = LLM()
        result = executor.execute_step("Create test_main.py with unit tests for main.py", state)
        self.assertEqual(result["status"], "error")
        self.assertIn("Protected source file", result["error"])

    def test_executor_converts_create_to_edit_when_test_file_exists(self):
        class Registry:
            def list_names(self): return ["create_file", "edit_file"]
            def summary(self): return "create_file, edit_file"

        class Response:
            content = '{"action":"tool_call","tools":[{"name":"create_file","args":{"path":"test_main.py","content":"def test_answer():\\n    assert 2 + 2 == 4\\n"}}]}'

        class LLM:
            def invoke(self, _messages): return Response()

        class Dispatcher:
            def __init__(self): self.bundle = None
            def dispatch(self, bundle, knowledge=None):
                self.bundle = bundle
                tool = bundle["tools"][0]
                return {"status": "success", "results": [{"tool": tool["name"], "status": "ok", "output": '{"file_modified":"test_main.py"}'}]}

        target = Path(self.tempdir.name) / "test_main.py"
        target.write_text("# placeholder\\n", encoding="utf-8")
        executor = Executor(Registry())
        executor.llm = LLM()
        dispatcher = Dispatcher()
        executor.dispatcher = dispatcher

        result = executor.execute_step("Create test_main.py with focused unit tests", RunState())

        self.assertEqual(result["status"], "success")
        tool = dispatcher.bundle["tools"][0]
        self.assertEqual(tool["name"], "edit_file")
        self.assertEqual(tool["args"]["old"], "# placeholder\\n")


if __name__ == "__main__":
    unittest.main()
