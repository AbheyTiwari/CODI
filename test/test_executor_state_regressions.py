import json
import unittest

from core.executor import Executor
from core.validator import Validator
from state.temp_db import RunState
from tools.local.file_tools import write_file


class DummyResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {}


class DummyLLM:
    def __init__(self, responses):
        self.responses = [DummyResponse(r) for r in responses]

    def invoke(self, _messages):
        if not self.responses:
            raise AssertionError("No more dummy LLM responses")
        return self.responses.pop(0)


class ExecutorStateRegressionTests(unittest.TestCase):
    def test_duplicate_content_first_write_is_skipped_and_logged(self):
        class DummyRegistry:
            def list_names(self):
                return ["write_file"]

            def summary(self):
                return "write_file"

        class DummyDispatcher:
            def __init__(self):
                self.calls = []

            def dispatch_file_write(self, tool, path, content, metadata=None):
                self.calls.append((tool, path, content, metadata))
                return {"status": "success", "results": [{"tool": tool, "status": "ok", "output": "wrote"}]}

        state = RunState()
        executor = Executor(DummyRegistry())
        executor.llm = DummyLLM(["```java\npackage demo;\nclass Book {}\n```\nCODI_FILE_WRITE_COMPLETE"])
        executor.dispatcher = DummyDispatcher()

        first = executor._execute_content_first("Write Book.java", "write_file", "Book.java", state)
        second = executor._execute_content_first("Write Book.java again", "write_file", "Book.java", state)

        self.assertEqual(first["status"], "success")
        self.assertEqual(second["status"], "success")
        self.assertEqual(len(executor.dispatcher.calls), 1)
        self.assertTrue(any("skip_duplicate_write" in result.output for result in state.tool_results if result.tool == "dispatcher"))

    def test_invalid_noop_for_write_step_retries_before_failing(self):
        class DummyRegistry:
            def list_names(self):
                return ["write_file"]

            def summary(self):
                return "write_file"

        class DummyDispatcher:
            def __init__(self):
                self.calls = []

            def dispatch(self, action_bundle):
                self.calls.append(action_bundle)
                return {"status": "success", "results": [{"tool": "write_file", "status": "ok", "output": "wrote"}]}

        state = RunState()
        executor = Executor(DummyRegistry())
        executor.llm = DummyLLM([
            '{"action":"noop"}',
            '{"action":"tool_call","tools":[{"name":"write_file","args":{"path":"demo.txt","content":"hello"}}]}',
        ])
        executor.dispatcher = DummyDispatcher()

        result = executor.execute_step("Write the parser implementation", state)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(executor.dispatcher.calls), 1)
        self.assertTrue(any("tool_call" in item.output.lower() for item in state.tool_results if item.tool == "dispatcher"))

    def test_content_first_prompt_includes_project_manifest_package(self):
        class DummyRegistry:
            def list_names(self):
                return ["write_file"]

            def summary(self):
                return "write_file"

        class DummyDispatcher:
            def __init__(self):
                self.calls = []

            def dispatch_file_write(self, tool, path, content, metadata=None):
                self.calls.append((tool, path, content, metadata))
                return {"status": "success", "results": [{"tool": tool, "status": "ok", "output": "wrote"}]}

        state = RunState()
        state.project_manifest = {"package": "com.example.library", "files_created": {}}
        executor = Executor(DummyRegistry())

        class PromptAwareLLM:
            def __init__(self):
                self.calls = []

            def invoke(self, messages):
                prompt = messages[0].content if messages else ""
                self.calls.append(prompt)
                if "MUST use this exact package declaration" in prompt:
                    return DummyResponse("```java\npackage com.example.library;\nclass Book {}\n```\nCODI_FILE_WRITE_COMPLETE")
                return DummyResponse("```java\nclass Book {}\n```\nCODI_FILE_WRITE_COMPLETE")

        executor.llm = PromptAwareLLM()
        executor.dispatcher = DummyDispatcher()

        result = executor._execute_content_first("Write Book.java", "write_file", "Book.java", state)
        self.assertEqual(result["status"], "success")
        self.assertIn("package com.example.library;", executor.dispatcher.calls[0][2])

    def test_write_file_reports_java_structural_warning_for_unbalanced_braces(self):
        payload = write_file({"path": "Broken.java", "content": "class Broken {\n    void method() {\n"})
        result = json.loads(payload)
        self.assertTrue(result["success"])
        self.assertIn("syntax_warning", result)
        self.assertIn("brace", result["syntax_warning"].lower())

    def test_validator_falls_back_to_java_brace_check_without_pom(self):
        state = RunState()
        state.tool_results = [
            type("ToolResult", (), {"tool": "write_file", "status": "ok", "output": json.dumps({"success": True, "file_modified": "Broken.java"})})()
        ]

        with open("Broken.java", "w", encoding="utf-8") as handle:
            handle.write("class Broken {\n    void method() {\n")

        validator = Validator()
        reason = validator._java_compile_check(state)
        self.assertIn("brace", reason.lower())

        import os
        if os.path.exists("Broken.java"):
            os.remove("Broken.java")


if __name__ == "__main__":
    unittest.main()
