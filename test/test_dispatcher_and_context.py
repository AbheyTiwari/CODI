import unittest

from context_trimmer import trim_context_for_llm
from core import executor as executor_module
from core.executor import (
    _detect_file_write_step,
    _estimate_token_count,
    _extract_token_metrics,
    _should_use_content_first,
    Executor,
)
from dispatcher import Dispatcher
from tools.local.file_tools import write_file


class DispatcherTruncationTests(unittest.TestCase):
    def test_truncated_json_keeps_content_unmodified_and_records_warning(self):
        raw = '{"content":"hello world"}' + ("x" * 1000)

        parsed = Dispatcher.parse_llm_json(raw)

        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["content"], "hello world")
        self.assertIn("truncation_warning", parsed)

    def test_truncation_warning_is_attached_to_tool_output(self):
        raw = '{"action":"tool_call","tools":[{"name":"write_file","args":{"path":"demo.txt","content":"hello world"}}],"content":"hello world"}' + ("x" * 1000)

        parsed = Dispatcher.parse_llm_json(raw)

        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["content"], "hello world")

        class DummyRegistry:
            def list_names(self):
                return ["write_file"]

            def get(self, name):
                if name != "write_file":
                    return None
                return lambda args: f"wrote {args['path']}"

        result = Dispatcher(DummyRegistry()).dispatch(parsed)
        self.assertEqual(result["status"], "success")
        self.assertIn("WARNING: content may be truncated, verify the file.", result["results"][0]["output"])

    def test_write_file_returns_error_for_non_mapping_args(self):
        result = write_file(None)
        self.assertIn("ERROR writing file: missing path", result)

    def test_hybrid_mode_uses_cloud_budget_for_context_trimming(self):
        long_text = "abcdefghij" * 800
        hybrid = trim_context_for_llm(
            "task",
            long_text,
            [long_text, long_text],
            mode="hybrid",
        )
        cloud = trim_context_for_llm(
            "task",
            long_text,
            [long_text, long_text],
            mode="cloud",
        )

        self.assertEqual(hybrid["history"], cloud["history"])
        self.assertEqual(hybrid["tool_outputs"], cloud["tool_outputs"])

    def test_large_xml_step_uses_content_first_strategy(self):
        step = "Create pom.xml with Java 16+, JUnit 5, and Lombok dependencies"
        tool, path = _detect_file_write_step(step)

        self.assertEqual(tool, "create_file")
        self.assertEqual(path, "pom.xml")

    def test_simple_file_write_steps_use_content_first_strategy(self):
        step = "Write README.md with a short project summary"
        tool, path = _detect_file_write_step(step)

        self.assertEqual(tool, "write_file")
        self.assertEqual(path, "README.md")
        self.assertTrue(_should_use_content_first(step, path))

    def test_token_metrics_are_extracted_from_response_metadata(self):
        class DummyResponse:
            usage_metadata = {"input_tokens": 123, "output_tokens": 45, "total_tokens": 168}

        metrics = _extract_token_metrics(DummyResponse())

        self.assertEqual(metrics["prompt_tokens"], 123)
        self.assertEqual(metrics["output_tokens"], 45)
        self.assertEqual(metrics["total_tokens"], 168)
        self.assertEqual(_estimate_token_count("one two three"), 3)

    def test_executor_exposes_repair_prompt_constant(self):
        self.assertTrue(hasattr(executor_module, "_REPAIR_PROMPT"))

    def test_single_tool_bundle_is_accepted(self):
        class DummyRegistry:
            def list_names(self):
                return ["write_file"]

            def summary(self):
                return "write_file"

        executor = Executor(DummyRegistry())
        valid, error, tools = executor._validate_atomic_contract(
            "Write demo.txt",
            {"action": "tool_call", "tools": [{"name": "write_file", "args": {"path": "demo.txt", "content": "x"}}]},
            "{}",
            None,
        )
        self.assertTrue(valid)
        self.assertIsNone(error)
        self.assertEqual(len(tools), 1)

    def test_multiple_tool_bundle_is_rejected(self):
        class DummyRegistry:
            def list_names(self):
                return ["write_file"]

            def summary(self):
                return "write_file"

        executor = Executor(DummyRegistry())
        valid, error, tools = executor._validate_atomic_contract(
            "Write demo.txt and readme",
            {"action": "tool_call", "tools": [{"name": "write_file", "args": {"path": "demo.txt", "content": "x"}}, {"name": "write_file", "args": {"path": "readme.txt", "content": "y"}}]},
            "{}",
            None,
        )
        self.assertFalse(valid)
        self.assertIn("Expected exactly one tool call", error)
        self.assertEqual(len(tools), 2)

    def test_child_steps_are_split_for_multiple_files(self):
        class DummyRegistry:
            def list_names(self):
                return ["write_file"]

            def summary(self):
                return "write_file"

        executor = Executor(DummyRegistry())
        children = executor._split_child_steps("Implement Book and Member and Loan")
        self.assertEqual(children, ["Create Implement Book", "Create Member", "Create Loan"])


if __name__ == "__main__":
    unittest.main()
