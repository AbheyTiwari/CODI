import unittest

from context_trimmer import trim_context_for_llm
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


if __name__ == "__main__":
    unittest.main()
