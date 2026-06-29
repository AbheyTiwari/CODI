import unittest

from context_trimmer import trim_context_for_llm
from dispatcher import Dispatcher


class DispatcherTruncationTests(unittest.TestCase):
    def test_truncated_json_keeps_content_unmodified_and_records_warning(self):
        raw = '{"content":"hello world"}' + ("x" * 1000)

        parsed = Dispatcher.parse_llm_json(raw)

        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["content"], "hello world")
        self.assertIn("truncation_warning", parsed)

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
