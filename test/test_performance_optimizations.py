import json
import unittest
from unittest.mock import patch

import dispatcher
import logger
from state.temp_db import RunState


class PerformanceOptimizationTests(unittest.TestCase):
    def test_logging_skips_caller_info_by_default(self):
        with patch.object(logger, "_CALLER_DEBUG", False), \
             patch.object(logger, "_caller_info", side_effect=AssertionError("caller lookup should be skipped")):
            logger.log("test_event", {"value": 1})

    def test_handler_info_is_cached(self):
        dispatcher._handler_info_cache.clear()

        def handler():
            return None

        with patch.object(dispatcher.inspect, "getmodule", return_value=type("Module", (), {"__name__": "demo"})) as getmodule_mock, \
             patch.object(dispatcher.inspect, "getsourcefile", return_value="/tmp/demo.py") as getsourcefile_mock:
            dispatcher._handler_info(handler)
            dispatcher._handler_info(handler)

        self.assertEqual(getmodule_mock.call_count, 1)
        self.assertEqual(getsourcefile_mock.call_count, 1)

    def test_context_snapshot_summarizes_old_results(self):
        state = RunState()
        for index in range(8):
            state.add_tool_result("read_file", "ok", f"content {index}")
        state.add_tool_result("write_file", "ok", json.dumps({"file_modified": "demo.py"}))

        snapshot = state.context_snapshot(max_recent=3)

        self.assertIn("[SUMMARY]", snapshot)
        self.assertIn("read_file", snapshot)
        self.assertIn("write_file", snapshot)
        self.assertIn("demo.py", snapshot)

    def test_tool_history_compresses_after_threshold(self):
        state = RunState()
        for index in range(12):
            state.add_tool_result("read_file", "ok", f"content {index}")

        self.assertGreaterEqual(len(state.tool_results), 4)
        self.assertEqual(state.tool_results[0].tool, "context_summary")
        self.assertIn("[SUMMARY] Earlier tool activity:", state.tool_results[0].output)


if __name__ == "__main__":
    unittest.main()
