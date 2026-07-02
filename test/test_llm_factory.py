import unittest
from types import SimpleNamespace
from unittest.mock import patch

import llm_factory


class LLMFactoryTests(unittest.TestCase):
    def test_local_mode_without_ollama_returns_fallback(self):
        with patch.object(llm_factory, "MODE", "local"), patch.object(llm_factory, "_ollama_is_running", return_value=False):
            llm = llm_factory.get_coder_llm()
        self.assertIsInstance(llm, llm_factory._FallbackLLM)

    def test_fallback_llm_raises_clear_error(self):
        llm = llm_factory._FallbackLLM("backend missing")
        with self.assertRaisesRegex(RuntimeError, "backend missing"):
            llm.invoke([])

    def test_thinking_blocks_are_stripped_from_llm_response(self):
        class DummyLLM:
            def invoke(self, *_args, **_kwargs):
                return SimpleNamespace(content="<think>private scratchpad</think>\nHello.")

        llm = llm_factory._ReasoningFilteredLLM(DummyLLM())

        with patch.object(llm_factory, "emit_status"):
            response = llm.invoke([])

        self.assertEqual(response.content, "Hello.")


if __name__ == "__main__":
    unittest.main()
