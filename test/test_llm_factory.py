import unittest
from unittest.mock import patch

import llm_factory


class LLMFactoryTests(unittest.TestCase):
    def test_local_mode_without_ollama_returns_fallback(self):
        with patch.object(llm_factory, "MODE", "local"), patch.object(llm_factory, "_ollama_is_running", return_value=False):
            llm = llm_factory.get_coder_llm()
        self.assertIsInstance(llm, llm_factory._FallbackLLM)


if __name__ == "__main__":
    unittest.main()
