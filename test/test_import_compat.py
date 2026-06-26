import importlib
import inspect
import os
import unittest


class ImportCompatTests(unittest.TestCase):
    def test_root_validator_reexports_core_validator(self):
        validator_module = importlib.import_module("validator")
        validator_cls = validator_module.Validator
        self.assertEqual(validator_cls.__module__, "core.validator")
        self.assertTrue(os.path.normpath(inspect.getsourcefile(validator_cls)).endswith(os.path.normpath("core/validator.py")))

    def test_root_executor_reexports_core_executor(self):
        executor_module = importlib.import_module("executor")
        executor_cls = executor_module.Executor
        self.assertEqual(executor_cls.__module__, "core.executor")
        self.assertTrue(os.path.normpath(inspect.getsourcefile(executor_cls)).endswith(os.path.normpath("core/executor.py")))


if __name__ == "__main__":
    unittest.main()
