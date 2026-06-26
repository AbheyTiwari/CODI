import ast
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

    def test_no_duplicate_class_names_across_source(self):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        duplicate_classes = {}

        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Exclude virtualenv, caches, and test artifacts
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".venv", "venv", "codi.egg-info"}]
            if "test" in dirpath.split(os.sep):
                continue
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                filepath = os.path.join(dirpath, filename)
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                try:
                    tree = ast.parse(text)
                except SyntaxError:
                    self.fail(f"Could not parse Python file: {filepath}")
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        duplicate_classes.setdefault(node.name, []).append(filepath)

        duplicates = {name: paths for name, paths in duplicate_classes.items() if len(paths) > 1}
        if duplicates:
            formatted = "\n".join(f"{name}: {paths}" for name, paths in sorted(duplicates.items()))
            self.fail(f"Duplicate class names detected across source files:\n{formatted}")


if __name__ == "__main__":
    unittest.main()
