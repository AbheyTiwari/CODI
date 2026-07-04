from core.executor import Executor
from tools.registry import ToolRegistry
from state.temp_db import RunState


def test_executor_allows_small_mount_shell_html():
    ex = Executor(ToolRegistry())
    state = RunState()
    state.requirements.framework = "react"

    action_bundle = {"tools": [{"name": "write_file", "args": {"path": "index.html", "content": "<html>\n<body>\n<div id=\"root\"></div>\n</body>\n</html>"}}]}

    violation = ex._framework_violation(state, action_bundle)
    assert violation is None


def test_executor_flags_substantive_html():
    ex = Executor(ToolRegistry())
    state = RunState()
    state.requirements.framework = "react"

    # Substantive HTML with header and many lines
    content = "\n".join(["<html>"] + [f"<div>line {i}</div>" for i in range(30)] + ["</html>"])
    action_bundle = {"tools": [{"name": "write_file", "args": {"path": "index.html", "content": content}}]}

    violation = ex._framework_violation(state, action_bundle)
    assert violation is not None
