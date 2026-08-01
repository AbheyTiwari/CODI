"""Regression tests for action normalization and local tool contracts."""

from __future__ import annotations

from dispatcher import Dispatcher
from tools.registry import ToolRegistry


def _dispatcher_with_recorder():
    calls: list[dict] = []
    registry = ToolRegistry()
    registry.register_local("create_file", lambda args: calls.append(args) or '{"success": true}')
    return Dispatcher(registry), calls


def test_direct_tool_action_unwraps_args_before_dispatch():
    dispatcher, calls = _dispatcher_with_recorder()

    result = dispatcher.dispatch({
        "action": "create_file",
        "reason": "Create the requested file.",
        "args": {"path": "plan.md", "content": "# Plan"},
    })

    assert result["status"] == "success"
    assert calls == [{"path": "plan.md", "content": "# Plan"}]


def test_direct_tool_action_keeps_legacy_root_arguments():
    dispatcher, calls = _dispatcher_with_recorder()

    result = dispatcher.dispatch({"action": "create_file", "path": "plan.md"})

    assert result["status"] == "success"
    assert calls == [{"path": "plan.md"}]


def test_direct_tool_action_normalizes_content_lines():
    dispatcher, calls = _dispatcher_with_recorder()

    result = dispatcher.dispatch({
        "action": "create_file",
        "args": {"path": "plan.md", "content_lines": ["# Plan", "- first step"]},
    })

    assert result["status"] == "success"
    assert calls == [{"path": "plan.md", "content": "# Plan\n- first step"}]


def test_nested_tool_arguments_are_flattened_before_dispatch():
    dispatcher, calls = _dispatcher_with_recorder()

    result = dispatcher.dispatch({
        "action": "tool_call",
        "tools": [{"name": "create_file", "args": {"args": {"path": "plan.md"}}}],
    })

    assert result["status"] == "success"
    assert calls == [{"path": "plan.md"}]


def test_invalid_required_argument_is_rejected_before_the_handler_runs():
    dispatcher, calls = _dispatcher_with_recorder()

    result = dispatcher.dispatch({
        "action": "tool_call",
        "tools": [{"name": "create_file", "args": {"path": 42}}],
    })

    assert result["status"] == "error"
    assert "Tool contract error" in result["results"][0]["output"]
    assert calls == []
