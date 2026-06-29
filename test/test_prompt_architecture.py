import importlib

from core.prompts import correction_system_prompt, executor_system_prompt, planner_system_prompt


def test_executor_prompt_mentions_dispatcher_architecture():
    prompt = executor_system_prompt(["write_file"])
    assert "dispatcher" in prompt.lower()
    assert "micro-task" in prompt.lower() or "single task" in prompt.lower()
    assert "json" in prompt.lower()


def test_planner_prompt_mentions_planning_and_validation_flow():
    prompt = planner_system_prompt()
    assert "plan" in prompt.lower()
    assert "validator" in prompt.lower()
    assert "next-step" in prompt.lower() or "next step" in prompt.lower()


def test_correction_prompt_mentions_improver_repair_flow():
    prompt = correction_system_prompt()
    assert "improver" in prompt.lower()
    assert "validator" in prompt.lower()
    assert "fix" in prompt.lower()


def test_validator_prompt_mentions_verification_and_improver_handoff():
    validator_module = importlib.import_module("core.validator")
    prompt = validator_module._VALIDATE_PROMPT
    assert "validator" in prompt.lower()
    assert "improver" in prompt.lower()
    assert "verify" in prompt.lower() or "validation" in prompt.lower()
