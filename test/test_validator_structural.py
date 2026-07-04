import os
import tempfile
from core.validator import Validator
from state.temp_db import RunState, TaskRequirements


def test_validator_reacts_to_missing_package_json(monkeypatch, tmp_path):
    # Create a temporary working dir without package.json
    wd = tmp_path / "proj"
    wd.mkdir()
    monkeypatch.chdir(wd)
    monkeypatch.setenv("CODI_WORKING_DIR", str(wd))

    state = RunState()
    state.requirements = TaskRequirements(framework="react")
    state.plan_steps = []
    state.iteration = 1
    # Ensure some file-write activity exists so deterministic checks don't short-circuit
    state.add_tool_result("create_file", "ok", '{"file_modified":"index.html"}')

    v = Validator()
    passed = v.validate(state)
    assert passed is False
    assert "package.json" in (state.validation_notes or "").lower()
