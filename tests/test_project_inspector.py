from __future__ import annotations

from tools.local.project_inspector import inspect_project


def test_project_inspection_discovers_repository_instructions(tmp_path):
    (tmp_path / "agents.md").write_text("# Repository instructions", encoding="utf-8")
    (tmp_path / "main.py").write_text("print('ok')", encoding="utf-8")

    result = inspect_project({"path": str(tmp_path)})

    assert result["success"] is True
    assert "agents.md" in result["instructions"]


def test_project_inspection_excludes_codi_runtime_artifacts(tmp_path):
    (tmp_path / "main.py").write_text("print('ok')", encoding="utf-8")
    checkpoint = tmp_path / ".codi" / "checkpoints" / "run" / "files"
    checkpoint.mkdir(parents=True)
    (checkpoint / "main.py").write_text("stale backup", encoding="utf-8")

    result = inspect_project({"path": str(tmp_path)})

    assert result["files"] == ["main.py"]
