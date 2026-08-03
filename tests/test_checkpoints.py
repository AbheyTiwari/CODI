from __future__ import annotations

import json

from state.checkpoints import checkpoint_before_write


def test_checkpoint_preserves_existing_file_once(tmp_path, monkeypatch):
    monkeypatch.setenv("CODI_WORKING_DIR", str(tmp_path))
    monkeypatch.delenv("CODI_RUN_ID", raising=False)
    target = tmp_path / "example.py"
    target.write_text("before", encoding="utf-8")

    first = checkpoint_before_write("write_file", {"path": "example.py"})
    target.write_text("after", encoding="utf-8")
    second = checkpoint_before_write("write_file", {"path": "example.py"})

    assert first == second
    backup = tmp_path / ".codi" / "checkpoints" / first["run_id"] / "files" / "example.py"
    assert backup.read_text(encoding="utf-8") == "before"
    manifest = json.loads((backup.parents[1] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest[0]["existed"] == "true"


def test_checkpoint_marks_new_file_without_creating_it(tmp_path, monkeypatch):
    monkeypatch.setenv("CODI_WORKING_DIR", str(tmp_path))
    monkeypatch.delenv("CODI_RUN_ID", raising=False)

    checkpoint = checkpoint_before_write("create_file", {"path": "new.py"})

    marker = tmp_path / ".codi" / "checkpoints" / checkpoint["run_id"] / "created" / "new.py"
    assert marker.exists()
    assert not (tmp_path / "new.py").exists()
