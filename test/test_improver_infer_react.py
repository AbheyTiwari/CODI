import json
from core.improver import Improver
from state.temp_db import RunState
from tools.registry import ToolRegistry


def test_improver_infers_react_from_package_json(tmp_path, monkeypatch):
    wd = tmp_path / "proj"
    wd.mkdir()
    pkg = {
        "name": "my-app",
        "dependencies": {
            "react": "^18.0.0",
            "react-dom": "^18.0.0"
        }
    }
    pkg_path = wd / "package.json"
    pkg_path.write_text(json.dumps(pkg))

    monkeypatch.chdir(wd)
    monkeypatch.setenv("CODI_WORKING_DIR", str(wd))

    state = RunState()
    imp = Improver(ToolRegistry())
    imp._extract_requirements(state)

    assert state.requirements.framework == "react"
