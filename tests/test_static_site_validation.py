from __future__ import annotations

import json

from tools.local.static_server_tools import shutdown_static_servers, verify_static_site


def test_static_site_validation_checks_http_and_local_assets(tmp_path, monkeypatch):
    (tmp_path / "styles.css").write_text("body { color: black; }", encoding="utf-8")
    (tmp_path / "index.html").write_text(
        '<!doctype html><link rel="stylesheet" href="styles.css"><script src="app.js"></script>',
        encoding="utf-8",
    )
    (tmp_path / "app.js").write_text("console.log('ok');", encoding="utf-8")
    monkeypatch.setenv("CODI_WORKING_DIR", str(tmp_path))

    result = verify_static_site({"path": "index.html"})

    assert '"success": true' in result
    payload = json.loads(result)
    assert payload["javascript_syntax_checked"] in (True, False)
    assert payload["browser_runtime_checked"] is False
    shutdown_static_servers()


def test_static_site_validation_reports_missing_assets(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text('<script src="missing.js"></script>', encoding="utf-8")
    monkeypatch.setenv("CODI_WORKING_DIR", str(tmp_path))

    result = verify_static_site({"path": "index.html"})

    assert '"success": false' in result
    assert "missing.js" in result


def test_static_site_validation_reports_broken_navigation_anchor(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text(
        '<a href="#checkout">Checkout</a><main id="catalog"></main>', encoding="utf-8"
    )
    monkeypatch.setenv("CODI_WORKING_DIR", str(tmp_path))

    result = verify_static_site({"path": "index.html"})

    assert '"success": false' in result
    assert "#checkout" in result
