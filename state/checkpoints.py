"""Non-destructive per-file checkpoints for agent-initiated mutations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_MUTATING_TOOLS = {"create_file", "write_file", "edit_file", "apply_patch"}
_LOCK = threading.Lock()


def _root() -> Path:
    return Path(os.environ.get("CODI_WORKING_DIR", os.getcwd())).resolve()


def _run_id() -> str:
    existing = os.environ.get("CODI_RUN_ID")
    if existing:
        return existing
    created = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    os.environ["CODI_RUN_ID"] = created
    return created


def checkpoint_before_write(tool: str, args: Any) -> dict[str, str] | None:
    """Snapshot a file before a mutation; never modifies the target itself."""
    if tool not in _MUTATING_TOOLS or not isinstance(args, dict):
        return None
    raw_path = args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    root = _root()
    target = Path(raw_path)
    target = target if target.is_absolute() else root / target
    try:
        target = target.resolve()
        relative = target.relative_to(root)
    except (OSError, ValueError):
        return None

    run_id = _run_id()
    checkpoint_root = root / ".codi" / "checkpoints" / run_id
    backup = checkpoint_root / "files" / relative
    manifest_path = checkpoint_root / "manifest.json"
    with _LOCK:
        # Preserve the first pre-mutation version only; later writes in the
        # same agent run must still be reversible to the original user state.
        if backup.exists() or (checkpoint_root / "created" / relative).exists():
            return {"run_id": run_id, "path": relative.as_posix()}
        existed = target.is_file()
        if existed:
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        else:
            marker = checkpoint_root / "created" / relative
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            digest = ""
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, str]] = []
        if manifest_path.exists():
            records = json.loads(manifest_path.read_text(encoding="utf-8"))
        records.append({
            "tool": tool,
            "path": relative.as_posix(),
            "existed": str(existed).lower(),
            "sha256": digest,
        })
        manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return {"run_id": run_id, "path": relative.as_posix()}


def list_checkpoints() -> list[dict[str, Any]]:
    """List recoverable agent checkpoints without changing the workspace."""
    base = _root() / ".codi" / "checkpoints"
    if not base.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for directory in sorted(base.iterdir(), reverse=True):
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            records = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        results.append({"run_id": directory.name, "files": records})
    return results


def restore_checkpoint_file(run_id: str, path: str) -> dict[str, str]:
    """Restore one explicitly selected file from its pre-write snapshot."""
    root = _root()
    checkpoint_root = root / ".codi" / "checkpoints" / run_id
    try:
        requested = Path(path).as_posix()
        target = (root / requested).resolve()
        relative = target.relative_to(root).as_posix()
    except (OSError, ValueError):
        return {"success": "false", "error": "path escapes project directory"}
    manifest_path = checkpoint_root / "manifest.json"
    if not manifest_path.is_file():
        return {"success": "false", "error": f"checkpoint not found: {run_id}"}
    try:
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"success": "false", "error": "checkpoint manifest is unreadable"}
    record = next((item for item in records if item.get("path") == relative), None)
    if not record:
        return {"success": "false", "error": f"file not present in checkpoint: {relative}"}

    with _LOCK:
        if record.get("existed") == "true":
            backup = checkpoint_root / "files" / relative
            if not backup.is_file():
                return {"success": "false", "error": f"backup data missing for {relative}"}
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)
            action = "restored"
        elif target.exists():
            if target.is_dir():
                return {"success": "false", "error": "refusing to remove a directory"}
            target.unlink()
            action = "removed newly-created file"
        else:
            action = "already absent"
    return {"success": "true", "path": relative, "action": action, "run_id": run_id}
