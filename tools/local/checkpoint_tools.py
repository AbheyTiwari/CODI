"""Explicit recovery and change-summary tools for CODI agent runs."""

from __future__ import annotations

import json
import os
import subprocess

from state.checkpoints import list_checkpoints, restore_checkpoint_file


def list_change_checkpoints(_args: dict | None = None) -> str:
    """List agent checkpoints that can be restored explicitly."""
    return json.dumps({"success": True, "checkpoints": list_checkpoints()})


def restore_change_checkpoint(args: dict | None = None) -> str:
    """Restore one file from a checkpoint. Requires confirm='RESTORE'."""
    args = args or {}
    if args.get("confirm") != "RESTORE":
        return json.dumps({"success": False, "error": "Restoration is destructive. Reissue with confirm='RESTORE'."})
    run_id = str(args.get("run_id", ""))
    path = str(args.get("path", ""))
    if not run_id or not path:
        return json.dumps({"success": False, "error": "run_id and path are required"})
    result = restore_checkpoint_file(run_id, path)
    result["success"] = result.get("success") == "true"
    return json.dumps(result)


def summarize_workspace_changes(_args: dict | None = None) -> str:
    """Return a concise Git change summary without modifying the workspace."""
    working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
    try:
        status = subprocess.run(["git", "status", "--short"], cwd=working_dir, capture_output=True, text=True, timeout=15)
        diff = subprocess.run(["git", "diff", "--stat"], cwd=working_dir, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return json.dumps({"success": False, "error": str(exc)})
    if status.returncode or diff.returncode:
        return json.dumps({"success": False, "error": (status.stderr or diff.stderr or "git failed").strip()})
    return json.dumps({"success": True, "changed_files": [line for line in status.stdout.splitlines() if line], "diff_stat": diff.stdout.strip() or "(no tracked differences)"})


def register_checkpoint_tools(registry) -> None:
    registry.register_local("list_change_checkpoints", list_change_checkpoints)
    registry.register_local("restore_change_checkpoint", restore_change_checkpoint)
    registry.register_local("summarize_workspace_changes", summarize_workspace_changes)
