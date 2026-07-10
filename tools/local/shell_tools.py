# tools/local/shell_tools.py

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid

from context_trimmer import trim_tool_output
from logger import log

# Shell execution is intentionally constrained so destructive commands are
# blocked before they can run and callers receive structured feedback.
DANGEROUS_PATTERNS = [
    "rm -rf", "rm -f", "mkfs", "dd if=",
    "chmod 777", "> /dev/", "format c:",
    "DROP TABLE", "DELETE FROM", ":(){:|:&};:",
]

# Commands that install/uninstall packages into a Python venv. Used to detect
# the "CODI is trying to modify its own running venv" deadlock below.
_PACKAGE_MANAGER_PATTERNS = (
    "pip install", "pip uninstall", "pip3 install", "pip3 uninstall",
    "uv pip", "uv sync", "uv add", "uv remove",
    "poetry install", "poetry add", "poetry remove",
)

# Set CODI_AUTO_APPROVE_SHELL=1 to skip the interactive permission prompt
# (e.g. for CI or fully unattended runs). Default is to always ask.
_AUTO_APPROVE_EXTERNAL_SHELL = os.environ.get("CODI_AUTO_APPROVE_SHELL", "0").lower() in ("1", "true", "yes")

_EXTERNAL_SHELL_TIMEOUT_SECONDS = int(os.environ.get("CODI_EXTERNAL_SHELL_TIMEOUT", "300"))


def _working_dir() -> str:
    return os.environ.get("CODI_WORKING_DIR", os.getcwd())


def _targets_own_running_venv(command: str) -> bool:
    """
    True if this command is a pip/uv/poetry install-or-uninstall that would
    modify the EXACT Python venv CODI itself is currently running in.

    Windows (and similarly macOS/Linux) locks loaded .pyd/.dll/.so files
    against deletion or replacement while any process — including CODI's
    own process — has them mapped into memory. No amount of retrying,
    wrapping in a different shell, or opening a new terminal window fixes
    this: the lock is held by CODI itself, not by whatever shell is trying
    to run the install command. The only real fix is to exit CODI first.
    Detecting this up front turns 3+ wasted correction cycles (each
    guessing "close your IDE" style fixes that don't apply) into one
    clear, honest message.
    """
    lowered = (command or "").lower()
    if not any(pattern in lowered for pattern in _PACKAGE_MANAGER_PATTERNS):
        return False

    try:
        own_venv = os.path.normcase(os.path.abspath(sys.prefix))
    except Exception:
        return False

    working_dir = os.path.normcase(os.path.abspath(_working_dir()))

    # Common case: CODI's working directory sits inside (or matches) CODI's
    # own venv root — e.g. D:\CODI\test\... resolving requirements against
    # D:\CODI\.venv, as in the reported bug.
    return own_venv in lowered or own_venv.startswith(working_dir) or working_dir.startswith(own_venv)


def _own_venv_lock_error(command: str, tool_name: str) -> str:
    return json.dumps({
        "success": False,
        "tool": tool_name,
        "error": (
            "This command would modify the Python venv CODI itself is currently "
            f"running in ({sys.prefix}). Windows (and similarly macOS/Linux) locks "
            "loaded .pyd/.dll/.so files while any process holds them open — this "
            "will fail every time while CODI is running, no matter how many times "
            "it's retried or which shell runs it. Exit CODI first (/quit), then run "
            "this command yourself, or point it at a different virtual environment."
        ),
        "command": command,
    })


def run_command(args: dict) -> str:
    """Run a shell command in the project directory (hidden, in-process). Returns stdout + stderr."""
    command = args.get("command", "")
    if not command:
        return "ERROR: no command provided"

    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in command.lower():
            log("tool_call", {"tool": "run_command", "status": "BLOCKED", "input": command})
            return f"BLOCKED: dangerous pattern '{pattern}'"

    if _targets_own_running_venv(command):
        log("tool_call", {"tool": "run_command", "status": "BLOCKED_OWN_VENV", "input": command})
        return _own_venv_lock_error(command, "run_command")

    log("tool_call", {"tool": "run_command", "input": command})
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            cwd=_working_dir(),
        )
        raw = "\n".join(filter(None, [result.stdout.strip(), result.stderr.strip()])) or "(no output)"
        output = trim_tool_output(raw, max_tokens=600)
        exit_code = result.returncode
        log("tool_result", {"tool": "run_command", "status": "ok", "output": output[:200]})
        return json.dumps({
            "success":   exit_code == 0,
            "tool":      "run_command",
            "command":   command,
            "exit_code": exit_code,
            "output":    output,
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "tool": "run_command", "error": "timed out after 60s", "command": command})
    except Exception as e:
        return json.dumps({"success": False, "tool": "run_command", "error": str(e), "command": command})


def _ask_permission(command: str) -> bool:
    """
    Ask the user, directly on the terminal, whether CODI may open a separate
    shell window and run this command. Blocking by design — CODI's CLI is
    interactive, so this is safe to call from a tool handler.

    Set CODI_AUTO_APPROVE_SHELL=1 to skip this (unattended/CI use).
    """
    if _AUTO_APPROVE_EXTERNAL_SHELL:
        log("shell_permission", {"command": command[:200], "auto_approved": True})
        return True

    try:
        print()
        print("  ┌─ CODI wants to run a command in a separate shell window ─")
        print(f"  │  {command}")
        print("  └─────────────────────────────────────────────────────────")
        answer = input("  Allow? [y/N]: ").strip().lower()
    except Exception:
        answer = ""

    granted = answer in ("y", "yes")
    log("shell_permission", {"command": command[:200], "granted": granted})
    return granted


def run_command_external(args: dict) -> str:
    """Run a shell command in a SEPARATE, visible terminal window (PowerShell
    on Windows, bash on macOS/Linux) after asking the user for one-time
    permission. Use this instead of run_command when the task needs a real
    interactive terminal (long-running dev servers, installers that prompt,
    anything the user should be able to see/watch live). Output is captured
    by tee-ing to a log file; once the process exits, exit code + full
    output are read back and returned so the agent loop can inspect the
    result and self-correct automatically on failure, same as run_command."""
    command = (args or {}).get("command", "")
    if not command:
        return "ERROR: no command provided"

    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in command.lower():
            log("tool_call", {"tool": "run_command_external", "status": "BLOCKED", "input": command})
            return f"BLOCKED: dangerous pattern '{pattern}'"

    if _targets_own_running_venv(command):
        log("tool_call", {"tool": "run_command_external", "status": "BLOCKED_OWN_VENV", "input": command})
        return _own_venv_lock_error(command, "run_command_external")

    if not _ask_permission(command):
        return json.dumps({
            "success": False,
            "tool": "run_command_external",
            "error": "User denied permission to run this command.",
            "command": command,
        })

    log("tool_call", {"tool": "run_command_external", "input": command})

    work_dir = _working_dir()
    run_id = uuid.uuid4().hex[:8]
    tmp = tempfile.gettempdir()
    log_path = os.path.join(tmp, f"codi_run_{run_id}.log")
    done_path = os.path.join(tmp, f"codi_run_{run_id}.done")

    system = platform.system()

    try:
        if system == "Windows":
            # Tee output to a log file so we can read it back, and drop an
            # exit-code sentinel file once the command finishes — all while
            # the command runs live and visible in its own window.
            escaped_command = command.replace('"', '`"')
            ps_wrapper = (
                f'$ErrorActionPreference = "Continue"; '
                f'Set-Location -Path "{work_dir}"; '
                f'& cmd /c "{escaped_command}" 2>&1 | Tee-Object -FilePath "{log_path}"; '
                f'$code = $LASTEXITCODE; if ($null -eq $code) {{ $code = 0 }}; '
                f'Set-Content -Path "{done_path}" -Value $code'
            )
            subprocess.Popen(
                ["powershell", "-NoExit", "-Command", ps_wrapper],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        elif system == "Darwin":
            # Open a new Terminal.app window running the wrapped command.
            shell_wrapper = (
                f'cd "{work_dir}" && ({command}) 2>&1 | tee "{log_path}"; '
                f'echo $? > "{done_path}"'
            )
            script_path = os.path.join(tmp, f"codi_run_{run_id}.sh")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(f"#!/bin/bash\n{shell_wrapper}\n")
            os.chmod(script_path, 0o755)
            subprocess.Popen(["open", "-a", "Terminal", script_path])
        else:
            # Linux: try a few common terminal emulators.
            shell_wrapper = (
                f'cd "{work_dir}" && ({command}) 2>&1 | tee "{log_path}"; '
                f'echo $? > "{done_path}"; exec bash'
            )
            launched = False
            for term in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
                try:
                    if term == "gnome-terminal":
                        subprocess.Popen([term, "--", "bash", "-c", shell_wrapper])
                    else:
                        subprocess.Popen([term, "-e", f"bash -c '{shell_wrapper}'"])
                    launched = True
                    break
                except FileNotFoundError:
                    continue
            if not launched:
                # No GUI terminal available — fall back to running hidden,
                # still tee'd to the same log path so downstream code is
                # identical either way.
                subprocess.Popen(["bash", "-c", shell_wrapper])
    except Exception as e:
        return json.dumps({
            "success": False,
            "tool": "run_command_external",
            "error": f"Failed to launch external shell: {e}",
            "command": command,
        })

    # ── Poll for the completion sentinel ────────────────────────────────────
    waited = 0.0
    interval = 0.5
    while waited < _EXTERNAL_SHELL_TIMEOUT_SECONDS:
        if os.path.exists(done_path):
            break
        time.sleep(interval)
        waited += interval

    if not os.path.exists(done_path):
        return json.dumps({
            "success": False,
            "tool": "run_command_external",
            "error": (
                f"Command timed out after {_EXTERNAL_SHELL_TIMEOUT_SECONDS}s "
                f"(it may still be running in its own window)."
            ),
            "command": command,
            "log_path": log_path,
        })

    try:
        with open(done_path, "r", encoding="utf-8", errors="replace") as f:
            exit_code_raw = f.read().strip()
        exit_code = int(exit_code_raw) if exit_code_raw.lstrip("-").isdigit() else -1
    except Exception:
        exit_code = -1

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            raw_output = f.read()
    except Exception:
        raw_output = "(no output captured)"

    output = trim_tool_output(raw_output.strip() or "(no output)", max_tokens=600)

    for p in (log_path, done_path):
        try:
            os.remove(p)
        except Exception:
            pass

    success = exit_code == 0
    log("tool_result", {
        "tool": "run_command_external",
        "status": "ok" if success else "error",
        "exit_code": exit_code,
    })

    payload = {
        "success":   success,
        "tool":      "run_command_external",
        "command":   command,
        "exit_code": exit_code,
        "output":    output,
    }

    if not success:
        # Prefix with ERROR so the dispatcher marks this result "error" and
        # the existing validator/improver correction loop engages
        # automatically — same convention already used by WRITE REJECTED /
        # BLOCKED elsewhere in this codebase. This is what makes the "read
        # the output and try to fix it autonomously" behavior work without
        # any changes to the improve/validate loop itself.
        return f"ERROR: command failed (exit {exit_code})\n{json.dumps(payload)}"
    return json.dumps(payload)


def register_shell_tools(registry):
    registry.register_local("run_command", run_command)
    registry.register_local("run_command_external", run_command_external)