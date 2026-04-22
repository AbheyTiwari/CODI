# tools/local/shell_tools.py

import os
import subprocess

from context_trimmer import trim_tool_output
from logger import log

DANGEROUS_PATTERNS = [
    "rm -rf", "rm -f", "mkfs", "dd if=",
    "chmod 777", "> /dev/", "format c:",
    "DROP TABLE", "DELETE FROM", ":(){:|:&};:",
]


def _working_dir() -> str:
    return os.environ.get("CODI_WORKING_DIR", os.getcwd())


def run_command(args: dict) -> str:
    """Run a shell command in the project directory. Returns stdout + stderr."""
    command = args.get("command", "")
    if not command:
        return "ERROR: no command provided"

    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in command.lower():
            log("tool_call", {"tool": "run_command", "status": "BLOCKED", "input": command})
            return f"BLOCKED: dangerous pattern '{pattern}'"

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
        log("tool_result", {"tool": "run_command", "status": "ok", "output": output[:200]})
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out after 60s"
    except Exception as e:
        return f"ERROR: {e}"


def register_shell_tools(registry):
    registry.register_local("run_command", run_command)
