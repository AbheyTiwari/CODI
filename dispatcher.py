# dispatcher.py
# ─────────────────────────────────────────────────────────────────────────────
# THE HEART OF THE SYSTEM.
#
# Receives a JSON action bundle from the planner/executor.
# Routes each action to the correct tool (local or MCP).
# Executes actions in parallel where possible.
# Returns a normalized JSON response.
#
# Nothing here talks to an LLM. It is a pure router/executor.
# ─────────────────────────────────────────────────────────────────────────────

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from logger import log
from tools.registry import ToolRegistry


class Dispatcher:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    # ── Public entry point ────────────────────────────────────────────────────

    def dispatch(self, action_bundle: dict) -> dict:
        """
        Accepts a JSON action bundle from the planner.

        Expected format:
        {
            "action": "tool_call",
            "tools": [
                {"name": "read_file",  "args": {"path": "main.py"}},
                {"name": "list_files", "args": {"dir": "."}}
            ]
        }

        Returns:
        {
            "status": "success" | "partial" | "error",
            "results": [
                {"tool": "read_file",  "status": "ok",    "output": "..."},
                {"tool": "list_files", "status": "error", "output": "..."}
            ]
        }
        """
        action = action_bundle.get("action", "tool_call")

        if action == "tool_call":
            return self._execute_tools(action_bundle.get("tools", []))

        if action == "noop":
            return {"status": "success", "results": []}

        log("dispatcher_unknown_action", {"action": action})
        return {"status": "error", "results": [], "error": f"Unknown action: {action}"}

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_tools(self, tool_calls: list[dict]) -> dict:
        if not tool_calls:
            return {"status": "success", "results": []}

        results = []

        # Single tool — run inline, no thread overhead
        if len(tool_calls) == 1:
            results.append(self._run_one(tool_calls[0]))
        else:
            # Multiple tools — run in parallel
            with ThreadPoolExecutor(max_workers=min(len(tool_calls), 6)) as pool:
                futures = {
                    pool.submit(self._run_one, tc): tc
                    for tc in tool_calls
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        tc = futures[future]
                        results.append({
                            "tool":   tc.get("name", "unknown"),
                            "status": "error",
                            "output": f"Executor crash: {e}",
                        })

        # Determine overall status
        statuses = [r["status"] for r in results]
        if all(s == "ok" for s in statuses):
            overall = "success"
        elif any(s == "ok" for s in statuses):
            overall = "partial"
        else:
            overall = "error"

        log("dispatcher_result", {
            "tools": [r["tool"] for r in results],
            "overall": overall,
        })

        return {"status": overall, "results": results}

    def _run_one(self, tool_call: dict) -> dict:
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})

        log("dispatcher_call", {"tool": name, "args": str(args)[:200]})

        handler = self.registry.get(name)
        if handler is None:
            log("dispatcher_not_found", {"tool": name})
            return {
                "tool":   name,
                "status": "error",
                "output": f"Tool not found: '{name}'. Available: {self.registry.list_names()}",
            }

        try:
            output = handler(args)
            output_text = str(output)
            status = "error" if output_text.startswith(("ERROR", "WRITE REJECTED", "BLOCKED")) else "ok"
            log("dispatcher_ok", {"tool": name, "status": status, "output_len": len(output_text)})
            return {
                "tool":   name,
                "status": status,
                "output": output_text,
            }
        except Exception as e:
            log("dispatcher_error", {"tool": name, "error": str(e)})
            return {
                "tool":   name,
                "status": "error",
                "output": f"Tool error: {e}",
            }

    # ── Convenience: parse JSON string from LLM output ───────────────────────

    @staticmethod
    def parse_llm_json(raw: str) -> dict | None:
        """
        Safely parse a JSON blob from LLM output.
        Strips markdown fences if present.
        Returns None on failure — never crashes.
        """
        if not raw:
            return None
        text = raw.strip()
        # Strip ```json ... ``` fences
        if text.startswith("```"):
            lines = text.splitlines()
            text  = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find first {...} block in the string
            start = text.find("{")
            end   = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end+1])
                except json.JSONDecodeError:
                    pass
        log("dispatcher_parse_fail", {"raw": raw[:200]})
        return None
