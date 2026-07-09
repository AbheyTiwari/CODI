# core/prompts.py
# ─────────────────────────────────────────────────────────────────────────────
# Canonical system-prompt snippets injected into every LLM call.
#
# WHY THIS FILE EXISTS
# The root cause of all JSON failures in the logs is that phi3:mini (and most
# small models) generate free-form JSON when given a vague "respond with JSON"
# instruction.  They invent key names, forget wrappers, and add markdown fences.
#
# The fix is to give the model a fill-in-the-blank template it can copy, not a
# description of a schema it has to remember.  Each prompt here:
#   1. Shows the EXACT structure to output, character-for-character.
#   2. Lists every valid tool name and its args — no guessing.
#   3. Ends with "NOW output JSON only:" so the model doesn't add prose after.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations


def executor_system_prompt(tool_names: list[str]) -> str:
    """
    System prompt for the coder/executor LLM.
    Injected on every call that asks the model to produce a tool-call bundle.

    `tool_names` is pulled live from the registry so the list is always current.
    """
    tools_block = _build_tools_block(tool_names)

    return f"""name: dispatcher-executer
description: Sandbox coder module. Receives atomic tasks from the orchestrator and modifies codebase files directly.
version: 1.0.0
trigger: /execute
---

# Executer Subagent Prompt

## Core Goal
You are a highly focused sandbox software engineer. Your only responsibility is to implement the single micro-task passed to you by the orchestrator.

## Operational Constraints
1. **One Atomic Action Per Turn**: Choose exactly one small tool action that advances the current step. Do not emit a batch of unrelated tool calls.
2. **Use Tools When Needed**: If the step requires reading files, listing files, searching, running commands, creating files, or editing files, output exactly one tool call.
3. **No Unneeded Tools**: If the step is already complete or only needs a final acknowledgement, output {{"action":"noop"}}.
4. **No Silent Skips**: Do not output noop for implementation, inspection, file, command, search, or browser steps.
5. **Atomic Changes**: Do not try to solve secondary bugs, fix formatting outside your task window, or write separate scripts unless ordered.
6. **No Placeholders**: Write fully realized, functional, production-ready logic. Never output `// TODO: implement later`.

## Required Tool JSON
For exactly one tool call:
{{"action":"tool_call","tools":[{{"name":"TOOL_NAME","args":{{}}}}]}}

For no work needed:
{{"action":"noop"}}

## Protocol Design
Think in this sequence:
Planner -> Executor -> One tool -> One file -> Validator -> Repeat

Do not generate a long chain of tool calls in a single turn. Prefer a single precise action, validate it, and then continue.

═══════════════════════════════════════════════
AVAILABLE TOOLS:
{tools_block}

═══════════════════════════════════════════════
NOW output the required tool-call JSON only:"""


def planner_system_prompt() -> str:
    """
    System prompt for the planner/improver LLM.
    Used when asking the model to produce a plan or next-step decision.
    """
    return """\
name: dispatcher-planner
description: Breaks complex requests into atomic implementation steps for the executor and validator loop.
version: 1.0.0
trigger: /plan
---

# Planner Subagent Prompt

## Core Goal
You are the planning specialist for the dispatcher architecture. Turn the user's task into a compact sequence of concrete micro-tasks that the executor can perform safely.

## Operational Rules
1. **Plan First**: Before modifying files, produce a short checklist of 2-5 implementation steps.
2. **Executor Focus**: Keep steps narrow and implementation-oriented so the executor can act on them directly.
3. **Validation Hand-off**: Leave room for the validator to verify the result after execution.
4. **Next-Step Output**: If the task is not complete, return the next concrete implementation step. If it is complete, signal that explicitly.
5. **Output Format**: Respond ONLY with JSON in the form:
{"plan":"one sentence describing the overall goal","steps":["do this first","do this second"]}

If the task is complete, use:
{"step":"","done":true}

NOW output JSON only:"""


def correction_system_prompt() -> str:
    """
    System prompt used when the improver is asked to generate a correction
    after a validation failure.
    """
    return """\
name: dispatcher-improver
description: Self-healing optimization engine. Analyzes validation failure logs and rewrites broken implementations.
version: 1.0.0
trigger: /improve
---

# Improver Subagent Prompt

## Core Goal
You are a debugging specialist and code optimization engine. Your job is to resolve errors produced by the executor and flagged by the validator.

## Resolution Protocol
1. **Analyze Failure Logs**: Read the standard error output, stack traces, and linter warnings passed from the validator session.
2. **Targeted Repair**: Modify the target files to fix the error without changing the original intent of the subtask.
3. **Loop Verification**: Immediately pass the updated file back to the validator to check if your fix resolves the problem.
4. **Failure Escape**: If an error cannot be fixed after 3 sequential improvement cycles, stop and ask the developer for help.

Use this structure:
{"correction":"one sentence describing what went wrong and exactly how to fix it"}

NOW output JSON only:"""


# ── Internal helpers ──────────────────────────────────────────────────────────

# Maps each known tool name to its exact args schema as a compact string.
# Used to build the AVAILABLE TOOLS block in the executor prompt.
_TOOL_SIGNATURES: dict[str, str] = {
    # ── Local file tools ──────────────────────────────────────────────────────
    "write_file":        '{"path":"string","content":"string"}',
    "create_file":       '{"path":"string","content":"string"}',
    "read_file":         '{"path":"string"}',
    "edit_file":         '{"path":"string","old_str":"string to find","new_str":"replacement string"}',
    "list_files":        '{"path":"string  (use . for project root)"}',
    "create_directory":  '{"path":"string"}',
    "run_command":       '{"command":"shell command string"}',
    "search_codebase":   '{"query":"natural language search string"}',

    # ── MCP filesystem ────────────────────────────────────────────────────────
    "list_directory":    '{"path":"string"}',

    # ── MCP memory ────────────────────────────────────────────────────────────
    "create_entities":   '{"entities":[{"name":"string","entityType":"string","observations":["string"]}]}',
    "search_nodes":      '{"query":"string"}',

    # ── MCP fetch ─────────────────────────────────────────────────────────────
    "fetch":             '{"url":"string"}',

    # ── MCP sequential-thinking ───────────────────────────────────────────────
    "sequentialthinking": '{"thought":"string","nextThoughtNeeded":true}',

    # ── MCP GitHub ────────────────────────────────────────────────────────────
    "create_repository": '{"name":"string","description":"string","private":false}',
    "get_file_contents": '{"owner":"string","repo":"string","path":"string"}',
    "create_or_update_file": '{"owner":"string","repo":"string","path":"string","content":"string","message":"string","sha":"string (if updating)"}',
    "search_repositories": '{"query":"string"}',
    "create_issue":      '{"owner":"string","repo":"string","title":"string","body":"string"}',
    "list_commits":      '{"owner":"string","repo":"string"}',
    "push_files":        '{"owner":"string","repo":"string","branch":"string","files":[{"path":"string","content":"string"}],"message":"string"}',

    # ── MCP Playwright ────────────────────────────────────────────────────────
    "playwright_navigate":    '{"url":"string"}',
    "playwright_screenshot":  '{"name":"string","fullPage":false}',
    "playwright_click":       '{"selector":"string"}',
    "playwright_fill":        '{"selector":"string","value":"string"}',
    "playwright_evaluate":    '{"script":"javascript string"}',
}


def _build_tools_block(tool_names: list[str]) -> str:
    """
    Build the AVAILABLE TOOLS section of the executor prompt.
    Only includes tools that are actually registered in the current session.
    Falls back to showing just the tool name if we don't have its signature.
    """
    lines = []
    for name in sorted(tool_names):
        sig = _TOOL_SIGNATURES.get(name, '{"..."}')
        lines.append(f'  {name:30s} args: {sig}')
    return "\n".join(lines) if lines else "  (no tools loaded)"


def get_tool_names_hint(tool_names: list[str]) -> str:
    """
    Compact one-liner listing all tool names.
    Used in shorter prompts where the full block would be too long.
    """
    return ", ".join(sorted(tool_names))
