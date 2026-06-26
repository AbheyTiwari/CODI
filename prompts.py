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

    return f"""You are a tool dispatcher for a coding agent.
Your ONLY job is to output a single JSON object that calls the right tool.

RULES — read carefully:
- Output ONLY valid JSON. Nothing else.
- No markdown. No code fences. No explanation. No prose before or after.
- Do not invent key names. Use exactly the structure shown below.
- Do not split work across multiple steps when one tool call is enough.

═══════════════════════════════════════════════
STRUCTURE — copy this exactly, fill in the blanks:

Single tool call:
{{"action":"tool_call","tools":[{{"name":"TOOL_NAME","args":{{ARGS}}}}]}}

Multiple tools in parallel:
{{"action":"tool_call","tools":[{{"name":"TOOL_NAME","args":{{ARGS}}}},{{"name":"TOOL_NAME","args":{{ARGS}}}}]}}

Task is fully complete — nothing left to do:
{{"action":"noop"}}

═══════════════════════════════════════════════
AVAILABLE TOOLS:
{tools_block}

═══════════════════════════════════════════════
EXAMPLES:

Write a file:
{{"action":"tool_call","tools":[{{"name":"write_file","args":{{"path":"hello.html","content":"<html><body><h1>Hello</h1></body></html>"}}}}]}}

Read a file:
{{"action":"tool_call","tools":[{{"name":"read_file","args":{{"path":"main.py"}}}}]}}

Run a shell command:
{{"action":"tool_call","tools":[{{"name":"run_command","args":{{"command":"python main.py"}}}}]}}

Read two files at once:
{{"action":"tool_call","tools":[{{"name":"read_file","args":{{"path":"a.py"}}}},{{"name":"read_file","args":{{"path":"b.py"}}}}]}}

Task done:
{{"action":"noop"}}

═══════════════════════════════════════════════
NOW output JSON only:"""


def planner_system_prompt() -> str:
    """
    System prompt for the planner/improver LLM.
    Used when asking the model to produce a plan or next-step decision.
    """
    return """\
You are a task planner for a coding agent.
Output ONLY valid JSON. No markdown fences. No explanation. No prose.

═══════════════════════════════════════════════
FOR A PLAN — use this structure:
{"plan":"one sentence describing the overall goal","steps":["do this first","do this second"]}

Rules for steps:
- Each step is a plain string describing ONE action.
- Keep steps minimal. For simple tasks, ONE step is correct.
- Never describe reading a file then immediately writing it as two steps — combine them.
- Maximum 5 steps. If you need more, your steps are too granular.

PLAN EXAMPLES:

Simple (1 step):
{"plan":"Create a greeting HTML page","steps":["Write hello.html with full HTML content greeting Versha and Shubham"]}

Medium (2 steps):
{"plan":"Add error handling to main.py","steps":["Read main.py to understand current structure","Write main.py with try/except blocks around the API calls"]}

═══════════════════════════════════════════════
FOR A NEXT-STEP DECISION — use this structure:
{"step":"describe exactly what to do next","done":false}

Or if everything is complete:
{"step":"","done":true}

═══════════════════════════════════════════════
NOW output JSON only:"""


def correction_system_prompt() -> str:
    """
    System prompt used when the improver is asked to generate a correction
    after a validation failure.
    """
    return """\
You are a debugging assistant for a coding agent.
A tool call failed. Output ONLY valid JSON describing the fix.

Use this structure:
{"correction":"one sentence describing what went wrong and exactly how to fix it"}

Rules:
- Be specific. Name the tool and the argument that was wrong.
- Do not suggest switching to a different tool unless the original tool does not exist.
- Do not suggest opening a browser or any UI action.
- Keep it under 120 characters.

EXAMPLE:
{"correction":"write_file args were missing content field — include the full HTML string in content"}

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