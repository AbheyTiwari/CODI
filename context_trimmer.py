# context_trimmer.py
# Sits between MCP tool output and the LLM.
# Prevents 413 / token-overflow errors by aggressively pruning context.

import re

# Rough chars-to-tokens ratio for LLaMA/Qwen family models
CHARS_PER_TOKEN = 3.5

def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)

def trim_tool_output(output: str, max_tokens: int = 600) -> str:
    """
    Trim a single MCP tool output to fit within max_tokens.
    Keeps the head and tail so context is not completely lost.
    """
    if not output:
        return output
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    if len(output) <= max_chars:
        return output
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        output[:head_chars]
        + f"\n... [TRIMMED {len(output) - max_chars} chars] ...\n"
        + output[-tail_chars:]
    )

def trim_history(history: str, max_tokens: int = 1200) -> str:
    """
    Trim conversation history to last N tokens worth of lines.
    Preserves the [Historical Summary] block if present.
    """
    if not history or estimate_tokens(history) <= max_tokens:
        return history

    summary = ""
    body = history

    # Preserve summary block if it exists
    if history.startswith("[Historical Summary]"):
        parts = history.split("---", 1)
        if len(parts) == 2:
            summary = parts[0] + "---\n"
            body = parts[1]

    lines = body.strip().split("\n")
    max_chars = int(max_tokens * CHARS_PER_TOKEN)

    # Walk from the end, accumulate until budget is used
    kept = []
    used = 0
    for line in reversed(lines):
        cost = len(line) + 1
        if used + cost > max_chars:
            break
        kept.append(line)
        used += cost

    kept.reverse()
    trimmed_body = "\n".join(kept)

    if summary:
        return summary + trimmed_body
    return trimmed_body

def trim_context_for_llm(
    user_input: str,
    history: str,
    tool_outputs: list[str],
    system_prompt: str = "",
    mode: str = "hybrid",
) -> dict:
    """
    Master trim function called before every LLM invocation.
    Returns dict with trimmed fields + a token budget warning if tight.

    Token budgets (rough):
      local   — 3500 total context
      hybrid  — 3500 for local calls; cloud gets full 8000
      cloud   — 8000 total context
      air     — 2000 total (phones are memory-constrained)
    """
    budgets = {
        "local":  3500,
        "hybrid": 3500,   # local-tier calls; cloud calls skip this trimmer
        "cloud":  8000,
        "air":    2000,
    }
    total_budget = budgets.get(mode, 3500)

    # Reserve tokens for: system prompt, user input, response headroom
    system_tokens   = estimate_tokens(system_prompt)
    input_tokens    = estimate_tokens(user_input)
    headroom        = 800   # room for the model's reply
    reserved        = system_tokens + input_tokens + headroom

    remaining = max(total_budget - reserved, 400)

    # Split remaining between history (40%) and tool outputs (60%)
    history_budget = int(remaining * 0.40)
    tools_budget   = int(remaining * 0.60)
    per_tool       = max(tools_budget // max(len(tool_outputs), 1), 200)

    trimmed_history = trim_history(history, max_tokens=history_budget)
    trimmed_tools   = [trim_tool_output(o, max_tokens=per_tool) for o in tool_outputs]

    total_used = (
        system_tokens
        + input_tokens
        + estimate_tokens(trimmed_history)
        + sum(estimate_tokens(o) for o in trimmed_tools)
    )

    warning = ""
    if total_used > total_budget * 0.90:
        warning = f"[Context at {int(total_used/total_budget*100)}% capacity — consider /clear]"

    return {
        "history":      trimmed_history,
        "tool_outputs": trimmed_tools,
        "token_estimate": total_used,
        "warning":      warning,
    }