import os
import json
import re
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from tools import get_all_tools
from llm_factory import get_coder_llm, get_refiner_llm
from logger import log

SIMPLE_TRIGGERS = (
    "hello", "hi ", "hey ", "what is", "what are", "who is", "explain",
    "how do", "how does", "tell me", "what's", "whats", "thanks", "thank you",
    "yes", "no", "ok", "okay", "sure", "help", "why ", "when ", "where "
)

ACTION_TRIGGERS = (
    "create", "write", "make", "build", "fix", "edit", "update", "delete",
    "run", "execute", "generate", "refactor", "implement", "add", "code",
    "put", "save", "html", "css", "script", "file", "folder", "index",
    "function", "class", "api", "page", "deploy", "install", "setup",
    "rename", "move", "copy", "read", "open", "parse", "fetch", "download",
    "list", "search", "find", "show", "get", "check", "access", "browse",
    "navigate", "click", "screenshot", "scrape", "query", "lookup", "pull",
    "push", "commit", "clone", "diff", "status", "remember", "store",
    "repo", "repository", "github", "git",
)

def is_simple_input(text: str) -> bool:
    t = text.lower().strip()
    if any(w in t for w in ACTION_TRIGGERS):
        return False
    if len(t) < 80:
        return True
    if any(t.startswith(trigger) for trigger in SIMPLE_TRIGGERS):
        return True
    return False

def _system_prompt() -> str:
    working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
    return (
        f"You are Codi, an offline-first AI coding agent.\n\n"
        f"CURRENT PROJECT DIRECTORY: {working_dir}\n"
        f"All file operations, shell commands, and searches are relative to this directory.\n"
        f"When the user says 'list files', 'read file', or 'run command' — always operate "
        f"in {working_dir} unless they specify a different absolute path.\n\n"
        f"You have tools: run_command, read_file, write_file, list_files, search_codebase, "
        f"and MCP tools (filesystem, git, github, fetch, memory, playwright, etc).\n"
        f"Use tools immediately — do not explain what you're about to do, just do it.\n"
        f"For GitHub operations use the github MCP tools, not the gh CLI.\n"
        f"For git operations on the local repo use run_command with git commands.\n"
    )

# ── Tool call text parser ─────────────────────────────────────────────────────
# When a model outputs raw JSON tool calls as text instead of structured calls,
# this extracts and executes them anyway.

def _extract_tool_calls_from_text(content: str) -> list[dict]:
    """
    Parse raw JSON tool call objects from model text output.
    Handles unescaped quotes inside string values (e.g. print("Hello")).
    Returns list of dicts with 'name' and 'arguments' keys.
    """
    if not content:
        return []

    found = []

    # Extract tool name + raw argument blob using named groups
    # Captures everything between the outer braces of "arguments": {...}
    pattern = re.compile(
        r'"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*(\{.*?\})',
        re.DOTALL
    )

    for m in re.finditer(pattern, content):
        tool_name = m.group(1)
        raw_args  = m.group(2)

        # Try standard parse first
        try:
            args = json.loads(raw_args)
            found.append({"name": tool_name, "arguments": args})
            continue
        except json.JSONDecodeError:
            pass

        # Fallback: extract key-value pairs manually
        # Handles: "key": "value with unescaped "quotes" inside"
        args = {}
        kv_pattern = re.compile(r'"(\w+)"\s*:\s*"(.*?)"(?=\s*[,}])', re.DOTALL)
        for kv in re.finditer(kv_pattern, raw_args):
            args[kv.group(1)] = kv.group(2)

        if args:
            found.append({"name": tool_name, "arguments": args})

    return found


def _execute_raw_tool_calls(tool_calls: list[dict], tools: list) -> tuple[list, list]:
    """
    Execute extracted tool calls directly.
    Returns (tool_messages, output_strings).
    """
    tool_messages = []
    outputs = []

    for i, tc in enumerate(tool_calls):
        tool_name = tc["name"]
        args = tc["arguments"]
        tool_obj = next((t for t in tools if t.name == tool_name), None)

        if tool_obj:
            try:
                result = tool_obj.invoke(args)
                msg = f"{tool_name}: {str(result)[:500]}"
            except Exception as e:
                msg = f"{tool_name} error: {e}"
        else:
            msg = f"Tool not found: {tool_name}"

        outputs.append(msg)
        tool_messages.append(ToolMessage(
            content=msg,
            name=tool_name,
            tool_call_id=f"raw_{i}",
        ))

    return tool_messages, outputs

# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    input: str
    history: str
    plan: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tool_outputs: Annotated[list[str], operator.add]
    status: str
    iterations: int

def create_agent():
    tools          = get_all_tools()
    llm            = get_coder_llm()
    llm_with_tools = llm.bind_tools(tools)
    refiner_llm    = get_refiner_llm()

    # ── Direct Q&A (no tools) ─────────────────────────────────────────────────
    def direct_node(state: AgentState):
        log("direct_response", {"input": state["input"][:100]})
        response = refiner_llm.invoke([
            SystemMessage(content=_system_prompt()),
            HumanMessage(content=state["input"]),
        ])
        return {"messages": [AIMessage(content=response.content)], "status": "complete"}

    # ── Plan ──────────────────────────────────────────────────────────────────
    def plan_node(state: AgentState):
        working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
        prompt = (
            f"Project directory: {working_dir}\n"
            f"Task: {state['input']}\n"
            f"Write a concise step-by-step plan (max 4 steps) using available tools."
        )
        response = refiner_llm.invoke([HumanMessage(content=prompt)])
        log("plan_created", {"plan": response.content[:200]})
        return {
            "plan": response.content,
            "iterations": 1,
            "messages": [SystemMessage(content=f"Plan:\n{response.content}")]
        }

    # ── Execute ───────────────────────────────────────────────────────────────
    def execute_node(state: AgentState):
        sys_msg  = SystemMessage(content=_system_prompt())
        recent   = list(state.get("messages", []))[-6:]
        user_msg = HumanMessage(content=(
            f"Task: {state['input']}\n"
            f"Project dir: {os.environ.get('CODI_WORKING_DIR', os.getcwd())}\n"
            f"Execute using tools NOW. Do not explain."
        ))
        messages = [sys_msg] + recent + [user_msg]

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            err = str(e)
            if "tool_use_failed" in err or "400" in err:
                log("execute_fallback", {"error": err[:200]})
                plain = refiner_llm.invoke([
                    SystemMessage(content=_system_prompt()),
                    HumanMessage(content=state["input"]),
                ])
                return {
                    "messages": [AIMessage(content=plain.content)],
                    "status": "complete",
                }
            raise

        # ── Check for structured tool calls ───────────────────────────────────
        has_structured_calls = (
            hasattr(response, 'tool_calls') and bool(response.tool_calls)
        )

        # ── Check for raw JSON tool calls in text content ─────────────────────
        content = response.content if hasattr(response, 'content') else ""
        raw_tool_calls = _extract_tool_calls_from_text(content) if not has_structured_calls else []
        print(f"DEBUG raw_tool_calls: {raw_tool_calls}")

        if raw_tool_calls:
            log("execute_node", {
                "iteration": state["iterations"],
                "has_tool_calls": True,
                "source": "text_parser",
                "count": len(raw_tool_calls),
            })
            # Execute them immediately and return results as tool messages
            tool_messages, outputs = _execute_raw_tool_calls(raw_tool_calls, tools)
            return {
                "messages": [response] + tool_messages,
                "tool_outputs": outputs,
                "iterations": state["iterations"] + 1,
            }

        log("execute_node", {
            "iteration": state["iterations"],
            "has_tool_calls": has_structured_calls,
            "source": "structured",
        })
        return {"messages": [response]}

    # ── Tool execution ────────────────────────────────────────────────────────
    def tool_node(state: AgentState):
        last_message  = state["messages"][-1]
        tool_messages = []
        outputs       = []

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_obj = next((t for t in tools if t.name == tool_call["name"]), None)
                if tool_obj:
                    try:
                        result = tool_obj.invoke(tool_call["args"])
                        msg    = f"{tool_obj.name}: {str(result)[:500]}"
                    except Exception as e:
                        msg = f"{tool_obj.name} error: {e}"
                else:
                    msg = f"Tool not found: {tool_call['name']}"

                outputs.append(msg)
                tool_messages.append(ToolMessage(
                    content=msg,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                ))

        return {
            "messages": tool_messages,
            "tool_outputs": outputs,
            "iterations": state["iterations"] + 1,
        }

    # ── Verify ────────────────────────────────────────────────────────────────
    def verify_node(state: AgentState):
        if state.get("status") == "complete":
            return {"status": "complete"}

        if not state.get("tool_outputs"):
            # Only loop back once if no tools ran — avoid infinite spin
            if state["iterations"] >= 3:
                return {"status": "complete"}
            return {
                "status": "incomplete",
                "iterations": state["iterations"] + 1,
                "messages": [SystemMessage(content=(
                    "You have not called any tools yet. "
                    "You MUST use tools to complete this task. Call them now."
                ))]
            }

        recent_outputs = "\n".join(state['tool_outputs'][-3:])
        prompt = (
            f"Task: {state['input']}\n"
            f"Tool outputs so far:\n{recent_outputs}\n"
            f"Is the task fully complete? Reply YES or NO only."
        )
        response    = refiner_llm.invoke([HumanMessage(content=prompt)])
        is_complete = response.content.strip().upper().startswith("YES")
        log("verify_node", {"iteration": state["iterations"], "is_complete": is_complete})

        if is_complete:
            return {"status": "complete"}
        return {
            "status": "incomplete",
            "iterations": state["iterations"] + 1,
            "messages": [SystemMessage(content="Not done yet. Continue using tools.")]
        }

    # ── Synthesize ────────────────────────────────────────────────────────────
    def synthesize_node(state: AgentState):
        if not state.get("tool_outputs"):
            msgs = state.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and m.content:
                    return {"messages": [m]}
            return {"messages": [AIMessage(content=(
                "Task could not be completed — no tools were executed. "
                "Try rephrasing or check /mcp to see if required servers are running."
            ))]}

        outputs_summary = "\n".join(state['tool_outputs'][-5:])
        response = refiner_llm.invoke([
            SystemMessage(content="Summarize what was completed in 2-3 sentences. Be direct and specific."),
            HumanMessage(content=f"Task: {state['input']}\nResults:\n{outputs_summary}")
        ])
        log("synthesize_node", {"output": response.content[:200]})
        return {"messages": [AIMessage(content=response.content)]}

    # ── Routing ───────────────────────────────────────────────────────────────
    def route_input(state: AgentState):
        return "direct" if is_simple_input(state["input"]) else "plan"

    def should_use_tools(state: AgentState):
        if state.get("status") == "complete":
            return "verify"
        last = state["messages"][-1]

        # Structured tool calls (proper LangChain format)
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return "tool"

        # If last message is a ToolMessage, raw calls were already executed
        # in execute_node — go straight to verify
        if isinstance(last, ToolMessage):
            return "verify"

        return "verify"

    def should_loop(state: AgentState):
        if state["status"] == "complete" or state["iterations"] >= 8:
            return "synthesize"
        return "execute"

    # ── Graph ─────────────────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("direct",     direct_node)
    workflow.add_node("plan",       plan_node)
    workflow.add_node("execute",    execute_node)
    workflow.add_node("tool",       tool_node)
    workflow.add_node("verify",     verify_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.set_conditional_entry_point(route_input, {"direct": "direct", "plan": "plan"})
    workflow.add_edge("direct", END)
    workflow.add_edge("plan",   "execute")
    workflow.add_conditional_edges("execute", should_use_tools, {"tool": "tool", "verify": "verify"})
    workflow.add_edge("tool", "verify")
    workflow.add_conditional_edges("verify", should_loop, {"synthesize": "synthesize", "execute": "execute"})
    workflow.add_edge("synthesize", END)

    app = workflow.compile()

    class CodiGraphAgent:
        def __init__(self, graph):
            self.graph = graph

        def invoke(self, inputs: dict) -> dict:
            state = {
                "input":        inputs["input"],
                "history":      inputs.get("history", ""),
                "plan":         "",
                "messages":     [],
                "tool_outputs": [],
                "status":       "start",
                "iterations":   0,
            }
            try:
                result       = self.graph.invoke(state, {"recursion_limit": 60})
                msgs         = result.get("messages", [])
                tool_outputs = result.get("tool_outputs", [])
                if msgs and isinstance(msgs[-1], AIMessage):
                    return {"output": msgs[-1].content, "tool_outputs": tool_outputs}
                return {"output": "Completed but no output generated.", "tool_outputs": tool_outputs}
            except Exception as e:
                log("agent_error", {"error": str(e)})
                return {"output": f"Agent error: {e}", "tool_outputs": []}

    log("agent_created", {"tools": len(tools)})
    return CodiGraphAgent(app)