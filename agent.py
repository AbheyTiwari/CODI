from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from tools import get_all_tools
from llm_factory import get_coder_llm, get_refiner_llm
from logger import log

# Words that mean "just answer me" — no file ops needed
SIMPLE_TRIGGERS = (
    "hello", "hi ", "hey ", "what is", "what are", "who is", "explain",
    "how do", "how does", "tell me", "what's", "whats", "thanks", "thank you",
    "yes", "no", "ok", "okay", "sure", "help", "why ", "when ", "where "
)

# Words that mean "do something" — must go through full agentic loop
ACTION_TRIGGERS = (
    "create", "write", "make", "build", "fix", "edit", "update", "delete",
    "run", "execute", "generate", "refactor", "implement", "add", "code",
    "put", "save", "html", "css", "script", "file", "folder", "index",
    "function", "class", "api", "page", "deploy", "install", "setup",
    "rename", "move", "copy", "read", "open", "parse", "fetch", "download",
    "list", "search", "find", "show", "get", "check", "access", "browse",
    "navigate", "click", "screenshot", "scrape", "query", "lookup", "pull",
    "push", "commit", "clone", "diff", "status", "remember", "store"
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

class AgentState(TypedDict):
    input: str
    history: str
    plan: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tool_outputs: Annotated[list[str], operator.add]
    status: str
    iterations: int

def create_agent():
    tools = get_all_tools()
    llm = get_coder_llm()
    llm_with_tools = llm.bind_tools(tools)
    refiner_llm = get_refiner_llm()

    # ── Short-circuit — single LLM call for simple Q&A ─────────────
    def direct_node(state: AgentState):
        log("direct_response", {"input": state["input"][:100]})
        response = refiner_llm.invoke([
            SystemMessage(content="You are Codi, a helpful coding assistant. Answer concisely."),
            HumanMessage(content=state["input"])
        ])
        return {"messages": [AIMessage(content=response.content)], "status": "complete"}

    # ── Full agentic loop ───────────────────────────────────────────
    def plan_node(state: AgentState):
        prompt = (
            f"Task: {state['input']}\n"
            f"Write a concise step-by-step plan (max 4 steps) using tools to complete this."
        )
        response = refiner_llm.invoke([HumanMessage(content=prompt)])
        log("plan_created", {"plan": response.content[:200]})
        return {
            "plan": response.content,
            "iterations": 1,
            "messages": [SystemMessage(content=f"Plan:\n{response.content}")]
        }

    def execute_node(state: AgentState):
        sys_msg = SystemMessage(content=(
            "You are an executor. Call tools immediately to complete the task. "
            "Do NOT explain — just call tools now."
        ))
        recent_messages = list(state.get("messages", []))[-6:]
        user_msg = HumanMessage(content=f"Task: {state['input']}\nExecute using tools NOW.")
        messages = [sys_msg] + recent_messages + [user_msg]

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            err = str(e)
            # Groq returns 400 when tool call JSON is malformed (tool_use_failed).
            # Fall back to a plain call so the graph doesn't crash.
            if "tool_use_failed" in err or "400" in err:
                log("execute_node_tool_fallback", {"error": err[:200]})
                plain_response = refiner_llm.invoke([
                    SystemMessage(content="You are Codi, a helpful coding assistant. Answer concisely."),
                    HumanMessage(content=state["input"])
                ])
                return {
                    "messages": [AIMessage(content=plain_response.content)],
                    "status": "complete",
                }
            raise

        has_calls = hasattr(response, 'tool_calls') and bool(response.tool_calls)
        log("execute_node", {"iteration": state["iterations"], "has_tool_calls": has_calls})
        return {"messages": [response]}

    def tool_node(state: AgentState):
        last_message = state["messages"][-1]
        tool_messages = []
        outputs = []
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_obj = next((t for t in tools if t.name == tool_call["name"]), None)
                if tool_obj:
                    try:
                        result = tool_obj.invoke(tool_call["args"])
                        msg = f"{tool_obj.name}: {str(result)[:500]}"
                    except Exception as e:
                        msg = f"{tool_obj.name} error: {e}"
                else:
                    msg = f"Tool not found: {tool_call['name']}"
                outputs.append(msg)
                tool_messages.append(ToolMessage(
                    content=msg,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
        # increment iterations so the cap accounts for every tool round
        return {
            "messages": tool_messages,
            "tool_outputs": outputs,
            "iterations": state["iterations"] + 1,
        }

    def verify_node(state: AgentState):
        # Honour complete status set by execute_node fallback
        if state.get("status") == "complete":
            return {"status": "complete"}

        if not state.get("tool_outputs"):
            log("verify_node", {"iteration": state["iterations"], "response": "NO_OUTPUTS"})
            return {
                "status": "incomplete",
                "iterations": state["iterations"] + 1,
                "messages": [SystemMessage(content=(
                    "CRITICAL: You have not called any tools yet. "
                    "You MUST call write_file to create files. Execute now, do not explain."
                ))]
            }
        recent_outputs = "\n".join(state['tool_outputs'][-3:])
        prompt = (
            f"Task: {state['input']}\n"
            f"Tool outputs: {recent_outputs}\n"
            f"Done? Reply YES or NO only."
        )
        response = refiner_llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip().upper()
        is_complete = content.startswith("YES")
        log("verify_node", {"iteration": state["iterations"], "is_complete": is_complete})

        if is_complete:
            return {"status": "complete"}
        return {
            "status": "incomplete",
            "iterations": state["iterations"] + 1,
            "messages": [SystemMessage(content="Not done yet. Continue with tools.")]
        }

    def synthesize_node(state: AgentState):
        if not state.get("tool_outputs"):
            log("synthesize_node", {"output": "NO_TOOLS_CALLED"})
            # Surface any direct AI response from the fallback path
            msgs = state.get("messages", [])
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and m.content:
                    return {"messages": [m]}
            return {"messages": [AIMessage(content=(
                "Task could not be completed — no tools were executed. "
                "Try rephrasing with an explicit file path."
            ))]}
        outputs_summary = "\n".join(state['tool_outputs'][-5:])
        response = refiner_llm.invoke([
            SystemMessage(content="Summarize what was completed in 2-3 sentences. Be direct."),
            HumanMessage(content=f"Task: {state['input']}\nResults:\n{outputs_summary}")
        ])
        log("synthesize_node", {"output": response.content[:200]})
        return {"messages": [AIMessage(content=response.content)]}

    # ── Routing ─────────────────────────────────────────────────────
    def route_input(state: AgentState):
        if is_simple_input(state["input"]):
            return "direct"
        return "plan"

    def should_use_tools(state: AgentState):
        # Fallback path: execute_node set status=complete, skip tool execution
        if state.get("status") == "complete":
            return "verify"
        last = state["messages"][-1]
        if hasattr(last, 'tool_calls') and last.tool_calls:
            return "tool"
        return "verify"

    def should_loop(state: AgentState):
        if state["status"] == "complete" or state["iterations"] >= 8:
            return "synthesize"
        return "execute"

    # ── Graph ────────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)
    workflow.add_node("direct", direct_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.set_conditional_entry_point(route_input, {"direct": "direct", "plan": "plan"})
    workflow.add_edge("direct", END)
    workflow.add_edge("plan", "execute")
    workflow.add_conditional_edges("execute", should_use_tools, {"tool": "tool", "verify": "verify"})
    # tool → verify (not tool → execute) prevents infinite tool loops
    workflow.add_edge("tool", "verify")
    workflow.add_conditional_edges("verify", should_loop, {"synthesize": "synthesize", "execute": "execute"})
    workflow.add_edge("synthesize", END)

    app = workflow.compile()

    class CodiGraphAgent:
        def __init__(self, graph):
            self.graph = graph

        def invoke(self, inputs: dict) -> dict:
            state = {
                "input": inputs["input"],
                "history": inputs.get("history", ""),
                "plan": "",
                "messages": [],
                "tool_outputs": [],
                "status": "start",
                "iterations": 0
            }
            try:
                result = self.graph.invoke(state, {"recursion_limit": 60})
                msgs = result.get("messages", [])
                if msgs and isinstance(msgs[-1], AIMessage):
                    return {"output": msgs[-1].content}
                return {"output": "Completed but no output generated."}
            except Exception as e:
                log("agent_error", {"error": str(e)})
                return {"output": f"Graph execution failed: {e}"}

    log("agent_executor_created", {"engine": "langgraph_v2_optimized"})
    return CodiGraphAgent(app)