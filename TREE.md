# CODI — NEW PROJECT STRUCTURE

```
codi/                               ← repo root (where you run `pip install -e .`)
│
├── agent.py                        ← NEW  thin agent shell, explicit loop, no LangGraph
├── dispatcher.py                   ← NEW  central router — receives JSON, fans out to tools
│
├── core/                           ← NEW  four focused LLM modules
│   ├── __init__.py
│   ├── improver.py                 ← NEW  orchestrator LLM (fast/cheap): plans, drives loop, summarizes
│   ├── planner.py                  ← NEW  routes simple vs complex, refines input (absorbs refiner.py)
│   ├── executor.py                 ← NEW  coder LLM: turns step → JSON action bundle
│   └── validator.py                ← NEW  explicit pass/fail checks (deterministic + LLM fallback)
│
├── tools/                          ← NEW  all tools live here, dispatcher calls them
│   ├── __init__.py
│   ├── registry.py                 ← NEW  single source of truth — local + MCP unified
│   ├── local/
│   │   ├── __init__.py
│   │   ├── file_tools.py           ← NEW  read_file, write_file, list_files, create_directory
│   │   ├── shell_tools.py          ← NEW  run_command
│   │   └── search_tools.py         ← NEW  search_codebase (wraps indexer/chroma)
│   └── mcp/
│       ├── __init__.py
│       └── mcp_tools.py            ← NEW  wraps MCP tool objects → plain callables for registry
│
├── state/                          ← NEW  centralized run state (replaces scattered AgentState)
│   ├── __init__.py
│   └── temp_db.py                  ← NEW  RunState dataclass: plan, tool results, iteration, validation
│
│── main.py                         ← KEPT (1 line changed: refiner → core.planner)
├── cli.py                          ← KEPT unchanged
├── config.py                       ← KEPT unchanged
├── config_loader.py                ← KEPT unchanged
├── context_trimmer.py              ← KEPT unchanged
├── indexer.py                      ← KEPT unchanged
├── llm_factory.py                  ← KEPT unchanged
├── logger.py                       ← KEPT unchanged
├── log_viewer.py                   ← KEPT unchanged
├── mcp_manager.py                  ← KEPT unchanged
├── mcp_servers.json                ← KEPT unchanged
├── memory.py                       ← KEPT unchanged
├── pyproject.toml                  ← UPDATED (removed old modules, added new packages)
├── setup.cfg                       ← KEPT unchanged
│
└── DELETED (do not keep):
    ├── refiner.py                  ← absorbed into core/planner.py
    ├── mcp_client.py               ← duplicate of mcp_manager.py, removed
    └── tools.py                    ← replaced by tools/registry.py + tools/local/ + tools/mcp/
```

---

## DATA FLOW (how a request moves through the new system)

```
User types input
      │
      ▼
main.py                     ← pure UI shell, no logic
      │
      ▼
agent.py → CodiAgent.invoke()
      │
      ├─ core/planner.py    ← is this simple Q&A or needs execution?
      │       │
      │       └─ simple → direct_answer() → back to user
      │
      ├─ core/improver.py   ← read_context() → list_files + search_codebase
      │
      ├─ core/improver.py   ← create_plan()  → JSON {plan, steps[]}
      │
      └─ LOOP:
            │
            ├─ core/improver.py  → next_step()   → JSON {step, done}
            │
            ├─ core/executor.py  → execute_step()
            │       │
            │       ├─ Coder LLM → JSON action bundle
            │       │       {action: "tool_call", tools: [{name, args}, ...]}
            │       │
            │       └─ dispatcher.py → parallel tool execution
            │               │
            │               ├─ tools/registry.py → local tool handler
            │               └─ tools/registry.py → mcp tool handler
            │
            ├─ state/temp_db.py  ← results stored here
            │
            ├─ core/validator.py → validate()  → pass / fail
            │
            └─ fail → core/improver.py → improve() → correction → retry
                 done → core/improver.py → summarize() → plain text output
                              │
                              ▼
                         main.py renders to terminal
```

---

## KEY CONTRACTS

### Every tool in registry:
```python
fn(args: dict) -> str
```
No LangChain. No decorators. Plain function.

### Coder LLM output (executor → dispatcher):
```json
{
  "action": "tool_call",
  "tools": [
    {"name": "write_file", "args": {"path": "app.py", "content": "..."}},
    {"name": "run_command", "args": {"command": "python app.py"}}
  ]
}
```

### Dispatcher response (back to executor → state):
```json
{
  "status": "success",
  "results": [
    {"tool": "write_file",  "status": "ok",    "output": "Written 420 chars to app.py"},
    {"tool": "run_command", "status": "ok",    "output": "Hello world"}
  ]
}
```

### Validator response (LLM path):
```json
{"passed": true,  "notes": ""}
{"passed": false, "notes": "file app.py was not written"}
```

---

## WHAT CHANGED VS OLD SYSTEM

| Old                        | New                                    |
|----------------------------|----------------------------------------|
| agent.py (500 line monolith)| agent.py (80 lines) + core/ modules   |
| LangGraph StateGraph        | plain while loop in agent.py           |
| tools.py + bind_tools(llm)  | tools/registry.py — dispatcher calls   |
| _extract_tool_calls_from_text() regex | deleted — JSON only          |
| refiner.py                  | core/planner.py (refine_input method)  |
| mcp_client.py               | deleted — mcp_manager.py is enough     |
| memory.py AgentState        | state/temp_db.py RunState              |
| validation buried in loop   | core/validator.py explicit module      |
| improvement random          | core/improver.py dedicated step        |
| sequential tool execution   | dispatcher runs tools in parallel      |
| scattered state             | RunState owns everything               |
