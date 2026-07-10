"""Iterative, evidence-gated discovery before a plan may be created."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage
from dispatcher import Dispatcher
from llm_factory import get_refiner_llm
from logger import log
from state.knowledge import KnowledgeBase
from tools.registry import ToolRegistry, registry

_PROMPT = """You are CODI's context-discovery controller. Do not plan or write code.
Choose exactly one action: inspect_project, inspect_file, search_codebase, read_file, or done.
Prefer inspect_project then inspect_file. read_file is allowed only when structure is insufficient.
Use only paths already shown in PROJECT KNOWLEDGE or search results. Never repeat an action
already taken and never request a file that is reported missing.
Return JSON only: {\"action\":\"inspect_file\",\"path\":\"...\",\"reason\":\"short evidence need\"}.
For done include {\"action\":\"done\",\"confidence\":0.95,\"summary\":\"...\"}."""

@dataclass
class ContextState:
    confidence: float = 0.0
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)
    actions: list[dict] = field(default_factory=list)
    context: str = ""
    complete: bool = False
    source_evidence: list[str] = field(default_factory=list)

class ContextBuilder:
    def __init__(self, tool_registry: ToolRegistry | None = None):
        self.llm = get_refiner_llm()
        self.dispatcher = Dispatcher(tool_registry or registry)

    @staticmethod
    def _parse(raw: str) -> dict | None:
        try:
            start, end = raw.find("{"), raw.rfind("}")
            return json.loads(raw[start:end + 1]) if start >= 0 and end > start else None
        except (ValueError, TypeError):
            return None

    def _run(self, state: ContextState, name: str, args: dict) -> None:
        result = self.dispatcher.dispatch({"action": "tool_call", "tools": [{"name": name, "args": args}]}, knowledge=state.knowledge)
        state.actions.append({"tool": name, "args": args, "status": result.get("status")})
        if result.get("status") == "error":
            state.knowledge.add_unknown(f"{name} failed for {args}: inspect a different path or proceed with available evidence.")
        if name == "read_file":
            for item in result.get("results", []):
                if item.get("status") == "ok":
                    state.source_evidence.append(f"SOURCE {args.get('path', '')}:\n{item.get('output', '')}")

    def _decide(self, state: ContextState, mission, conversation_history: str) -> dict | None:
        evidence = "\n\n".join(state.source_evidence)[-70000:]
        prompt = f"Mission: {mission.goal}\nLikely files: {mission.files_needed}\nLikely symbols: {mission.symbols_needed}\n{state.knowledge.summary_for_prompt()}\nConversation history:\n{conversation_history[-50000:]}\nSource/history evidence:\n{evidence}\nActions already taken: {state.actions}"
        try:
            response = self.llm.invoke([SystemMessage(content=_PROMPT), HumanMessage(content=prompt)])
            return self._parse(response.content)
        except Exception as exc:
            log("context_llm_error", {"error": str(exc)})
            return None

    def build(self, mission, history: str = "", knowledge: KnowledgeBase | None = None,
              max_iterations: int = 8, full_codebase: bool = False) -> ContextState:
        state = ContextState(knowledge=knowledge or KnowledgeBase())
        # Project shape is deterministic and required for every code-changing task.
        if not state.knowledge.project:
            self._run(state, "inspect_project", {"path": "."})
        if full_codebase:
            # A full-context session reads every eligible project file through
            # the normal tool boundary.  The collected prompt evidence remains
            # bounded, but every file is inspected/read before planning.
            paths = state.knowledge.project.get("files", [])
            eligible = [p for p in paths if str(p).lower().endswith(
                (".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".json", ".md", ".yml", ".yaml", ".toml", ".sh")
            )]
            limit = 150
            for path in eligible[:limit]:
                self._run(state, "inspect_file", {"path": path})
                self._run(state, "read_file", {"path": path})
            if len(eligible) > limit:
                state.knowledge.add_unknown(
                    f"Full-codebase read capped at {limit} of {len(eligible)} eligible files."
                )
        # Give the planner actual source, not only AST metadata. This is a
        # high-value deterministic step for a 7B model and avoids repeated
        # discovery turns for files that were already named by the task.
        known_paths = {str(path).replace("\\", "/") for path in state.knowledge.project.get("files", [])}
        for path in mission.files_needed:
            if isinstance(path, str) and path:
                normalized = path.replace("\\", "/")
                if normalized in known_paths:
                    self._run(state, "inspect_file", {"path": path})
                    self._run(state, "read_file", {"path": path})
                else:
                    state.knowledge.add_unknown(f"Requested file is not present: {path}")
        for _ in range(max_iterations):
            action = self._decide(state, mission, history)
            if not action:
                # Bad JSON from a small model must not block an otherwise
                # evidence-backed task.
                state.knowledge.add_unknown("Context controller returned invalid JSON; using collected evidence.")
                break
            name = str(action.get("action", "done")).lower()
            if name == "done":
                state.confidence = float(action.get("confidence", 0.0))
                state.knowledge.summaries.append(str(action.get("summary", "")))
                break
            if name not in {"inspect_file", "search_codebase", "read_file"}:
                state.knowledge.add_unknown(f"Unsupported context action: {name}")
                continue
            args = {key: action[key] for key in ("path", "query") if key in action}
            if not args:
                state.knowledge.add_unknown(f"Missing arguments for {name}")
                continue
            if any(existing["tool"] == name and existing["args"] == args for existing in state.actions):
                state.knowledge.add_unknown(f"Context controller repeated {name} {args}; using collected evidence.")
                break
            self._run(state, name, args)
        # Before asking the user, consume the persistent CODI conversation log.
        # It is evidence, not private hidden reasoning, and is supplied only as
        # far as the model context can safely hold.
        if not state.complete and not any(a["tool"] == "read_agent_history" for a in state.actions):
            self._run(state, "read_agent_history", {})
            action = self._decide(state, mission, history)
            if action and str(action.get("action", "")).lower() == "done":
                state.confidence = float(action.get("confidence", 0.0))
                state.knowledge.summaries.append(str(action.get("summary", "")))
        # Planning is evidence-gated, not model-confidence-gated. A 7B model
        # can emit weak confidence or malformed JSON despite the project and
        # relevant files already being inspected.
        has_project = bool(state.knowledge.project)
        # An inspected project is enough to begin a plan. A task may
        # legitimately create every file it names, so requiring an existing
        # inspected file wrongly blocks new projects and empty directories.
        state.complete = has_project
        if state.complete and state.confidence < 0.90:
            state.confidence = 0.90
            state.knowledge.summaries.append("Planning permitted from verified project and file evidence.")
        state.context = state.knowledge.summary_for_prompt() + "\n\n" + "\n\n".join(state.source_evidence)[-70000:]
        if not state.complete:
            state.knowledge.add_unknown("Context confidence/evidence threshold was not met; planning is blocked.")
        log("context_complete", {"confidence": state.confidence, "complete": state.complete, "actions": state.actions})
        return state
