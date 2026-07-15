"""Iterative, evidence-gated discovery before a plan may be created."""
from __future__ import annotations

import difflib
import json
import os
import re
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

# Penalty applied to confidence per unresolved "unknown" the discovery
# process recorded (a failed tool call, a repeated/blocked action, a
# genuinely missing requested file, etc). Previously confidence was floored
# UP to 0.90 whenever state.complete was true, regardless of how many
# unknowns remained unresolved — so a run that had e.g. a failed
# inspect_file call still reported confidence=1.00 if the LLM's own "done"
# call happened to say so. This constant makes each unresolved unknown cost
# real, visible confidence instead of being silently absorbed by the floor.
_CONFIDENCE_PENALTY_PER_UNKNOWN = 0.12
_MIN_FLOOR_CONFIDENCE = 0.55

@dataclass
class ContextState:
    confidence: float = 0.0
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)
    actions: list[dict] = field(default_factory=list)
    context: str = ""
    complete: bool = False
    source_evidence: list[str] = field(default_factory=list)
    # Files the mission explicitly named that could not be resolved against
    # any known project path, even fuzzily. This is distinct from a generic
    # "unknown" — it means planning must not proceed on a silent guess.
    missing_requested_files: list[str] = field(default_factory=list)
    needs_user_clarification: bool = False
    clarification_question: str = ""

# The Mission Analyzer's files_needed list is meant to hold actual project
# file paths, but a small model sometimes fills it with capability/dependency
# descriptions instead — e.g. "Library or service for PDF extraction" for a
# brand-new feature that names no existing file at all. Those are not file
# paths and must never be run through the missing-file hard-stop below; doing
# so previously blocked a plain "add a new feature" request by demanding the
# user "confirm the exact file path" for something that was never a file.


def _looks_like_file_path(candidate: str) -> bool:
    """
    True only for strings that plausibly reference an actual file path,
    not prose capability/dependency descriptions like "Library or service
    for PDF extraction". A real path (even a Windows path with spaces in a
    folder name, e.g. "...\\cyientist AI\\app\\api\\ingest.py") always ends
    in a short recognizable extension and has few consecutive prose words;
    a description sentence has many.
    """
    candidate = candidate.strip().strip("'\"`")
    if not candidate or len(candidate) > 260:
        return False
    # Must end in a plausible file extension — this is the strongest signal
    # and is what every genuine path has, spaces in directories or not.
    ext_match = re.search(r"\.([A-Za-z0-9]{1,6})$", candidate)
    if not ext_match:
        return False
    known_exts = {
        "py", "js", "ts", "jsx", "tsx", "html", "css", "json", "md", "txt",
        "svg", "sh", "yml", "yaml", "toml", "cfg", "ini", "env", "log",
        "sql", "db", "csv", "xlsx", "docx", "pdf", "png", "jpg", "jpeg",
        "gitignore",
    }
    if ext_match.group(1).lower() not in known_exts:
        return False
    # Reject prose: a genuine path has at most a couple of "words" separated
    # by spaces (e.g. a folder named "cyientist AI"), whereas a capability
    # description reads as a full sentence/phrase with several words and
    # often prepositions like "for"/"to"/"and".
    word_count = len(candidate.split())
    if word_count > 4:
        return False
    if re.search(r"\b(for|service|library|elements?|feature)\b", candidate, re.IGNORECASE):
        return False
    return True


def _resolve_close_path(requested: str, known_paths: set[str]) -> str | None:
    """
    If `requested` doesn't exact-match anything in known_paths, look for the
    closest-matching known path by basename similarity. Mirrors the
    executor's edit-path fuzzy resolution, so a near-miss filename (case,
    typo, different directory than the model assumed) doesn't get written
    off as "not present" when the real file is one character away.
    """
    if requested in known_paths:
        return requested
    target_name = os.path.basename(requested)
    by_basename = {os.path.basename(p): p for p in known_paths}
    matches = difflib.get_close_matches(target_name, list(by_basename.keys()), n=1, cutoff=0.75)
    if matches:
        return by_basename[matches[0]]
    return None


def _penalized_confidence(raw_confidence: float, unknown_count: int) -> float:
    """
    Apply a deterministic penalty per unresolved unknown so a discovery run
    with failed/blocked tool calls can never report full confidence just
    because the discovery-controller LLM's own "done" call said so, or
    because the completion floor below would otherwise round it up.

    This does not replace the floor logic in build() — it clips the value
    BEFORE the floor is applied, so a genuinely evidence-backed run with
    zero unknowns still gets floored up to a workable minimum, while a run
    with several unresolved unknowns is visibly and proportionally less
    confident instead of being indistinguishable from a clean run.
    """
    penalty = min(0.9, unknown_count * _CONFIDENCE_PENALTY_PER_UNKNOWN)
    return max(0.0, min(raw_confidence, 1.0) - penalty)


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
            # Chroma remains the semantic search layer; SQLite supplies exact
            # paths, declarations, and references for surgical changes.
            self._run(state, "refresh_code_index", {"path": "."})
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
        filelike_requested = []
        for path in mission.files_needed:
            if not (isinstance(path, str) and path):
                continue
            if not _looks_like_file_path(path):
                # Not a real path — e.g. "Library or service for PDF
                # extraction", a dependency/capability note the Mission
                # Analyzer sometimes puts in files_needed for brand-new
                # features. Nothing to resolve; must never feed the
                # missing-file hard-stop below.
                continue
            filelike_requested.append(path)
            normalized = path.replace("\\", "/")
            resolved = _resolve_close_path(normalized, known_paths)
            if resolved:
                self._run(state, "inspect_file", {"path": resolved})
                self._run(state, "read_file", {"path": resolved})
            else:
                # No exact match and no close fuzzy match — this file is
                # genuinely not identifiable in the project. Track it
                # explicitly instead of silently dropping it into
                # "unknowns" and continuing, which previously let an
                # edit task plan against a guessed substitute file.
                state.missing_requested_files.append(path)
                state.knowledge.add_unknown(f"Requested file is not present: {path}")
        # Hard stop: if the mission named specific, real-looking file path(s)
        # (the normal case for a single-file edit task) and NONE could be
        # resolved, even fuzzily, planning must not proceed on a silent
        # substitute guess. Ask the user instead of quietly continuing on
        # weak evidence. A new-feature request with no actual file paths
        # named (only capability descriptions) never reaches this branch.
        if filelike_requested and len(state.missing_requested_files) == len(filelike_requested):
            state.needs_user_clarification = True
            state.clarification_question = (
                "I couldn't find "
                + (
                    f"the file '{state.missing_requested_files[0]}'"
                    if len(state.missing_requested_files) == 1
                    else f"any of these files: {', '.join(state.missing_requested_files)}"
                )
                + " in the current project directory. Could you confirm the exact "
                "file path, or let me know if it should be created as a new file?"
            )
            state.complete = False
            log("context_missing_requested_file", {
                "missing": state.missing_requested_files,
                "mission_goal": mission.goal[:160] if getattr(mission, "goal", None) else "",
            })
            return state

        for _ in range(max_iterations):
            action = self._decide(state, mission, history)
            if not action:
                # Bad JSON from a small model must not block an otherwise
                # evidence-backed task.
                state.knowledge.add_unknown("Context controller returned invalid JSON; using collected evidence.")
                break
            name = str(action.get("action", "done")).lower()
            if name == "done":
                # Clip the LLM's self-reported confidence against the actual
                # number of unresolved unknowns BEFORE storing it. Without
                # this, a run that hit e.g. a failed inspect_file call still
                # ends up reporting confidence=1.00 purely because the
                # discovery-controller LLM said so — the unknown was tracked
                # in state.knowledge.unknowns but nothing ever read that list
                # when computing confidence.
                raw_confidence = float(action.get("confidence", 0.0))
                state.confidence = _penalized_confidence(raw_confidence, len(state.knowledge.unknowns))
                state.knowledge.summaries.append(str(action.get("summary", "")))
                log("context_confidence_penalized", {
                    "raw_confidence": raw_confidence,
                    "unknown_count": len(state.knowledge.unknowns),
                    "penalized_confidence": state.confidence,
                })
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
                raw_confidence = float(action.get("confidence", 0.0))
                state.confidence = _penalized_confidence(raw_confidence, len(state.knowledge.unknowns))
                state.knowledge.summaries.append(str(action.get("summary", "")))
                log("context_confidence_penalized", {
                    "raw_confidence": raw_confidence,
                    "unknown_count": len(state.knowledge.unknowns),
                    "penalized_confidence": state.confidence,
                    "source": "post_history",
                })
        # Planning is evidence-gated, not model-confidence-gated. A 7B model
        # can emit weak confidence or malformed JSON despite the project and
        # relevant files already being inspected.
        has_project = bool(state.knowledge.project)
        # An inspected project is enough to begin a plan. A task may
        # legitimately create every file it names, so requiring an existing
        # inspected file wrongly blocks new projects and empty directories.
        state.complete = has_project
        if state.complete and state.confidence < _MIN_FLOOR_CONFIDENCE:
            # Floor confidence so a project/file-evidence-backed run is never
            # blocked purely by a low or missing self-reported number — but
            # the floor is now BELOW the old 0.90, and is applied AFTER the
            # unknown-count penalty above, not instead of it. A run with
            # several unresolved unknowns will floor at 0.55 (visibly
            # "planning permitted, but shaky"), not silently jump to 0.90+.
            state.confidence = _MIN_FLOOR_CONFIDENCE
            state.knowledge.summaries.append(
                "Planning permitted from verified project and file evidence "
                "(confidence floored at minimum working threshold)."
            )
        # Source code goes FIRST, metadata summary AFTER. This matters
        # because create_plan() only ever shows the planner LLM the first
        # ~6000 chars of this string (a hard local-model context budget
        # constraint, not a bug to just remove). With metadata first (as
        # this was previously ordered), a project of even moderate size
        # fills that 6000-char window entirely with import/symbol lists
        # before a single line of actual source code appears — meaning the
        # planner reasons about *descriptions* of files, never their real
        # content, which is exactly the "doesn't read the actual code"
        # problem. Putting source first means the highest-value evidence
        # (the real code) survives truncation instead of the cheapest,
        # most-replaceable evidence (AST metadata) crowding it out.
        source_block = "\n\n".join(state.source_evidence)[-70000:]
        metadata_block = state.knowledge.summary_for_prompt()
        state.context = source_block + "\n\n" + metadata_block if source_block else metadata_block
        if not state.complete:
            state.knowledge.add_unknown("Context confidence/evidence threshold was not met; planning is blocked.")
        log("context_complete", {"confidence": state.confidence, "complete": state.complete, "actions": state.actions, "unknown_count": len(state.knowledge.unknowns)})
        return state