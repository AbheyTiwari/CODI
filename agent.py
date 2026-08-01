# agent.py
# ─────────────────────────────────────────────────────────────────────────────
# The Codi agent. Clean, explicit, no LangGraph magic.
#
# Routing:
#   "qa"    → direct LLM answer, no tools
#   "read"  → read/search only, answer from that content, never writes
#   "edit"  → one direct executor call for a targeted single-file change;
#             falls back to the full build pipeline if that doesn't land
#   "build" → full Improver → Executor → Validator loop
#
# Plan confirmation gate (build path only):
#   - On a FRESH "build" task, plan_confirmed is forced to False. After
#     create_plan() runs, if plan_confirmed is still False, we write
#     plan.md and return immediately WITHOUT executing anything, along
#     with the RunState object itself so the caller (main.py) can hold
#     onto it.
#   - To actually execute, the caller must call invoke() again passing that
#     same RunState back in via resume_state=. That call skips planning
#     entirely and goes straight into the execution loop.
#
#   FIX: this gate previously lived ONLY inline in the "fresh task" branch
#   of _run(). The "awaiting_context" resume branch (reached after the user
#   answers a context-clarification question) called create_plan() a SECOND
#   time but never routed through the gate — it fell straight into the
#   execution loop with state.status = "running", so plan.md was silently
#   never written and the user's "type y" confirmation had nothing real
#   behind it. Both branches now call the same _confirm_plan_gate() helper,
#   so there is exactly one place that decides "does this plan need to be
#   shown and confirmed before running" and it can't drift out of sync
#   between the two entry paths again.
#
# IMPORTANT: _run() MUST return a string in every code path. The execution
# loop below sets state.status = "complete" via break but does not itself
# produce user-facing output — the final summarize() call at the bottom of
# _run() is what turns "the loop finished" into an actual answer. Without
# it, invoke() returns output=None and the caller prints "No output
# returned." even after files were written successfully.
#
# LLM-BACKEND-DOWN HANDLING:
#   Improver.next_step() / .improve() now return an "llm_error" key when the
#   underlying LLM call itself failed (connection error, timeout, backend
#   unreachable) rather than the model genuinely returning nothing. Treating
#   that as "step empty -> task complete" (the previous behavior) silently
#   reported success on runs where the backend never responded at all. The
#   loop below now retries up to _MAX_LLM_BACKEND_ERRORS times, then fails
#   the run explicitly with a message naming the actual problem.
# ─────────────────────────────────────────────────────────────────────────────

import re
import traceback
import os
from langchain_core.messages import HumanMessage, SystemMessage
from core.mission_analyzer import MissionAnalyzer
from core.context_builder import ContextBuilder
from core.execution_reflector import ExecutionReflector
from core.executor  import Executor
from core.improver  import Improver, classify_plan_risk
from core.planner   import Planner, EDIT_VERBS, BUILD_VERBS
from core.quick_actions import try_fast_file_task
from core.validator import Validator
from dispatcher      import Dispatcher, wrap_prompt_data
from logger          import log
from state.temp_db   import RunState
from tools.registry  import ToolRegistry, registry as _global_registry
from status_stream   import emit_status

_FILE_MENTION_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,5}\b")

_IMPLEMENTATION_VERBS = (
    "add", "change", "create", "edit", "fix", "implement", "insert",
    "modify", "remove", "replace", "style", "update", "write",
)

# How many consecutive LLM-backend failures (connection errors, timeouts)
# the execution loop tolerates before giving up and reporting the real
# problem, instead of silently looping to max_iterations or — worse —
# treating the resulting empty step as "task complete".
_MAX_LLM_BACKEND_ERRORS = 3
_MISSING_FILE_UNKNOWN_RE = re.compile(r"^Requested file is not present:\s*(.+)$", re.IGNORECASE)
_CREATE_CONFIRMATION_RE = re.compile(
    r"^(?:y|yes|yeah|yep|ok|okay|sure|go ahead|cr(?:e?a)te(?:\s+(?:it|them|the files))?|build(?:\s+(?:it|them|the files))?)\b",
    re.IGNORECASE,
)


def _confirmed_missing_file_creation(response: str, unknowns: list[str]) -> list[str]:
    """Return missing targets explicitly approved for creation by the user."""
    if not _CREATE_CONFIRMATION_RE.search((response or "").strip()):
        return []
    return [
        match.group(1).strip()
        for item in unknowns
        if (match := _MISSING_FILE_UNKNOWN_RE.match(str(item)))
    ]

# ── QA-path misclassification backstop ──────────────────────────────────────
# classify_intent() (core/planner.py) uses keyword/phrase matching to decide
# "qa" (plain chat, no tools) vs "edit"/"build" (real execution). Keyword
# matching is inherently incomplete — some change request will always slip
# through as "qa" eventually. When that happens, Planner.direct_answer()
# happily produces a prose answer with code fences and "Steps to Implement:
# 1. Create or update index.html..." — content that LOOKS like an
# implementation but was never executed by any tool. Nothing downstream
# previously checked this; the inert answer was returned to the user as if
# the task were done.
#
# This is a deterministic backstop, not a replacement for fixing
# classify_intent() itself: if the "qa" answer content looks like an
# unexecuted implementation (code fences plus file-extension mentions or
# instructive "do this" phrasing) AND the ORIGINAL request used edit/build
# language, we treat that as a misclassification and fall through into the
# real execution pipeline instead of returning the prose. Genuine
# explanatory questions ("explain flexbox", "show me an example navbar")
# are untouched — they don't use edit/build verbs in the request, so the
# second condition never fires and code-in-answers keeps working normally.
_CODE_FENCE_RE = re.compile(r"```")
_FILE_EXT_MENTION_RE = re.compile(
    r"\.(?:html|css|js|ts|jsx|tsx|py|json)\b", re.IGNORECASE
)
_INSTRUCTIVE_IMPLEMENTATION_PHRASE_RE = re.compile(
    r"\b(create or update|steps to implement|save (?:this|it) as|"
    r"add this to|place this in|copy this into|paste this into)\b",
    re.IGNORECASE,
)
_EDIT_BUILD_VERBS_FOR_QA_GUARD = tuple(set(EDIT_VERBS) | set(BUILD_VERBS))


def _qa_answer_is_unexecuted_change(user_input: str, answer: str) -> bool:
    """True when a 'qa'-routed answer looks like a described-but-not-applied
    code change for a request that used real edit/build language."""
    if not answer:
        return False
    has_code_fence = bool(_CODE_FENCE_RE.search(answer))
    if not has_code_fence:
        return False
    looks_like_implementation = bool(
        _FILE_EXT_MENTION_RE.search(answer) or _INSTRUCTIVE_IMPLEMENTATION_PHRASE_RE.search(answer)
    )
    if not looks_like_implementation:
        return False
    lowered_input = (user_input or "").lower()
    return any(
        re.search(rf"\b{re.escape(verb)}\b", lowered_input)
        for verb in _EDIT_BUILD_VERBS_FOR_QA_GUARD
    )


def _clean_step_text(step: str) -> str:
    """Strip filenames and path references from the step text to prevent
    false-positive verb matches (e.g., 'change.md' matching 'change', or
    'style.css' matching 'style')."""
    # Remove things like change.md, styles.css
    cleaned = re.sub(r"[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,5}\b", " ", step or "")
    # Strip punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    return cleaned


def _step_requires_mutation(step: str) -> bool:
    cleaned = _clean_step_text(step)
    return any(re.search(rf"\b{verb}\b", cleaned.lower()) for verb in _IMPLEMENTATION_VERBS)


def _step_succeeded(state: RunState, step: str = "") -> bool:
    """
    Deterministic check: did the most recent tool action succeed?
    A step counts as done if the last tool result recorded is 'ok', OR if
    the executor explicitly signalled a duplicate-write skip (the file was
    already correctly written by an earlier identical step) or a noop
    (nothing needed doing). Does NOT consult the LLM semantic validator —
    that answers a different question (is the whole task done).
    """
    if not state.tool_results:
        return False
    last = state.tool_results[-1]

    # Dispatcher noop/duplicate-write signals also count as step success —
    # they mean "nothing left to do here," not "this failed". Check this first,
    # before _step_requires_mutation exits early.
    if last.tool == "dispatcher" and last.output in ("noop", "done"):
        return True

    if _step_requires_mutation(step):
        return last.status == "ok" and last.tool in {"create_file", "write_file", "edit_file", "apply_patch"}
    
    if last.status == "ok":
        return True
    return False


def _agent_status(message: str) -> None:
    """Show high-level agent progress without exposing hidden model reasoning."""
    emit_status("agent", message)


class CodiAgent:
    def __init__(self, registry: ToolRegistry = None):
        self.registry  = registry or _global_registry
        self.planner   = Planner()
        self.mission = MissionAnalyzer()
        self.improver  = Improver(self.registry)
        self.executor  = Executor(self.registry)
        self.validator = Validator()
        self.reflector = ExecutionReflector()

    def invoke(self, inputs: dict, resume_state: RunState = None) -> dict:
        """
        Main entry point.

        inputs: {"input": str, "history": str}   — ignored if resume_state is given
        resume_state: a RunState previously returned with status
                      "awaiting_plan_confirmation", after the user confirmed it.

        returns: {"output": str, "tool_outputs": list[str], "state": RunState}

        The "state" key is always present now. Callers (main.py) must check
        state.status == "awaiting_plan_confirmation" to know whether to hold
        onto it and prompt the user for confirmation instead of treating the
        output as a finished answer.
        """
        resuming = resume_state is not None

        if resuming:
            state = resume_state
            if state.status == "awaiting_plan_confirmation":
                # A follow-up question about a plan must never be interpreted
                # as approval. The UI supplies this explicit flag only for a
                # y/yes confirmation.
                state.plan_confirmed = bool(inputs.get("confirm_plan", False))
            elif state.status == "awaiting_context":
                response = str(inputs.get("context_response", "")).strip()
                state.context_response = response
                if response:
                    state.history = f"{state.history}\nUser context response: {response}".strip()
                # A declined request means proceed with the verified evidence
                # already gathered. Any other response is treated as added
                # context and discovery gets another pass before planning.
                state.plan_confirmed = False
            elif state.status == "awaiting_plan_revision":
                feedback = str(inputs.get("plan_feedback", "")).strip()
                if feedback and feedback.lower() not in {"retry", "try again"}:
                    state.plan_validation_feedback = feedback
                state.plan_validation_attempts = 0
                state.plan_confirmed = False
        else:
            state = RunState(
                user_input=inputs.get("input", ""),
                history=inputs.get("history", ""),
            )
            state.context_scope = "full" if inputs.get("read_entire_codebase") else "targeted"
            state.force_read = bool(inputs.get("force_read"))
            # Every fresh task starts unconfirmed. Fast-path / direct-answer /
            # read / edit tasks never reach the gate check, so this is safe
            # to force here — only the "build" path consults it.
            state.plan_confirmed = False

        _agent_status(f"Received task: {state.user_input[:120]}")
        log("agent_start", {"input": state.user_input[:120], "resuming": resuming})

        try:
            output = self._run(state, resuming=resuming)
        except Exception as e:
            log("agent_crash", {"error": str(e), "traceback": traceback.format_exc()[:4000]})
            output = f"Agent error: {e}"
            state.status = "failed"

        # Defensive: if some code path still slips through without setting
        # output (e.g. a future refactor breaks the invariant again), fall
        # back to a real summary rather than silently returning None.
        if output is None:
            log("agent_output_missing", {"status": state.status, "iterations": state.iteration})
            output = self.improver.summarize(state) or "Task finished, but no summary could be generated."

        log("agent_end", {
            "output": (output or "")[:120],
            "iterations": state.iteration,
            "status": state.status,
        })

        return {
            "output":       output,
            "tool_outputs": [{"tool": r.tool, "status": r.status, "output": r.output} for r in state.tool_results[-10:]],
            "state":        state,
        }

    # ── Plan confirmation gate ────────────────────────────────────────────────
    # FIX: previously duplicated inline only in the "fresh task" branch of
    # _run(). Extracted so BOTH the fresh-plan path and the post-context-
    # discovery resume path go through the exact same logic — writing
    # plan.md, computing risk, and setting state.status =
    # "awaiting_plan_confirmation" — instead of the resume path silently
    # skipping straight to execution with no plan.md and no real
    # confirmation behind the "type y" prompt the user saw.
    #
    # Returns the confirmation message string if the gate triggers (i.e.
    # plan_confirmed is still False and there are steps to confirm), or
    # None if the caller should proceed straight into execution (e.g. the
    # plan is already confirmed, or there are no steps at all — which
    # summarize() will report honestly rather than confirming an empty plan).
    def _confirm_plan_gate(self, state: RunState, analysis) -> str | None:
        if state.plan_confirmed or not state.plan_steps:
            return None

        plan_path = os.path.join(
            os.environ.get("CODI_WORKING_DIR", os.getcwd()), "plan.md"
        )
        plan_context = state.knowledge.plan_context()
        risk_info = classify_plan_risk(state)
        mission_confidence = float(getattr(analysis or state.mission, "confidence", 0.0) or 0.0)
        evidence_confidence = float(state.context_confidence or 0.0)
        # Both independent signals must support a high-confidence label. For a
        # broad, framework-unspecified storefront request, architecture is an
        # explicit user decision rather than something CODI can truthfully
        # infer to certainty from a directory listing.
        confidence = min(mission_confidence, evidence_confidence)
        if re.search(r"\be-?commerce|online store|web store\b", state.user_input, re.IGNORECASE) and not state.requirements.framework:
            confidence = min(confidence, 0.75)

        lines = [
            f"# Plan: {state.plan}", "",
            "## Mission", analysis.goal if analysis else state.user_input, "",
            "## Understanding", state.knowledge.summary_for_prompt(), "",
            "## Architecture", "```json", str(plan_context["dependency_graph"]), "```", "",
            "## Files inspected",
        ]
        lines.extend(f"- {path}" for path in plan_context["files_inspected"])

        lines.extend(["", "## Capabilities"])
        lines.append(f"- Framework: {state.requirements.framework or 'none locked'}")
        lines.extend(f"- {m}" for m in state.requirements.must_have) or lines.append("- (none extracted)")

        lines.extend(["", "## Files to Create"])
        lines.extend(f"- {f}" for f in risk_info["files_to_create"]) if risk_info["files_to_create"] else lines.append("- (none)")

        lines.extend(["", "## Files to Modify"])
        lines.extend(f"- {f}" for f in risk_info["files_to_modify"]) if risk_info["files_to_modify"] else lines.append("- (none)")

        lines.extend(["", "## Dependencies"])
        lines.extend(f"- {mn}" for mn in state.requirements.must_not) if state.requirements.must_not else lines.append("- (none)")

        if analysis is not None:
            lines.extend(["", "## Assumptions"] + [f"- {value}" for value in analysis.assumptions])
        lines.extend(["", "## Unknowns"] + [f"- {value}" for value in plan_context["unknowns"]])

        lines.extend(["", "## Risks"])
        lines.append(f"- Level: {risk_info['risk']}")
        lines.append(f"- Confidence: {confidence:.2f} (mission and inspected-evidence agreement)")
        lines.extend(f"- {value}" for value in plan_context["risks"])

        lines.extend(["", "## Execution Strategy"])
        for i, s in enumerate(state.plan_steps, 1):
            lines.append(f"{i}. {s}")
        lines.extend([
            "", "## Design and implementation rationale",
            state.plan or "(The planner did not provide a design summary.)",
            "", "## Planned file operations",
        ])
        for step in state.plan_steps:
            targets = _FILE_MENTION_RE.findall(step)
            operation = "create" if re.search(r"\b(create|write|generate)\b", step, re.IGNORECASE) else "modify"
            if targets:
                lines.extend(f"- {operation}: {target} — {step}" for target in targets)
            else:
                lines.append(f"- planned operation: {step}")
        planned_files = {
            target.replace("\\", "/").lower()
            for step in state.plan_steps
            for target in _FILE_MENTION_RE.findall(step)
        }
        lines.extend(["", "## Validation Strategy"])
        if {"index.html", "styles.css", "script.js"}.issubset(planned_files):
            lines.extend([
                "- Serve and fetch index.html over local HTTP; require a successful response.",
                "- Check every linked local CSS, JavaScript, and image asset exists.",
                "- Parse local JSON data and syntax-check linked JavaScript when Node.js is available.",
                "- Check in-page navigation anchors; inspect responsive layout and runtime console/navigation interactions with a browser adapter when configured.",
            ])
        else:
            lines.append("- Run the project test command or targeted tests.")
        lines.append("- Classify any failure, gather missing context where needed, and repair before retrying.")
        try:
            with open(plan_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            log("plan_md_write_error", {"error": str(e)})

        # A plan is itself an agent-produced artifact. Review the exact
        # plan.md on disk before asking the user to approve execution.
        review = self.validator.validate_plan_file(
            state, plan_path, risk_info["risk"], confidence
        )
        if not review["approved"]:
            if review.get("requires_user_review"):
                state.status = "awaiting_plan_confirmation"
                _agent_status("Plan needs your review before execution.")
                log("agent_plan_requires_user_review", {
                    "score": review["score"], "risk": risk_info["risk"],
                    "confidence": confidence, "policy": review.get("policy"),
                })
                return (
                    f"Plan written to {plan_path}, but CODI is not confident enough to auto-approve it "
                    f"(score: {review['score']:.1f}/10; confidence: {confidence:.2f}).\n\n"
                    f"Review concern: {review['suggested_edits']}\n\n"
                    "Review the plan and type 'y' only if you accept this uncertainty, or tell me what to change."
                )
            state.plan_validation_attempts += 1
            suggestion = review["suggested_edits"]
            _agent_status(
                f"Plan scored {review['score']:.1f}/10; regenerating with validator feedback."
            )
            log("agent_plan_validation_rejected", {
                "score": review["score"], "risk": risk_info["risk"],
                "confidence": confidence, "suggestion": suggestion[:300],
                "attempt": state.plan_validation_attempts,
            })
            if state.plan_validation_attempts >= 3:
                state.status = "awaiting_plan_revision"
                return (
                    "The plan validator rejected three plan drafts, so execution has not started. "
                    f"Last score: {review['score']:.1f}/10. Needed changes: {suggestion}\n\n"
                    "Type 'retry' to regenerate from this feedback, or describe how you want the plan changed."
                )
            self.improver.create_plan(state, state.knowledge.summary_for_prompt(), validator_feedback=suggestion)
            if state.plan.startswith("[PLANNING FAILED]"):
                state.status = "failed"
                return state.plan
            return self._confirm_plan_gate(state, analysis)

        state.status = "awaiting_plan_confirmation"
        _agent_status(f"Plan ready (risk: {risk_info['risk']}) — waiting for user confirmation.")
        return (
            f"Plan written to {plan_path} (risk: {risk_info['risk']}). Review it, then type 'y' to run it, "
            f"or give me a new instruction to replan."
        )

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _run(self, state: RunState, resuming: bool = False) -> str:

        if not resuming:
            # ── Route: qa / read / edit / build ────────────────────────────────
            intent = self.planner.classify(state)

            if intent == "qa":
                _agent_status("Answering directly; no tools needed.")
                log("agent_direct", {"input": state.user_input[:80]})
                answer = self.planner.direct_answer(state)
                if _qa_answer_is_unexecuted_change(state.user_input, answer):
                    # The classifier called this "qa" but the model's own
                    # answer just described a code change in prose instead
                    # of applying it. Do not return the inert answer as if
                    # the task were done — re-route into real execution.
                    _agent_status(
                        "Answer described a code change instead of applying it — switching to execution."
                    )
                    log("agent_qa_misclassification_corrected", {
                        "input": state.user_input[:160],
                        "discarded_answer_sample": answer[:200],
                    })
                    intent = "build"
                else:
                    state.status = "complete"
                    return answer

            if intent == "read":
                _agent_status("Reading code to answer — no files will be changed.")
                log("agent_read", {"input": state.user_input[:80]})
                state.status = "complete"
                return self._handle_read(state)

            # A user who opted into a repository-wide read explicitly asked
            # for evidence before changes, so do not bypass discovery via a
            # write-oriented fast path.
            _agent_status("Checking for a fast file action.")
            fast_output = None if state.context_scope == "full" else try_fast_file_task(
                state.user_input, self.registry, state
            )
            if fast_output:
                state.current_step = state.user_input
                fast_validation = self.validator.validate_current_write(state)
                if fast_validation is not False:
                    _agent_status("Completed and validated fast file action.")
                    log("agent_fast_path", {"input": state.user_input[:80], "output": fast_output[:120]})
                    state.status = "complete"
                    return fast_output
                # A fast action is not allowed to claim success if the file
                # review rejects it. Continue through the normal plan/repair
                # loop, which can fix the file rather than stopping here.
                _agent_status("Fast file action failed validation; switching to repair plan.")
                log("agent_fast_path_validation_failed", {"notes": state.validation_notes[:200]})
                state.validation_repair_instruction = ""

            if intent == "edit" and state.context_scope != "full":
                _agent_status("Making a targeted edit.")
                edit_output = self._handle_edit(state)
                if edit_output is not None:
                    log("agent_edit_success", {"input": state.user_input[:80]})
                    state.status = "complete"
                    return edit_output
                _agent_status("Targeted edit needs broader context — planning full task.")
                log("agent_edit_fallback", {"input": state.user_input[:80]})
                # falls through to the full build pipeline below

            # ── intent == "build" (or edit fallback) ────────────────────────────
            # ── Phase 1: Read context ──────────────────────────────────────────
            _agent_status("Reading project context.")

            analysis = self.mission.analyze(state.user_input)
            state.mission = analysis
            _agent_status(
                f"Mission confidence: {analysis.confidence:.2f}"
            )
            if analysis.confidence < 0.90:
                _agent_status(
                "Need additional project context before planning."
            )
            log(
                "mission_requires_context",
                {
                    "files": analysis.files_needed,
                    "unknowns": analysis.unknowns
                }
            )

            self.context_builder = ContextBuilder(self.registry)
            context_state = self.context_builder.build(
                analysis,
                history=state.history,
                full_codebase=state.context_scope == "full",
            )
            state.knowledge = context_state.knowledge
            state.context_confidence = context_state.confidence
            context = context_state.context
            log("agent_context_ready", {"context_len": len(context)})

            if not context_state.complete:
                state.status = "awaiting_context"
                _agent_status("Context is incomplete; planning is blocked.")
                if getattr(context_state, "needs_user_clarification", False) and context_state.clarification_question:
                    return context_state.clarification_question
                return (
                    "I inspected the project and its saved .agent_history, but I still need context "
                    "to plan safely. Reply with the missing details, or type 'no' to continue using "
                    "only the verified evidence I have.\n\nUnresolved items: "
                    + "; ".join(state.knowledge.unknowns[-4:])
                )

            # ── Phase 2: Create plan ────────────────────────────────────────────
            _agent_status("Creating an execution plan.")
            self.improver.create_plan(state, context)

            # FIX: if the planning LLM call itself failed (connection error /
            # timeout — see core/improver.py create_plan()), the plan text is
            # tagged with "[PLANNING FAILED]" and plan_steps is empty. This
            # must surface as a clear failure to the user, not silently
            # proceed to write an empty plan.md and ask for confirmation on
            # nothing.
            if state.plan.startswith("[PLANNING FAILED]"):
                state.status = "failed"
                _agent_status("Planning failed — LLM backend unreachable.")
                log("agent_planning_backend_failure", {"plan": state.plan[:200]})
                return (
                    f"{state.plan}\n\n"
                    "The planning step could not reach the configured LLM backend. "
                    "Check that it is running and reachable (see config.py MODE / "
                    "LLAMACPP_URL / OLLAMA_BASE_URL), then retry."
                )

            if state.plan_steps:
                _agent_status(f"Plan ready with {len(state.plan_steps)} step(s).")
            log("agent_plan_ready", {"steps": len(state.plan_steps), "plan": state.plan})

            # ── Plan confirmation gate ───────────────────────────────────────────
            gate_message = self._confirm_plan_gate(state, analysis)
            if gate_message is not None:
                return gate_message

        elif state.status == "awaiting_plan_revision":
            analysis = state.mission or self.mission.analyze(state.user_input)
            feedback = state.plan_validation_feedback or state.plan_validation_notes
            _agent_status("Regenerating the rejected plan with validator feedback.")
            self.improver.create_plan(
                state, state.knowledge.summary_for_prompt(), validator_feedback=feedback
            )
            if state.plan.startswith("[PLANNING FAILED]"):
                state.status = "failed"
                return state.plan
            gate_message = self._confirm_plan_gate(state, analysis)
            if gate_message is not None:
                return gate_message

        elif state.status == "awaiting_context":
            declined = state.context_response.lower() in {"n", "no", "proceed", "continue"}
            approved_creates = _confirmed_missing_file_creation(
                state.context_response, state.knowledge.unknowns
            )
            if declined:
                _agent_status("Proceeding with the available verified context at the user's request.")
                log("context_declined", {"confidence": state.context_confidence})
                analysis = state.mission
            else:
                _agent_status("Rechecking project context with the user's additional details.")
                # A clarification can replace an earlier, incorrect assumption
                # about which files exist. Re-analyse it as part of the task,
                # then treat any user-named, verified paths as authoritative.
                clarification_task = (
                    f"{state.user_input}\n\nUser clarification: {state.context_response}"
                )
                # A response to CODI's own "should I create it?" question is
                # authorization, not extra project context. Re-analysing
                # "yes" used to put the same absent files back in
                # files_needed and trap empty projects in this branch.
                analysis = state.mission if approved_creates else self.mission.analyze(clarification_task)
                state.mission = analysis
                if not hasattr(self, "context_builder"):
                    self.context_builder = ContextBuilder(self.registry)
                if approved_creates:
                    approved_set = {path.replace("\\", "/") for path in approved_creates}
                    analysis.files_needed = [
                        path for path in analysis.files_needed
                        if path.replace("\\", "/") not in approved_set
                    ]
                    analysis.files_new = list(dict.fromkeys([
                        *(analysis.files_new or []), *approved_creates
                    ]))
                    state.knowledge.unknowns = [
                        item for item in state.knowledge.unknowns
                        if not _MISSING_FILE_UNKNOWN_RE.match(str(item))
                    ]
                    analysis.unknowns = [
                        item for item in analysis.unknowns
                        if "requested file" not in str(item).lower()
                    ]
                    log("context_missing_files_creation_approved", {"files": approved_creates})
                supplied_paths = list(dict.fromkeys(_FILE_MENTION_RE.findall(state.context_response)))
                resolved_paths = self.context_builder.apply_user_supplied_paths(
                    state.knowledge, supplied_paths
                )
                if resolved_paths:
                    # The user corrected the target. Do not preserve stale
                    # hallucinated component paths as hard requirements.
                    state.knowledge.unknowns = [
                        item for item in state.knowledge.unknowns
                        if not item.startswith("Requested file is not present:")
                    ]
                    analysis.files_needed = resolved_paths
                    analysis.files_new = []
                    analysis.unknowns = [
                        item for item in analysis.unknowns
                        if "requested file" not in str(item).lower()
                    ]
                    log("context_clarification_paths_applied", {
                        "paths": resolved_paths,
                        "response": state.context_response[:200],
                    })
                context_state = self.context_builder.build(
                    analysis, history=state.history, knowledge=state.knowledge,
                    full_codebase=state.context_scope == "full",
                )
                state.knowledge = context_state.knowledge
                state.context_confidence = context_state.confidence
                if not context_state.complete:
                    state.status = "awaiting_context"
                    return (
                        "I still cannot verify enough context to plan safely. Add the missing details, "
                        "or type 'no' to continue with the evidence already collected.\n\nUnresolved items: "
                        + "; ".join(state.knowledge.unknowns[-4:])
                    )
            context = state.knowledge.summary_for_prompt()
            _agent_status("Creating an execution plan.")
            self.improver.create_plan(state, context)
            if state.plan.startswith("[PLANNING FAILED]"):
                state.status = "failed"
                _agent_status("Planning failed — LLM backend unreachable.")
                log("agent_planning_backend_failure", {"plan": state.plan[:200]})
                return (
                    f"{state.plan}\n\n"
                    "The planning step could not reach the configured LLM backend. "
                    "Check that it is running and reachable, then retry."
                )

            if state.plan_steps:
                _agent_status(f"Plan ready with {len(state.plan_steps)} step(s).")
            log("agent_plan_ready", {"steps": len(state.plan_steps), "plan": state.plan, "source": "post_context_resume"})

            # FIX: this is the gate that was previously MISSING here. Without
            # it, a plan built after answering a context-clarification
            # question was never written to plan.md and state.status went
            # straight to "running" below — the user's "type y" had no real
            # plan.md behind it and no genuine confirmation checkpoint.
            gate_message = self._confirm_plan_gate(state, analysis)
            if gate_message is not None:
                return gate_message

        state.status = "running"

        # ── Phase 3: Execution loop ────────────────────────────────────────────
        while not state.is_done():
            state.iteration += 1

            # Hard cap
            if state.exceeds_max():
                _agent_status("Reached max iterations; stopping.")
                log("agent_max_iterations", {"iterations": state.iteration})
                # Reaching the safety cap is not a successful completion.
                # Keeping this as "complete" made the UI claim success even
                # when validation still had outstanding repair work.
                state.status = "failed"
                break

            # Improver decides what to do next
            _agent_status(f"Choosing next step for iteration {state.iteration}.")
            next_decision = self.improver.next_step(state)
            if not isinstance(next_decision, dict):
                next_decision = {"step": str(next_decision), "done": False}

            llm_error = next_decision.get("llm_error")
            if llm_error:
                # FIX: an empty step here means the LLM backend call itself
                # failed — NOT that the planner decided the task is done.
                # Previously this fell straight into `if done or not step:`
                # below and silently reported "task complete" after a single
                # failed connection attempt. Retry a bounded number of times,
                # then fail explicitly with the real reason.
                state.llm_backend_errors += 1
                _agent_status(
                    f"LLM backend error ({state.llm_backend_errors}/{_MAX_LLM_BACKEND_ERRORS}): {llm_error[:120]}"
                )
                log("agent_llm_backend_error", {
                    "iteration": state.iteration,
                    "count": state.llm_backend_errors,
                    "error": llm_error[:300],
                })
                if state.llm_backend_errors >= _MAX_LLM_BACKEND_ERRORS:
                    state.status = "failed"
                    state.validation_notes = f"LLM backend unreachable after {state.llm_backend_errors} attempts: {llm_error}"
                    break
                continue  # retry — do not count this as a completed/failed step

            step = str(next_decision.get("step", "") or "")
            done = bool(next_decision.get("done", False))

            if done or not step:
                _agent_status("Planner says the task is complete.")
                log("agent_improver_done", {"iteration": state.iteration})
                state.status = "complete"
                break

            _agent_status(f"Working on step {state.iteration}: {step[:120]}")
            log("agent_step", {"iteration": state.iteration, "step": step[:100]})

            state.current_step = step

            # Executor runs the step — once.
            tool_results_before = len(state.tool_results)
            dispatch_result = self.executor.execute_step(step, state)
            reflection = self.reflector.reflect(dispatch_result, state.knowledge)
            state.reflections.append({"needs_context": reflection.needs_context, "reason": reflection.reason, "unknowns": reflection.unknowns})
            if reflection.needs_context:
                _agent_status("Execution found missing context; returning to discovery.")
                log("execution_needs_context", {"reason": reflection.reason})
                context_tool = dispatch_result.get("tool", "inspect_file")
                context_args = dispatch_result.get("args", {})
                if context_args:
                    context_result = Dispatcher(self.registry).dispatch(
                        {"action": "tool_call", "tools": [{"name": context_tool, "args": context_args}]},
                        knowledge=state.knowledge,
                    )
                    for item in context_result.get("results", []):
                        state.add_tool_result(item["tool"], item["status"], item["output"])
                # Do not validate or mark an implementation step complete when
                # the executor explicitly requested more evidence.
                continue

            # A write succeeding only proves bytes reached disk. Before this
            # AST-sized step may advance, independently ask the validator LLM
            # to compare the actual file content with the original request.
            write_validation = self.validator.validate_current_write(
                state, since=tool_results_before
            )
            if write_validation is False:
                _agent_status("Written file did not satisfy the request; preparing a repair.")
                log("agent_per_write_validation_failed", {
                    "step": step[:120],
                    "notes": state.validation_notes[:300],
                })

            # Step-level completion is a deterministic fact: did the most
            # recent tool action for THIS step succeed? This is independent
            # of whether the overall task is finished — do not let the
            # semantic validator gate this.
            #
            # IMPORTANT: mark completion against state.target_plan_step, the
            # ORIGINAL plan_steps entry this attempt was satisfying — NOT
            # against `step`, the text actually sent to the executor. After
            # a validation failure, core/improver.py's next_step() rewrites
            # `step` into a repair/correction instruction ("edit_file the
            # 'refusing to overwrite' issue in index.html", etc). If that
            # rewritten text were what got appended to completed_steps, a
            # SUCCESSFUL retry would still never match anything in
            # state.plan_steps (an exact-string list) — the step remains
            # permanently "not completed", the plan-progress validation
            # guard keeps failing it, and the run silently burns every
            # remaining iteration before reporting "Stopped before
            # completion... 1 plan step(s) not yet completed" even though
            # every tool call in the log actually succeeded.
            if write_validation is not False and _step_succeeded(state, step):
                completed_target = state.target_plan_step or step
                state.mark_step_complete(completed_target)
                log("step_marked_complete", {
                    "step": completed_target[:120],
                    "executed_as": step[:120] if step != completed_target else None,
                    "completed_count": len(state.completed_steps),
                })

            # Validator now answers ONLY "is the overall task done?" —
            # not "did this step succeed" (that's already been decided above).
            # Keep a failed file review intact; the task-level validator would
            # otherwise replace its repair instruction with plan-progress text.
            if write_validation is False:
                is_valid = False
            else:
                _agent_status("Validating overall plan progress.")
                is_valid = self.validator.validate(state)

            if is_valid:
                _agent_status("Validation passed.")
                state.status = "complete"
                break

            # Validation failed — ask Improver to correct only real failures
            if not state.validation_passed and state.iteration < state.max_iterations and getattr(state, "validation_requires_correction", True):
                repair_key = "__validation_repair__"
                state.repair_attempts[repair_key] = state.repair_attempts.get(repair_key, 0) + 1
                if state.repair_attempts[repair_key] >= 3:
                    state.status = "awaiting_context"
                    target_matches = _FILE_MENTION_RE.findall(
                        state.target_plan_step or state.current_step or ""
                    )
                    target_hint = target_matches[0] if target_matches else "the intended target file"
                    notes_lower = (state.validation_notes or "").lower()
                    if "noop" in notes_lower:
                        question = (
                            f"Should I create or update {target_hint} exactly as the approved plan says? "
                            "If not, tell me which existing file should own this behavior."
                        )
                    elif "does not exist" in notes_lower:
                        question = (
                            f"I could not find {target_hint}. Should I create it, or what exact existing file should I use instead?"
                        )
                    else:
                        question = (
                            f"For {target_hint}, what behavior or dependency should CODI use that is not currently in the project?"
                        )
                    state.clarification_prompt = (
                        "I paused after three unsuccessful repair attempts. "
                        f"The last failure was: {state.validation_notes}\n\n"
                        f"Specific question: {question}\n\n"
                        "Reply with the answer and I will resume from the failing step."
                    )
                    log("agent_clarification_required", {"attempts": 3, "notes": state.validation_notes[:200]})
                    return state.clarification_prompt
                    state.clarification_prompt = (
                        "I could not safely complete this after three repair attempts. "
                        f"The last failure was: {state.validation_notes}\n\n"
                        "Please clarify the intended behavior or provide the missing dependency/configuration. "
                        "I've paused here — send a follow-up with that info and I'll resume from the failing step.\n\n"
                        f"Original plan: {state.plan or '(no plan summary available)'}"
                    )
                    log("agent_clarification_required", {"attempts": 3, "notes": state.validation_notes[:200]})
                    return state.clarification_prompt
                _agent_status(f"Validation needs repair: {state.validation_notes[:120]}")
                correction = state.validation_repair_instruction or str(self.improver.improve(state))
                log("agent_correction", {"correction": correction[:100]})
                state.validation_repair_instruction = correction

        # ── Phase 4: Final summary ────────────────────────────────────────────
        # THIS WAS PREVIOUSLY MISSING. The loop above only sets state.status
        # via break — it never itself produces user-facing output. Without
        # this call, _run() fell off the end and returned None, which is
        # why every task that reached the execution loop printed "No output
        # returned." even when files were written successfully.
        final_output = self.improver.summarize(state)
        state.final_output = final_output
        log("agent_final_summary", {
            "output": final_output[:200],
            "status": state.status,
            "iterations": state.iteration,
        })
        return final_output

    # ── Lightweight intent handlers ─────────────────────────────────────────

    def _handle_read(self, state: RunState) -> str:
        """
        Answer a question about the code without writing anything.
        Reads named file(s) if present in the input, otherwise falls back
        to a semantic search + directory listing. Never dispatches a
        write/edit/create tool — this path is read-only by construction.
        """
        dispatcher = Dispatcher(self.registry)
        file_matches = list(dict.fromkeys(_FILE_MENTION_RE.findall(state.user_input)))

        if file_matches:
            tools_to_run = [{"name": "read_file", "args": {"path": p}} for p in file_matches]
        else:
            tools_to_run = [
                {"name": "search_codebase", "args": {"query": state.user_input[:200]}},
                {"name": "list_files", "args": {}},
            ]

        result = dispatcher.dispatch({"action": "tool_call", "tools": tools_to_run})

        context_parts = []
        for r in result.get("results", []):
            if r["status"] == "ok":
                context_parts.append(
                    wrap_prompt_data(r["output"], path=(r.get("args") or {}).get("path"))
                )
            state.add_tool_result(r["tool"], r["status"], r["output"])

        context_text = "\n\n".join(context_parts) or "(no matching files found)"

        try:
            resp = self.improver.llm.invoke([
                SystemMessage(content=(
                    "You are Codi, explaining code to the user. You are in "
                    "read-only mode — never claim to have written or changed "
                    "any file, and never invent file contents you haven't seen."
                )),
                HumanMessage(content=(
                    f"Answer using ONLY the context below.\n\n"
                    f"Question: {state.user_input}\n\nContext:\n{context_text}"
                )),
            ])
            answer = resp.content.strip()
            log("agent_read_answer", {"output": answer[:120]})
            return answer
        except Exception as e:
            log("agent_read_error", {"error": str(e)})
            return f"Error reading code: {e}"

    def _handle_edit(self, state: RunState):
        """
        Attempt a targeted, single-file edit with one direct executor call —
        skips the full read-context + multi-step-plan pipeline entirely.

        Returns the summary string on success, or None if it couldn't be
        resolved this way, so the caller falls back to the full build loop
        instead of failing outright.
        """
        step = state.user_input
        state.current_step = step

        tool_results_before = len(state.tool_results)
        self.executor.execute_step(step, state)

        if self.validator.validate_current_write(state, since=tool_results_before) is False:
            log("agent_edit_per_write_validation_failed", {"notes": (state.validation_notes or "")[:160]})
            return None

        if not _step_succeeded(state):
            log("agent_edit_step_failed", {"step": step[:120]})
            return None

        state.mark_step_complete(step)
        state.plan_steps = [step]

        if self.validator.validate(state):
            return self.improver.summarize(state)

        log("agent_edit_validation_failed", {"notes": (state.validation_notes or "")[:160]})
        return None


def create_agent(mode: str = None) -> CodiAgent:
    """
    Factory function. Loads all tools and returns a ready CodiAgent.
    Called once at startup from main.py.
    """
    import config
    effective_mode = mode or config.MODE

    print(f"  [Agent] Loading tools for mode: {effective_mode}")
    _global_registry.load_all(mode=effective_mode)
    print(f"  [Agent] Registry ready — {len(_global_registry.list_names())} tools")

    log("agent_created", {"tools": len(_global_registry.list_names()), "mode": effective_mode})
    return CodiAgent(registry=_global_registry)
