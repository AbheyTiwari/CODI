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
# IMPORTANT: _run() MUST return a string in every code path. The execution
# loop below sets state.status = "complete" via break but does not itself
# produce user-facing output — the final summarize() call at the bottom of
# _run() is what turns "the loop finished" into an actual answer. Without
# it, invoke() returns output=None and the caller prints "No output
# returned." even after files were written successfully.
# ─────────────────────────────────────────────────────────────────────────────

import re
import traceback
import os
from langchain_core.messages import HumanMessage, SystemMessage
from core.mission_analyzer import MissionAnalyzer
from core.context_builder import ContextBuilder
from core.execution_reflector import ExecutionReflector
from core.executor  import Executor
from core.improver  import Improver
from core.planner   import Planner
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


def _step_requires_mutation(step: str) -> bool:
    return any(re.search(rf"\b{verb}\b", (step or "").lower()) for verb in _IMPLEMENTATION_VERBS)


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
    if _step_requires_mutation(step):
        return last.status == "ok" and last.tool in {"create_file", "write_file", "edit_file"}
    if last.status == "ok":
        return True
    # dispatcher noop/duplicate-write signals also count as step success —
    # they mean "nothing left to do here," not "this failed"
    if last.tool == "dispatcher" and last.output in ("noop", "done"):
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
                state.plan_confirmed = True
            elif state.status == "awaiting_context":
                response = str(inputs.get("context_response", "")).strip()
                state.context_response = response
                if response:
                    state.history = f"{state.history}\nUser context response: {response}".strip()
                # A declined request means proceed with the verified evidence
                # already gathered. Any other response is treated as added
                # context and discovery gets another pass before planning.
                state.plan_confirmed = False
        else:
            state = RunState(
                user_input=inputs.get("input", ""),
                history=inputs.get("history", ""),
            )
            state.context_scope = "full" if inputs.get("read_entire_codebase") else "targeted"
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
            "tool_outputs": state.recent_tool_outputs(n=10),
            "state":        state,
        }

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _run(self, state: RunState, resuming: bool = False) -> str:

        if not resuming:
            # ── Route: qa / read / edit / build ────────────────────────────────
            intent = self.planner.classify(state)

            if intent == "qa":
                _agent_status("Answering directly; no tools needed.")
                log("agent_direct", {"input": state.user_input[:80]})
                state.status = "complete"
                return self.planner.direct_answer(state)

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
                _agent_status("Completed with fast file action.")
                log("agent_fast_path", {"input": state.user_input[:80], "output": fast_output[:120]})
                state.status = "complete"
                return fast_output

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
            if state.plan_steps:
                _agent_status(f"Plan ready with {len(state.plan_steps)} step(s).")
            log("agent_plan_ready", {"steps": len(state.plan_steps), "plan": state.plan})

            # ── Plan confirmation gate ───────────────────────────────────────────
            if not state.plan_confirmed:
                plan_path = os.path.join(
                    os.environ.get("CODI_WORKING_DIR", os.getcwd()), "plan.md"
                )
                plan_context = state.knowledge.plan_context()
                lines = [f"# Plan: {state.plan}", "", "## Mission", analysis.goal, "", "## Understanding", state.knowledge.summary_for_prompt(), "", "## Architecture", "```json", str(plan_context["dependency_graph"]), "```", "", "## Files inspected"]
                lines.extend(f"- {path}" for path in plan_context["files_inspected"])
                lines.extend(["", "## Assumptions"] + [f"- {value}" for value in analysis.assumptions])
                lines.extend(["", "## Unknowns"] + [f"- {value}" for value in plan_context["unknowns"]])
                lines.extend(["", "## Risks"] + [f"- {value}" for value in plan_context["risks"]])
                lines.extend(["", "## Execution Strategy"])
                for i, s in enumerate(state.plan_steps, 1):
                    lines.append(f"{i}. {s}")
                lines.extend(["", "## Validation Strategy", "- Run the project test command or targeted tests.", "- Classify any failure, gather missing context where needed, and repair before retrying."])
                try:
                    with open(plan_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                except Exception as e:
                    log("plan_md_write_error", {"error": str(e)})

                state.status = "awaiting_plan_confirmation"
                _agent_status("Plan ready — waiting for user confirmation.")
                return (
                    f"Plan written to {plan_path}. Review it, then type 'y' to run it, "
                    f"or give me a new instruction to replan."
                )

        elif state.status == "awaiting_context":
            declined = state.context_response.lower() in {"n", "no", "proceed", "continue"}
            if declined:
                _agent_status("Proceeding with the available verified context at the user's request.")
                log("context_declined", {"confidence": state.context_confidence})
            else:
                _agent_status("Rechecking project context with the user's additional details.")
                analysis = state.mission or self.mission.analyze(state.user_input)
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

            # Step-level completion is a deterministic fact: did the most
            # recent tool action for THIS step succeed? This is independent
            # of whether the overall task is finished — do not let the
            # semantic validator gate this.
            if _step_succeeded(state, step):
                state.mark_step_complete(step)
                log("step_marked_complete", {"step": step[:120], "completed_count": len(state.completed_steps)})

            # Validator now answers ONLY "is the overall task done?" —
            # not "did this step succeed" (that's already been decided above).
            _agent_status("Validating the result.")
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

        self.executor.execute_step(step, state)

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