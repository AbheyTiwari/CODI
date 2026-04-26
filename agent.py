# agent.py
# ─────────────────────────────────────────────────────────────────────────────
# The new Codi agent. Clean, explicit, no LangGraph magic.
#
# Loop:
#   while not done:
#       step  = improver.next_step(state)
#       result = executor.execute_step(step, state)
#       done  = validator.validate(state)
#       if not done and validation failed:
#           correction = improver.improve(state)
#           → retry with correction
#   output = improver.summarize(state)
# ─────────────────────────────────────────────────────────────────────────────

import traceback

from core.executor  import Executor
from core.improver  import Improver
from core.planner   import Planner
from core.quick_actions import try_fast_file_task
from core.validator import Validator
from logger         import log
from state.temp_db  import RunState
from tools.registry import ToolRegistry, registry as _global_registry


class CodiAgent:
    def __init__(self, registry: ToolRegistry = None):
        self.registry  = registry or _global_registry
        self.planner   = Planner()
        self.improver  = Improver(self.registry)
        self.executor  = Executor(self.registry)
        self.validator = Validator()

    def invoke(self, inputs: dict) -> dict:
        """
        Main entry point. Mirrors the old CodiGraphAgent.invoke() interface
        so main.py needs minimal changes.

        inputs: {"input": str, "history": str}
        returns: {"output": str, "tool_outputs": list[str]}
        """
        state = RunState(
            user_input=inputs.get("input", ""),
            history=inputs.get("history", ""),
        )

        log("agent_start", {"input": state.user_input[:120]})

        try:
            output = self._run(state)
        except Exception as e:
            log("agent_crash", {"error": str(e), "traceback": traceback.format_exc()[:4000]})
            output = f"Agent error: {e}"

        log("agent_end", {"output": output[:120], "iterations": state.iteration})

        return {
            "output":       output,
            "tool_outputs": state.recent_tool_outputs(n=10),
        }

    # ── Core loop ─────────────────────────────────────────────────────────────

    def _run(self, state: RunState) -> str:

        # ── Route: simple Q&A or full execution? ──────────────────────────────
        if not self.planner.needs_execution(state):
            log("agent_direct", {"input": state.user_input[:80]})
            state.status = "complete"
            return self.planner.direct_answer(state)

        fast_output = try_fast_file_task(state.user_input, self.registry, state)
        if fast_output:
            log("agent_fast_path", {"input": state.user_input[:80], "output": fast_output[:120]})
            state.status = "complete"
            return fast_output

        # ── Phase 1: Read context ──────────────────────────────────────────────
        context = self.improver.read_context(state)
        log("agent_context_ready", {"context_len": len(context)})

        # ── Phase 2: Create plan ───────────────────────────────────────────────
        self.improver.create_plan(state, context)
        log("agent_plan_ready", {"steps": len(state.plan_steps), "plan": state.plan})

        state.status = "running"

        # ── Phase 3: Execution loop ────────────────────────────────────────────
        while not state.is_done():
            state.iteration += 1

            # Hard cap
            if state.exceeds_max():
                log("agent_max_iterations", {"iterations": state.iteration})
                state.status = "complete"
                break

            # Improver decides what to do next
            next_decision = self.improver.next_step(state)
            if not isinstance(next_decision, dict):
                next_decision = {"step": str(next_decision), "done": False}
            step = str(next_decision.get("step", "") or "")
            done = bool(next_decision.get("done", False))

            if done or not step:
                log("agent_improver_done", {"iteration": state.iteration})
                state.status = "complete"
                break

            log("agent_step", {"iteration": state.iteration, "step": step[:100]})

            # Executor runs the step
            self.executor.execute_step(step, state)

            # Validator checks if we're done
            is_valid = self.validator.validate(state)

            if is_valid:
                state.status = "complete"
                break

            # Validation failed — ask Improver to correct
            if not state.validation_passed and state.iteration < state.max_iterations:
                correction = str(self.improver.improve(state))
                log("agent_correction", {"correction": correction[:100]})
                # Inject correction as next step context
                state.plan = f"{state.plan}\n[CORRECTION]: {correction}"

        # ── Phase 4: Final output ──────────────────────────────────────────────
        output = self.improver.summarize(state)
        state.final_output = output
        return output


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
