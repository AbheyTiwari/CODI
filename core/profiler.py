# core/profiler.py
# ─────────────────────────────────────────────────────────────────────────────
# Agent Profiler — records timing, LLM call counts, token counts, cache
# hits, and tool activity for a single agent run, then produces a report.
#
# Design principles (matching existing architecture):
#   - One profiler instance per RunState (attached as state.profiler),
#     mirroring how state.knowledge / state.requirements are scoped per-run.
#   - Uses logger.log() for persistence — no new storage layer.
#   - Read by agent.py at the end of _run() to print/attach a summary.
#   - Every other module (llm_factory, dispatcher, executor, improver,
#     validator) calls into this via a thread-local "active profiler"
#     rather than needing RunState threaded into every function signature —
#     this avoids a large, invasive signature-changing refactor across
#     8 files, which would violate "prefer minimal, reversible changes".
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from logger import log


# ── Thread-local active profiler ──────────────────────────────────────────────
# Each agent run happens on its own call stack (main.py invokes agent_executor
# synchronously per user turn). A thread-local lets llm_factory.py and
# dispatcher.py — which have no reference to RunState — find "the profiler
# for the run currently executing" without threading a new parameter through
# every function signature in the codebase.
_local = threading.local()


def get_active_profiler() -> "AgentProfiler | None":
    return getattr(_local, "profiler", None)


@contextmanager
def active_profiler(profiler: "AgentProfiler"):
    """Bind `profiler` as the active profiler for the duration of a run."""
    previous = getattr(_local, "profiler", None)
    _local.profiler = profiler
    try:
        yield profiler
    finally:
        _local.profiler = previous


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class LLMCallRecord:
    role: str                 # "refiner" | "coder" | "validator"
    caller: str                # e.g. "improver.create_plan", "executor.execute_step"
    prompt_tokens_est: int
    completion_tokens_est: int
    duration_s: float
    cache_hit: bool = False
    error: str | None = None


@dataclass
class ToolCallRecord:
    tool: str
    duration_s: float
    status: str                # "ok" | "error"
    args_path: str | None = None


@dataclass
class PhaseRecord:
    name: str                  # "mission" | "context" | "plan" | "execute" | "validate" | "repair"
    duration_s: float


@dataclass
class AgentProfiler:
    task: str = ""
    start_time: float = field(default_factory=time.monotonic)
    end_time: float | None = None

    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    phases: list[PhaseRecord] = field(default_factory=list)

    files_read: set[str] = field(default_factory=set)
    files_modified: set[str] = field(default_factory=set)

    context_cache_hits: int = 0
    context_cache_misses: int = 0

    # ── Recording API ────────────────────────────────────────────────────────

    def record_llm_call(
        self, role: str, caller: str, prompt_tokens_est: int,
        completion_tokens_est: int, duration_s: float,
        cache_hit: bool = False, error: str | None = None,
    ) -> None:
        self.llm_calls.append(LLMCallRecord(
            role=role, caller=caller,
            prompt_tokens_est=prompt_tokens_est,
            completion_tokens_est=completion_tokens_est,
            duration_s=duration_s, cache_hit=cache_hit, error=error,
        ))

    def record_tool_call(self, tool: str, duration_s: float, status: str, args_path: str | None = None) -> None:
        self.tool_calls.append(ToolCallRecord(tool=tool, duration_s=duration_s, status=status, args_path=args_path))
        if tool in ("read_file", "read_file_numbered", "inspect_file") and args_path:
            self.files_read.add(args_path)
        if tool in ("create_file", "write_file", "edit_file", "apply_patch") and status == "ok" and args_path:
            self.files_modified.add(args_path)

    @contextmanager
    def phase(self, name: str):
        """Wrap a pipeline phase (mission/context/plan/execute/validate/repair)."""
        started = time.monotonic()
        try:
            yield
        finally:
            self.phases.append(PhaseRecord(name=name, duration_s=time.monotonic() - started))

    def record_context_cache(self, hit: bool) -> None:
        if hit:
            self.context_cache_hits += 1
        else:
            self.context_cache_misses += 1

    def finish(self) -> None:
        self.end_time = time.monotonic()

    # ── Reporting ────────────────────────────────────────────────────────────

    def _phase_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for p in self.phases:
            totals[p.name] = totals.get(p.name, 0.0) + p.duration_s
        return totals

    def _llm_wait_time(self) -> float:
        return sum(c.duration_s for c in self.llm_calls)

    def _tool_time(self) -> float:
        return sum(c.duration_s for c in self.tool_calls)

    def top_bottlenecks(self, n: int = 3) -> list[tuple[str, float, str]]:
        """
        Return up to n (label, seconds, explanation) tuples, ranked by
        wall-clock contribution — the three biggest time sinks, whatever
        category they fall in (a phase, a single slow LLM call, or a
        single slow tool call).
        """
        candidates: list[tuple[str, float, str]] = []

        for name, total in self._phase_totals().items():
            candidates.append((f"phase:{name}", total, f"{name} phase total across the run"))

        if self.llm_calls:
            slowest_llm = max(self.llm_calls, key=lambda c: c.duration_s)
            candidates.append((
                f"llm_call:{slowest_llm.caller}",
                slowest_llm.duration_s,
                f"single slowest LLM call ({slowest_llm.role}, ~{slowest_llm.prompt_tokens_est} prompt tokens)",
            ))
            total_llm_calls = len(self.llm_calls)
            if total_llm_calls > 10:
                candidates.append((
                    "llm_call_count",
                    self._llm_wait_time(),
                    f"{total_llm_calls} separate LLM round trips — each one pays full network+inference latency",
                ))

        if self.tool_calls:
            slowest_tool = max(self.tool_calls, key=lambda c: c.duration_s)
            candidates.append((
                f"tool_call:{slowest_tool.tool}",
                slowest_tool.duration_s,
                "single slowest tool call",
            ))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:n]

    def summary(self) -> str:
        if self.end_time is None:
            self.finish()

        total_wall = self.end_time - self.start_time
        llm_wait = self._llm_wait_time()
        tool_time = self._tool_time()
        phase_totals = self._phase_totals()

        prompt_tokens = sum(c.prompt_tokens_est for c in self.llm_calls)
        completion_tokens = sum(c.completion_tokens_est for c in self.llm_calls)
        avg_tps = (completion_tokens / llm_wait) if llm_wait > 0 else 0.0

        cache_total = self.context_cache_hits + self.context_cache_misses
        cache_rate = (self.context_cache_hits / cache_total * 100) if cache_total else 0.0

        largest_prompt = max((c.prompt_tokens_est for c in self.llm_calls), default=0)
        longest_op_name, longest_op_s = "", 0.0
        for c in self.llm_calls:
            if c.duration_s > longest_op_s:
                longest_op_name, longest_op_s = f"llm:{c.caller}", c.duration_s
        for t in self.tool_calls:
            if t.duration_s > longest_op_s:
                longest_op_name, longest_op_s = f"tool:{t.tool}", t.duration_s

        lines = [
            "── Performance Summary ──────────────────────────",
            f"Total wall-clock:      {total_wall:.2f}s",
            f"  LLM wait time:       {llm_wait:.2f}s ({llm_wait / total_wall * 100:.0f}% of total)" if total_wall else f"  LLM wait time:       {llm_wait:.2f}s",
            f"  Tool execution time: {tool_time:.2f}s",
        ]
        for name, total in sorted(phase_totals.items(), key=lambda x: -x[1]):
            lines.append(f"  Phase '{name}':".ljust(24) + f"{total:.2f}s")

        lines += [
            "",
            f"LLM calls:             {len(self.llm_calls)}",
            f"Tool calls:            {len(self.tool_calls)}",
            f"Prompt tokens (est):   {prompt_tokens}",
            f"Completion tokens:     {completion_tokens}",
            f"Avg tokens/sec:        {avg_tps:.1f}",
            f"Largest prompt:        {largest_prompt} tokens",
            f"Longest operation:     {longest_op_name} ({longest_op_s:.2f}s)",
            f"Files read:            {len(self.files_read)}",
            f"Files modified:        {len(self.files_modified)}",
            f"Context cache hit rate: {cache_rate:.0f}% ({self.context_cache_hits}/{cache_total})" if cache_total else "Context cache hit rate: n/a (no cache-aware retrieval yet)",
            "",
            "Top 3 bottlenecks:",
        ]
        for label, seconds, explanation in self.top_bottlenecks(3):
            lines.append(f"  - {label}: {seconds:.2f}s — {explanation}")

        lines.append("──────────────────────────────────────────────────")

        log("profiler_summary", {
            "task": self.task[:160],
            "total_wall_s": round(total_wall, 3),
            "llm_wait_s": round(llm_wait, 3),
            "tool_time_s": round(tool_time, 3),
            "llm_calls": len(self.llm_calls),
            "tool_calls": len(self.tool_calls),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "avg_tps": round(avg_tps, 2),
            "cache_hit_rate": round(cache_rate, 1),
            "files_read": len(self.files_read),
            "files_modified": len(self.files_modified),
            "bottlenecks": [{"label": l, "seconds": round(s, 3)} for l, s, _ in self.top_bottlenecks(3)],
        })

        return "\n".join(lines)