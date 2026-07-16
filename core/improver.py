# core/improver.py
# ─────────────────────────────────────────────────────────────────────────────
# The Improver is the orchestrating LLM (fast/cheap).
#
# Responsibilities:
#   1. Read context from the codebase
#   2. Build an execution plan
#   3. Decide which step to send to the Coder next
#   4. Generate corrections when validation fails
#   5. Produce the final user-facing summary
#
# All system-level prompt strings live in core/prompts.py.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import ast
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log
from core.prompts import planner_system_prompt, correction_system_prompt
from dispatcher import Dispatcher, wrap_prompt_data
from state.temp_db import RunState, TaskRequirements
from tools.registry import ToolRegistry


# ── Orchestrator system message ───────────────────────────────────────────────
_ORCHESTRATOR_SYSTEM = """\
You are the Improver — the orchestrator of a coding agent called Codi.
You plan, coordinate, and decide. You never call tools yourself.
Always respond in valid JSON (no prose, no markdown fences).
Working directory: {working_dir}"""


# ── Requirements extraction prompt ───────────────────────────────────────────
_REQUIREMENTS_PROMPT = """\
Extract the task requirements from this user request.

User request: {task}

Respond ONLY with JSON — no fences, no prose:
{{
  "framework": "fastapi" or "flask" or "react" or "vanilla" or "django" or null,
  "must_have": ["list of concrete things that must exist in the output"],
  "must_not":  ["list of things explicitly forbidden or that would contaminate the stack"],
  "files":     ["list of files that must be created or modified"]
}}

Rules:
- framework: only set if a specific framework is named or clearly implied
- must_have: be specific — "FastAPI GET / endpoint" not just "endpoint"
- must_not: always include competing frameworks if framework is set
- files: include all files needed to complete the task
- Keep each list item under 60 chars

JSON only:"""


# ── Plan prompt ───────────────────────────────────────────────────────────────
_PLAN_PROMPT = """\
Task: {task}

Requirements:
{requirements}

Available tools: {tools}

Codebase context:
{context}

Produce an execution plan. Respond ONLY with JSON — no fences, no prose:
{{"plan":"one sentence summary","steps":["Step 1: ...","Step 2: ..."]}}

Rules:
- Maximum 5 steps. Each step is a plain string — NOT an object.
- Reconcile every file name with the evidence before using it. If an intended
  file is absent, make the step explicitly create it; do not say "edit".
- Include a verification step only after all implementation steps.
- ONLY reference files that actually appear in the codebase context above.
  Do NOT invent a filename that is a typo or guess (e.g. do not write
  "scripts.js" if the context shows "script.js") — copy the exact filename
  as it appears in the context.
- If task mentions [BOILERPLATE CREATED: file1, file2], those files exist.
  Plan EDIT steps only — do NOT plan to create them again.
- For simple single-file tasks, ONE step is enough.
- Be specific: name the file, name the tool, name the content.
- Every implementation step MUST name at least one target file. Never use a
  vague step such as "Add CSS" or "Implement JavaScript"; say exactly where
  it will be added. A read-only action is not an implementation step.
- Do not plan browser navigation, screenshots, or `file:` URLs for local
  files. Use source reads and the semantic validator unless the user supplied
  a running http:// or https:// application URL.
- For a unit-test request, the named source file is READ ONLY. Plan a new or
  existing test file (for example `test_main.py`) and tests that import the
  source; never write, recreate, or replace the source file.

Example (simple):
{{"plan":"Create a greeting HTML page","steps":["Write hello.html with full HTML greeting Versha and Shubham"]}}

Example (edit existing):
{{"plan":"Add content to boilerplate files","steps":["Edit index.html to add hero section and greeting","Edit styles.css to style the hero section"]}}

JSON only:"""


# ── Next step prompt ──────────────────────────────────────────────────────────
_NEXT_STEP_PROMPT = """\
Task: {task}
Requirements:
{requirements}
Plan: {plan}

All plan steps:
{plan_steps}

Steps completed so far: {done_steps}
Tool results so far:
{tool_results}

Return the next uncompleted step from the plan above.
Copy the step text EXACTLY as written in the plan.
Respond ONLY with JSON — no fences:
{{"step":"exact step text","done":false}}

If ALL steps are done, respond:
{{"step":"","done":true}}

JSON only:"""


# ── Final summary prompt ──────────────────────────────────────────────────────
_FINAL_PROMPT = """\
Task: {task}
Tool results:
{tool_results}

Summarize what was accomplished in 2-3 sentences.
Name the exact files created or changed.
Be direct. No caveats. No suggestions.
Plain text only (not JSON):"""


def _as_text(value) -> str:
    """Safely coerce any value to a non-None string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _unique(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _extract_file_refs(text: str) -> list[str]:
    pattern = r"(?<![\w.-])([A-Za-z0-9_./\\-]+\.(?:py|java|html|css|js|ts|jsx|tsx|json|md|txt|xml|yml|yaml|sh|sql|svg))"
    return _unique([m.group(1).strip("'\"` ") for m in re.finditer(pattern, text or "")])


_VERIFICATION_WORDS = ("verify", "validate", "test", "check", "screenshot")
_IMPLEMENTATION_WORDS = ("add", "change", "create", "edit", "fix", "implement", "insert", "modify", "remove", "replace", "style", "update", "write")


# FIX (bug 1): human-readable descriptions of each strategy name recorded by
# core/executor.py, used to build an explicit "do not repeat these" block in
# the correction prompt. Keeping this mapping here (not duplicated inline)
# means adding a new executor strategy only requires one new entry.
_STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "edit_first_textmatch": "exact old/new text-match replacement",
    "edit_first_additive": "additive edit_file (append-style) targeting a specific anchor",
    "line_range_edit": "line-number-based replace_lines/delete_lines/insert_at_line",
    "additive_append": "appending new code to the end of the file",
    "content_first": "regenerating the file's full content from scratch",
    "edit_missing_file": "editing a file that does not exist on disk",
    "standard_json": "letting the coder freely choose a tool via the generic JSON path",
}


def _best_file_match(step: str, known_files: list[str]) -> str | None:
    """
    Deterministically pick the most plausible target file for a step that
    mentions no explicit file path, using simple keyword overlap between the
    step text and each candidate file's name/path segments.

    This replaces a previous approach that injected a literal
    "list files and identify which file this targets" step into the plan.
    That text was written as an instruction for an LLM to reason about, but
    the executor treated it as an ordinary implementation step with nothing
    concrete to write — the coder correctly had no file to act on, called
    list_files exactly as instructed, and then had no further action,
    producing a noop that burned all repair attempts on a step that could
    never have succeeded. Resolving the file here, deterministically, at
    plan-normalization time means every step handed to the executor always
    names a real target before execution ever starts.
    """
    if not known_files:
        return None
    if len(known_files) == 1:
        return known_files[0]

    step_words = set(re.findall(r"[a-z0-9]+", step.lower()))
    # Drop generic verbs/filler so they don't dilute the match against
    # every candidate equally.
    step_words -= set(_IMPLEMENTATION_WORDS) | {"the", "to", "and", "for", "this", "a", "an", "of", "in"}

    best_file = None
    best_score = 0
    for candidate in known_files:
        base = os.path.basename(candidate).lower()
        stem = re.sub(r"\.[a-z0-9]+$", "", base)
        candidate_words = set(re.findall(r"[a-z0-9]+", stem.replace("_", " ").replace("-", " ")))
        score = len(step_words & candidate_words)
        if score > best_score:
            best_score = score
            best_file = candidate

    if best_file and best_score > 0:
        return best_file

    # No keyword overlap at all — fall back to the first known file rather
    # than leaving the step unresolved. This is a deliberate best-effort
    # guess, but it's still an executable one; the old behavior (an
    # unexecutable "go figure out the file" step) was strictly worse.
    return known_files[0]


def _normalize_plan_steps(steps: list[str], requirements: TaskRequirements) -> list[str]:
    """Make every implementation step file-targeted before execution begins."""
    normalized: list[str] = []
    known_files = _unique(requirements.files)
    for raw_step in steps:
        step = _as_text(raw_step).strip()
        if not step:
            continue
        lowered = step.lower()
        # Browser MCP cannot open file:// URLs. Local HTML is verified from
        # source unless the user explicitly supplied a running web URL.
        if any(term in lowered for term in ("browser", "reload the page", "screenshot")):
            target = _extract_file_refs(step) or known_files
            if target:
                step = f"Read {target[0]} and verify its HTML structure and requested content"
                lowered = step.lower()
        is_verification = any(word in lowered for word in _VERIFICATION_WORDS)
        needs_file = any(re.search(rf"\b{word}\b", lowered) for word in _IMPLEMENTATION_WORDS)
        if needs_file and not is_verification and not _extract_file_refs(step):
            resolved_file = _best_file_match(step, known_files)
            if resolved_file:
                log("improver_step_file_resolved", {
                    "step": step[:160], "resolved_file": resolved_file,
                })
                normalized.append(f"{step} in {resolved_file}")
            else:
                # No known files at all to resolve against (e.g. a
                # from-scratch project with nothing indexed yet) — leave
                # the step as-is rather than injecting an unexecutable
                # placeholder; the executor's own file-detection can still
                # pick up an explicit path if the coder names one.
                normalized.append(step)
        else:
            normalized.append(step)
    return normalized


def _deterministic_requirements(task: str) -> TaskRequirements:
    lowered = (task or "").lower()
    framework = None
    # Pick an explicitly requested stack before scanning framework names.
    # A naive substring scan classified "no React" as React and caused the
    # coder to generate exactly the framework the user prohibited.
    if any(term in lowered for term in (
        "vanilla javascript", "vanilla js", "plain javascript", "plain html",
        "html5", "no framework", "no frameworks",
    )):
        framework = "vanilla"
    else:
        for candidate in ("fastapi", "flask", "django", "react"):
            if candidate in lowered and not re.search(rf"\b(no|without)\s+{candidate}\b", lowered):
                framework = candidate
                break

    files = _extract_file_refs(task)
    reqs = TaskRequirements(framework=framework, files=files)
    if framework:
        reqs.must_have.append(f"{framework} implementation")
        reqs.must_not.extend(reqs.framework_lock())
    return reqs


def _is_unit_test_task(task: str) -> bool:
    lowered = (task or "").lower()
    return "unit test" in lowered or "unittest" in lowered or "pytest" in lowered or "test case" in lowered


def _is_broad_every_function_request(task: str) -> bool:
    lowered = (task or "").lower()
    return _is_unit_test_task(task) and any(phrase in lowered for phrase in (
        "each function", "every function", "all functions", "each of functions",
    ))


def _function_by_function_test_steps() -> list[str]:
    """Use Python AST so a small model receives one function-sized task at a time."""
    root = os.environ.get("CODI_WORKING_DIR", os.getcwd())
    steps: list[str] = []
    skipped = {".git", ".venv", "venv", "node_modules", "__pycache__", ".codi"}
    for directory, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in skipped]
        for filename in sorted(files):
            if not filename.endswith(".py") or filename.startswith("test_"):
                continue
            absolute = os.path.join(directory, filename)
            try:
                with open(absolute, "r", encoding="utf-8", errors="replace") as handle:
                    tree = ast.parse(handle.read())
            except (OSError, SyntaxError):
                continue
            source = os.path.relpath(absolute, root).replace("\\", "/")
            target = f"test_{os.path.splitext(filename)[0]}.py"
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    steps.append(
                        f"Create or update {target} with unit tests only for function {node.name} in {source}"
                    )
    if steps:
        steps.append("Run the focused Python tests and report any failing function")
    return steps


def _test_file_for(source_file: str, project_files: list[str]) -> str:
    stem, extension = os.path.splitext(os.path.basename(source_file))
    filename = f"test_{stem}{extension or '.py'}"
    normalized = [str(path).replace("\\", "/") for path in project_files]
    for directory in ("tests", "test"):
        if any(path.startswith(f"{directory}/") for path in normalized):
            return f"{directory}/{filename}"
    return filename


def _enforce_test_intent(steps: list[str], requirements: TaskRequirements) -> list[str]:
    """Keep test generation additive: source is read-only, test file is the target."""
    if not requirements.protected_files or not requirements.files:
        return steps
    source_file = requirements.protected_files[0]
    test_file = requirements.files[0]
    adjusted: list[str] = []
    test_written = False
    for step in steps:
        lowered = step.lower()
        if any(word in lowered for word in ("run ", "verify", "validate", "pytest", "unittest")):
            adjusted.append(f"Run the focused tests in {test_file} against {source_file}")
            continue
        verb = "Edit" if test_written else "Create"
        adjusted.append(f"{verb} {test_file} with focused unit tests for behavior in {source_file}")
        test_written = True
    return adjusted


def _files_from_tool_results(state: RunState) -> list[str]:
    """
    Extract the list of files that were actually, successfully modified.

    CRITICAL FIX: this previously iterated ALL tool_results regardless of
    status, so a failed edit_file call (e.g. "ERROR editing scripts.js:
    file does not exist") still had its path text-matched by
    _extract_file_refs and reported to the user as "Changed" — even though
    nothing was written. The summary must only ever reflect real,
    successful writes.
    """
    files = []
    for result in state.tool_results:
        if result.tool not in ("create_file", "write_file", "edit_file", "apply_patch"):
            continue
        if result.status != "ok":
            continue  # <-- the fix: skip failed attempts entirely

        output = result.output or ""
        if output.strip().startswith("{"):
            try:
                payload = json.loads(output)
                if payload.get("success") is False:
                    continue
                path = payload.get("file_modified") or payload.get("path")
                if path:
                    files.append(str(path))
                    continue
            except json.JSONDecodeError:
                pass
        files.extend(_extract_file_refs(output))
    return _unique(files)


def _known_files_for_lint(state: RunState) -> set[str]:
    """Files the plan may assume already exist: verified project files plus
    anything TaskRequirements already extracted from the task text."""
    files: set[str] = set()
    try:
        project_files = state.knowledge.project.get("files", []) if state.knowledge else []
        files.update(str(f).replace("\\", "/") for f in project_files)
    except Exception:
        pass
    files.update(f.replace("\\", "/") for f in (state.requirements.files or []))
    return files


_LINT_CREATE_WORDS = ("create", "write", "generate", "produce")
_LINT_EDIT_WORDS = ("edit", "update", "modify", "change", "fix", "append", "insert", "remove", "replace", "style", "rename")


def _plan_lint_violations(steps: list[str], requirements: TaskRequirements, known_files: set[str]) -> list[str]:
    """
    Deterministic, no-LLM pass over a candidate plan — catches the most
    common self-contradictions before a single tool call burns iteration
    budget:
      (a) a step edits a file that is neither in the verified project
          context nor created by an earlier step in this same plan
      (b) a step's text contains a term forbidden by the locked framework
          (the same patterns TaskRequirements.framework_lock() feeds to the
          validator's post-execution contamination check — checked here too,
          BEFORE execution, not only after)
      (c) two steps target the identical (action, file) pair — a duplicate
          create or duplicate edit of the same target, almost always a sign
          the plan is repeating itself rather than making progress
    """
    violations: list[str] = []
    available = {f.lower() for f in known_files}
    seen_targets: set[tuple[str, str]] = set()
    forbidden = [p.lower() for p in requirements.framework_lock() if p]

    for index, step in enumerate(steps, start=1):
        lowered = (step or "").lower()

        for pattern in forbidden:
            if pattern in lowered:
                violations.append(
                    f"Step {index} references forbidden term '{pattern}' "
                    f"(task is locked to {requirements.framework})."
                )

        refs = _extract_file_refs(step)
        creates = any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in _LINT_CREATE_WORDS)
        edits = any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in _LINT_EDIT_WORDS)
        action = "create" if creates else ("edit" if edits else "other")

        for ref in refs:
            normalized = ref.replace("\\", "/").lower()
            if edits and not creates and normalized not in available:
                violations.append(
                    f"Step {index} edits '{ref}', but that file is not in the verified "
                    f"project context and no earlier step in this plan creates it."
                )
            if action != "other":
                key = (action, normalized)
                if key in seen_targets:
                    violations.append(
                        f"Step {index} duplicates an earlier step's action+target: {action} {ref}."
                    )
                seen_targets.add(key)
            if creates:
                available.add(normalized)

    return violations


def _trim_plan_violations(steps: list[str], requirements: TaskRequirements, known_files: set[str]) -> list[str]:
    """
    Deterministic last resort when a re-prompted plan still violates the
    linter: walk the steps in order and drop only the ones that introduce a
    NEW violation given everything kept so far, rather than discarding the
    whole plan. Always keeps at least one step so the run has something to
    attempt instead of failing outright on a lint technicality.
    """
    kept: list[str] = []
    for step in steps:
        candidate = kept + [step]
        if not _plan_lint_violations(candidate, requirements, known_files):
            kept.append(step)
        else:
            log("plan_lint_step_dropped", {"step": step[:160]})
    return kept or steps[:1]


def classify_plan_risk(state: RunState) -> dict:
    """
    Deterministic risk classification for a proposed plan — no LLM call, no
    guessing. Runs purely against state.requirements.files and the actual
    filesystem, so the result is reproducible and auditable.

    Returns:
        {
          "risk": "low" | "medium" | "high",
          "files_to_create": [...],   # files that do not yet exist on disk
          "files_to_modify": [...],   # files that already exist on disk
        }

    Risk heuristic (intentionally simple and explainable):
      - 0-1 target files, no framework lock  -> low
      - 2-4 target files, OR any framework lock -> medium
      - 5+ target files -> high
    A framework lock always raises a would-be "low" to "medium" because a
    single-file change can still fail structurally (wrong stack entirely),
    which the deterministic contamination checker in core/validator.py
    already treats as a hard failure, not a soft warning.
    """
    working_dir = os.environ.get("CODI_WORKING_DIR", os.getcwd())
    files = _unique(state.requirements.files or [])

    creates: list[str] = []
    modifies: list[str] = []
    for f in files:
        absolute = f if os.path.isabs(f) else os.path.join(working_dir, f)
        if os.path.exists(absolute):
            modifies.append(f)
        else:
            creates.append(f)

    file_count = len(files)
    if file_count <= 1:
        risk = "low"
    elif file_count <= 4:
        risk = "medium"
    else:
        risk = "high"

    if state.requirements.framework and risk == "low":
        risk = "medium"

    result = {"risk": risk, "files_to_create": creates, "files_to_modify": modifies}
    log("plan_risk_classified", {
        "risk": risk,
        "creates": len(creates),
        "modifies": len(modifies),
        "framework": state.requirements.framework,
    })
    return result


class Improver:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.llm      = get_refiner_llm()
        # FIX: set on every _call() invocation — None means the last call
        # either succeeded or hasn't been made yet; a string means the LLM
        # request itself failed (connection error, timeout, backend down).
        # Callers (next_step in particular) must check this BEFORE treating
        # an empty response as "nothing to do" / "task complete" — those are
        # not the same thing as "the backend never answered".
        self._last_llm_error: str | None = None

    def _orchestrator_sys(self) -> SystemMessage:
        return SystemMessage(content=_ORCHESTRATOR_SYSTEM.format(
            working_dir=os.environ.get("CODI_WORKING_DIR", os.getcwd())
        ))

    def _call(self, prompt: str, system: SystemMessage | None = None) -> str:
        sys_msg = system or self._orchestrator_sys()
        self._last_llm_error = None
        try:
            resp = self.llm.invoke([sys_msg, HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as e:
            self._last_llm_error = str(e)
            log("improver_error", {"error": str(e)})
            return ""

    # ── Phase 1: Read context ─────────────────────────────────────────────────

    def read_context(self, state: RunState) -> str:
        """
        Gather initial project context by running list_files + search_codebase.
        If boilerplate files were just created (tagged in user_input), read them
        so the Coder LLM can produce valid edit_file old/new pairs.
        """
        import re
        from context_trimmer import trim_tool_output
        dispatcher = Dispatcher(self.registry)

        tools_to_run = [
            {"name": "list_files",      "args": {}},
            {"name": "search_codebase", "args": {"query": state.user_input[:200]}},
        ]

        boilerplate_match = re.search(
            r'\[BOILERPLATE CREATED:\s*([^\]]+)\]', state.user_input
        )
        if boilerplate_match:
            files = [f.strip() for f in boilerplate_match.group(1).split(',')]
            for f in files:
                tools_to_run.append({"name": "read_file", "args": {"path": f}})

        result = dispatcher.dispatch({"action": "tool_call", "tools": tools_to_run})

        context_parts = []
        files_inspected = []
        total_context_chars = 0
        max_files = 5
        max_chars = 4000
        for r in result.get("results", []):
            tool_name = r["tool"]
            status = r["status"]
            output = r["output"]
            output_len = len(output)

            log("context_gathered_tool", {
                "tool": tool_name,
                "status": status,
                "output_len": output_len,
                "output_sample": trim_tool_output(output, max_tokens=10) if status == "ok" else output[:150],
            })

            if status == "ok":
                should_add_context = True
                if tool_name == "read_file":
                    path_value = (r.get("args") or {}).get("path")
                    if len(files_inspected) >= max_files or total_context_chars + len(output) > max_chars:
                        log("context_capped", {"path": path_value or "unknown", "reason": "read_context_limit"})
                        should_add_context = False
                    else:
                        files_inspected.append(path_value)
                elif total_context_chars + len(output) > max_chars:
                    log("context_capped", {"tool": tool_name, "reason": "read_context_limit"})
                    should_add_context = False

                if should_add_context:
                    wrapped_output = wrap_prompt_data(output, path=(r.get("args") or {}).get("path"))
                    context_parts.append(f"[{tool_name}]\n{wrapped_output}")
                    total_context_chars += len(wrapped_output)
            state.add_tool_result(tool_name, status, output)

        log("context_gathered", {
            "files_inspected": len(files_inspected),
            "total_context_chars": total_context_chars,
            "tools_run": len([r for r in result.get("results", []) if r["status"] == "ok"]),
            "files_list": files_inspected[:10],
        })

        return "\n\n".join(context_parts)

    # ── Phase 2: Create plan ──────────────────────────────────────────────────

    def _extract_requirements(self, state: RunState):
        state.requirements = _deterministic_requirements(state.user_input)
        if _is_unit_test_task(state.user_input):
            source_files = [path for path in _extract_file_refs(state.user_input) if path.lower().endswith(".py")]
            if source_files:
                source_file = source_files[0]
                project_files = state.knowledge.project.get("files", []) if state.knowledge else []
                state.requirements.protected_files = [source_file]
                state.requirements.files = [_test_file_for(source_file, project_files)]
        log("improver_requirements", {
            "source": "deterministic",
            "requirements": state.requirements.to_dict(),
        })

    def create_plan(self, state: RunState, context: str) -> dict:
        from dispatcher import Dispatcher

        self._extract_requirements(state)

        if _is_broad_every_function_request(state.user_input):
            steps = _function_by_function_test_steps()
            if steps:
                state.plan = "Generate and verify unit tests one function at a time."
                state.plan_steps = steps
                log("plan_created", {"plan_source": "function_inventory", "steps": len(steps)})
                # Deterministically generated, but still worth a lint pass —
                # e.g. a stale framework lock from a prior task in the same
                # session could still make a "create test_x.py" step invalid.
                self._lint_and_repair_plan(state, context="")
                return {"plan": state.plan, "steps": state.plan_steps}

        # Was a naive context[:6000] head-cut. context_builder now places
        # actual source code first and the cheap AST-metadata summary last,
        # specifically so the highest-value evidence survives truncation —
        # but a pure head-cut still throws away everything past 6000 chars,
        # which for a multi-file task could mean only the first read file's
        # source ever reaches the planner. Using trim_tool_output's head+tail
        # split keeps a meaningful chunk of the earliest-read source AND a
        # meaningful chunk of whatever comes later (other files' source, or
        # the metadata summary if source is short), rather than silently
        # dropping the entire second half of the collected evidence.
        from context_trimmer import trim_tool_output
        prompt = _PLAN_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            tools=", ".join(self.registry.list_names()),
            context=wrap_prompt_data(trim_tool_output(context, max_tokens=1700)),
        )

        raw = self._call(prompt, system=SystemMessage(content=planner_system_prompt()))
        state.record_llm("improver_plan_raw", raw)

        if self._last_llm_error is not None:
            # The planning LLM call itself failed (connection error, timeout,
            # backend down) — this is NOT the same as "the model decided
            # there is nothing to plan". Surface it plainly instead of
            # silently producing an empty plan that downstream code (agent.py,
            # the plan.md confirmation gate) will interpret as a legitimate
            # zero-step plan.
            state.plan = f"[PLANNING FAILED] LLM backend error: {self._last_llm_error}"
            state.plan_steps = []
            log("plan_created", {
                "plan_source": "llm_backend_error",
                "error": self._last_llm_error[:200],
            })
            return {"plan": state.plan, "steps": []}

        from context_trimmer import trim_tool_output
        plan_text, plan_steps, plan_source = self._parse_plan_response(raw, state.requirements)
        state.plan = plan_text
        state.plan_steps = plan_steps

        log("plan_created", {
            "plan_source": plan_source,
            "steps": len(state.plan_steps),
            "plan": trim_tool_output(state.plan, max_tokens=20),
            "step_samples": [trim_tool_output(s, max_tokens=15) for s in state.plan_steps[:3]],
        })

        self._lint_and_repair_plan(state, context)

        return {"plan": state.plan, "steps": state.plan_steps}

    @staticmethod
    def _parse_plan_response(raw: str, requirements: TaskRequirements) -> tuple[str, list[str], str]:
        """Shared JSON-or-line-fallback parsing used by both the initial plan
        call and the one-shot lint re-prompt, so both paths get identical
        step normalization instead of two hand-maintained copies."""
        from dispatcher import Dispatcher

        parsed = Dispatcher.parse_llm_json(raw)
        if isinstance(parsed, dict) and "steps" in parsed:
            plan_text = _as_text(parsed.get("plan", ""))
            raw_steps = parsed.get("steps", [])
            if isinstance(raw_steps, list):
                steps = _enforce_test_intent(_normalize_plan_steps(
                    [_as_text(s) for s in raw_steps if _as_text(s)], requirements
                ), requirements)
            else:
                s = _as_text(raw_steps)
                steps = _enforce_test_intent(_normalize_plan_steps([s] if s else [], requirements), requirements)
            return plan_text, steps, "json"

        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        collected = []
        for line in lines:
            line = line.lstrip("```").strip()
            if line.lower().startswith(("json", "{")):
                continue
            line = line.lstrip("0123456789.-) ").strip()
            if line:
                collected.append(line)
        plan_text = raw[:200]
        steps = _enforce_test_intent(_normalize_plan_steps(collected[:5], requirements), requirements)
        return plan_text, steps, "fallback"

    def _lint_and_repair_plan(self, state: RunState, context: str) -> None:
        """
        Deterministic contradiction check, run after every plan parse.
        On violation: re-prompt the planner ONCE with the specific
        violations listed as hard constraints. If the retry still violates
        the linter, fall back to a deterministic trim (drop only the steps
        that introduce a new violation) rather than looping indefinitely or
        silently executing a self-contradictory plan.
        """
        from context_trimmer import trim_tool_output

        known_files = _known_files_for_lint(state)
        violations = _plan_lint_violations(state.plan_steps, state.requirements, known_files)
        if not violations:
            return

        log("plan_lint_violation", {
            "attempt": 1,
            "violations": violations[:10],
            "steps": len(state.plan_steps),
        })

        retry_prompt = _PLAN_PROMPT.format(
            task=(
                f"{state.user_input}\n\n"
                "PREVIOUS PLAN REJECTED — it violated these constraints, fix them:\n"
                + "\n".join(f"- {v}" for v in violations)
            ),
            requirements=state.requirements.as_prompt_block(),
            tools=", ".join(self.registry.list_names()),
            context=wrap_prompt_data(trim_tool_output(context, max_tokens=1700)),
        )
        raw_retry = self._call(retry_prompt, system=SystemMessage(content=planner_system_prompt()))
        state.record_llm("improver_plan_lint_retry", raw_retry)

        if self._last_llm_error is None and raw_retry:
            plan_text, plan_steps, plan_source = self._parse_plan_response(raw_retry, state.requirements)
            retry_violations = _plan_lint_violations(plan_steps, state.requirements, known_files)
            log("plan_lint_retry_result", {
                "plan_source": plan_source,
                "steps": len(plan_steps),
                "remaining_violations": retry_violations[:10],
            })
            if not retry_violations:
                state.plan = plan_text
                state.plan_steps = plan_steps
                return
            # Retry still violates — keep whichever candidate has fewer
            # violations as the base for deterministic trimming below.
            if len(retry_violations) < len(violations):
                state.plan = plan_text
                state.plan_steps = plan_steps
                violations = retry_violations

        trimmed = _trim_plan_violations(state.plan_steps, state.requirements, known_files)
        log("plan_lint_trimmed", {
            "original_steps": len(state.plan_steps),
            "trimmed_steps": len(trimmed),
        })
        state.plan_steps = trimmed

    # ── Phase 3: Decide next step ─────────────────────────────────────────────

    def next_step(self, state: RunState) -> dict:
        """Return {"step": str, "done": bool} for the current iteration.

        FIX: also returns "llm_error": str when the underlying LLM call
        itself failed. agent.py MUST check this before treating an empty
        step + done=False as task completion — previously a connection
        error to the LLM backend produced an empty step string, which the
        `if done or not step:` check in agent.py silently interpreted as
        "planner says the task is complete", reporting success on a run
        that never actually did anything.
        """
        from dispatcher import Dispatcher
        from context_trimmer import trim_tool_output
        done_count = len(state.completed_steps)

        if state.validation_repair_instruction:
            repair = state.validation_repair_instruction
            state.validation_repair_instruction = ""
            # target_plan_step is deliberately left untouched here — a
            # validator repair is always in service of whichever plan step
            # was already selected in a prior iteration (that's what got
            # validated and failed). Overwriting it with the repair text
            # would reproduce the exact bug this field exists to prevent.
            log("step_selected", {
                "step": trim_tool_output(repair, max_tokens=30),
                "source": "validator_repair",
                "iteration": state.iteration,
                "done": False,
            })
            return {"step": repair, "done": False}

        if state.plan_steps and "[CORRECTION]" not in (state.plan or ""):
            selected_step = next(
                (step for step in state.plan_steps if step not in state.completed_steps),
                None,
            )
            if selected_step:
                # If the immediately preceding tool result was an error and
                # this is the same plan step we already tried (attempts > 0),
                # don't blindly hand the coder the identical instruction
                # again — that's how a blocked write (e.g. "refusing to
                # overwrite") turns into N identical failed retries burning
                # the iteration budget with nothing learned in between.
                # Instead, reason about the failure via improve() and hand
                # back a concrete correction step grounded in the actual
                # error AND in which strategies have already been tried and
                # failed for this exact step (see improve()'s
                # tried_strategies parameter — FIX bug 1), then retry with
                # that reasoning applied.
                last_error = next(
                    (r for r in reversed(state.tool_results) if r.status == "error"),
                    None,
                )
                prior_attempts = state.step_attempts.get(selected_step, 0)

                if last_error and prior_attempts > 0:
                    state.validation_notes = last_error.output
                    state.target_plan_step = selected_step
                    tried = state.tried_strategies(selected_step)
                    correction = self.improve(state, tried_strategies=tried)
                    if self._last_llm_error is not None:
                        return {"step": "", "done": False, "llm_error": self._last_llm_error}
                    state.step_attempts[selected_step] = prior_attempts + 1
                    log("step_selected", {
                        "step": trim_tool_output(correction, max_tokens=20),
                        "matched_plan": False,
                        "iteration": state.iteration,
                        "done": False,
                        "plan_steps_remaining": max(0, len(state.plan_steps) - done_count),
                        "source": "repeated_step_failure_reasoned_correction",
                        "original_step": trim_tool_output(selected_step, max_tokens=20),
                        "prior_attempts": prior_attempts,
                        "tried_strategies": tried,
                    })
                    return {"step": correction, "done": False}

                state.step_attempts[selected_step] = prior_attempts + 1
                state.target_plan_step = selected_step
                log("step_selected", {
                    "step": trim_tool_output(selected_step, max_tokens=20),
                    "matched_plan": True,
                    "iteration": state.iteration,
                    "done": False,
                    "plan_steps_remaining": max(0, len(state.plan_steps) - done_count),
                    "source": "first_incomplete_plan_step",
                })
                return {"step": selected_step, "done": False}

        prompt = _NEXT_STEP_PROMPT.format(
            task=state.user_input,
            requirements=state.requirements.as_prompt_block(),
            plan=state.plan,
            plan_steps="\n".join(state.plan_steps) if state.plan_steps else "(no steps)",
            done_steps=f"{done_count} of {len(state.plan_steps)}",
            tool_results=wrap_prompt_data("\n".join(state.recent_tool_outputs(5))),
        )
        raw = self._call(prompt)
        state.record_llm("improver_next_step", raw)

        if self._last_llm_error is not None:
            log("step_selected", {
                "iteration": state.iteration,
                "source": "llm_backend_error",
                "error": self._last_llm_error[:200],
            })
            return {"step": "", "done": False, "llm_error": self._last_llm_error}

        parsed = Dispatcher.parse_llm_json(raw)
        selected_step = None
        matched_plan = False
        done = False

        if isinstance(parsed, dict):
            selected_step = _as_text(parsed.get("step", ""))
            done = bool(parsed.get("done", False))
        elif parsed:
            selected_step = _as_text(parsed)
        else:
            selected_step = _as_text(raw)[:300]

        if selected_step and state.plan_steps:
            matched_plan = any(selected_step.strip() == ps.strip() for ps in state.plan_steps)
            if matched_plan:
                state.target_plan_step = selected_step

        log("step_selected", {
            "step": trim_tool_output(selected_step, max_tokens=20),
            "matched_plan": matched_plan,
            "iteration": state.iteration,
            "done": done,
            "plan_steps_remaining": max(0, len([s for s in state.plan_steps if s]) - done_count),
        })

        return {"step": selected_step, "done": done}

    # ── Phase 4: Generate correction after validation failure ─────────────────

    def improve(self, state: RunState, tried_strategies: list[str] | None = None) -> str:
        from dispatcher import Dispatcher
        notes = state.validation_notes or "unknown failure"
        reqs  = state.requirements
        forbidden = reqs.framework_lock()

        lines = [
            f"PREVIOUS ATTEMPT FAILED. Reason: {notes}",
            "",
            "REQUIREMENTS (non-negotiable):",
            reqs.as_prompt_block(),
            "",
        ]

        if forbidden and any(p.lower() in notes.lower() for p in forbidden):
            lines += [
                "CRITICAL: You used a forbidden framework.",
                "You MUST:",
                f"  1. DELETE every line containing: {', '.join(forbidden)}",
                f"  2. Rewrite the ENTIRE file using {reqs.framework} ONLY.",
                "  3. Do NOT import anything from the forbidden frameworks.",
                f"Any output still containing forbidden imports is automatic failure.",
            ]

        if "does not exist" in notes.lower():
            lines += [
                "CRITICAL: The referenced file does not exist. Do NOT keep",
                "retrying the same filename. Either it was typo'd — check the",
                "codebase context for the exact real filename — or it genuinely",
                "needs to be created first with write_file/create_file.",
            ]

        if "refusing to overwrite" in notes.lower():
            lines += [
                "CRITICAL: The target file already exists and create_file was blocked",
                "to protect it from accidental clobbering. Two valid fixes:",
                "  1. If the user's task genuinely called for a full replacement",
                "     (e.g. said 'override'/'overwrite'/'replace all'/'start fresh'),",
                "     switch to edit_file: read the existing file's exact current",
                "     content with read_file, then edit_file with old=<that exact",
                "     content> and new=<the full new content>.",
                "  2. If a full rewrite was NOT actually intended, use edit_file to",
                "     make the specific surgical change instead of a full replace.",
                "Do NOT call create_file on this path again — it will be blocked",
                "identically every time. Name the exact tool (edit_file) and path",
                "in your correction.",
            ]

        if "protected" in notes.lower() and "cannot be changed" in notes.lower():
            lines += [
                "CRITICAL: This file is protected and cannot be modified for this",
                "task. Re-read the actual task — it likely wants a different file",
                "created or edited (e.g. a companion test file), not this one.",
            ]

        # FIX (bug 1): explicit, structured "do not repeat" block built from
        # everything the executor has already tried and failed for this
        # exact step. Previously the correction prompt only had the raw
        # error text — nothing told the model it had already tried (and
        # exhausted) a specific approach, so it would frequently regenerate
        # something functionally identical to the failed attempt. This is
        # the LLM-assisted half of the fix; the deterministic half lives in
        # core/executor.py's _execute_edit_first, which forces a strategy
        # switch outright for the most common failure mode (text-match edit)
        # without waiting on the model to obey this instruction.
        if tried_strategies:
            described = [
                f"  - {_STRATEGY_DESCRIPTIONS.get(name, name)}"
                for name in tried_strategies
            ]
            lines += [
                "",
                "STRATEGIES ALREADY TRIED AND FAILED FOR THIS STEP — do NOT repeat any of these:",
                *described,
                "Your correction MUST use a genuinely different approach than every strategy",
                "listed above. If text-match replacement failed, ask for a line-range edit",
                "instead. If a line-range edit failed, re-read the file for fresh line numbers",
                "or use apply_patch. If content-first generation failed, break the change into",
                "a smaller, targeted edit_file operation instead of regenerating the whole file.",
            ]

        lines += [
            "",
            "Last tool outputs for context:",
            *state.recent_tool_outputs(3),
            "",
            "Think step by step first, silently: what is the ROOT CAUSE of this",
            "failure (not just its symptom), and why would repeating the exact",
            "same action fail again the same way? Then decide the concrete fix.",
            "",
            "Respond ONLY with JSON in this exact shape:",
            '{"reasoning":"one sentence root-cause diagnosis","correction":"exact fix instruction naming the tool and path"}',
            "JSON only:",
        ]

        raw = self._call(
            "\n".join(lines),
            system=SystemMessage(content=correction_system_prompt())
        )
        state.record_llm("improver_improve", raw)

        if self._last_llm_error is not None:
            log("improver_correction", {"source": "llm_backend_error", "error": self._last_llm_error[:200]})
            return ""

        parsed = Dispatcher.parse_llm_json(raw)
        reasoning = ""
        if isinstance(parsed, dict):
            correction = _as_text(parsed.get("correction", raw))
            reasoning = _as_text(parsed.get("reasoning", ""))
        elif parsed:
            correction = _as_text(parsed)
        else:
            correction = f"FAILED: {notes}. Fix required: {reqs.as_prompt_block()}"

        log("improver_correction", {
            "correction": correction[:200],
            "reasoning": reasoning[:200],
            "tried_strategies": tried_strategies or [],
        })
        return correction

    # ── Phase 5: Final summary ────────────────────────────────────────────────

    def summarize(self, state: RunState) -> str:
        """Produce the final user-facing output without an LLM round trip."""
        if not state.tool_results:
            if getattr(state, "llm_backend_errors", 0) > 0:
                return (
                    "Stopped — the LLM backend was unreachable and no tool "
                    "executions could be attempted. Check that your configured "
                    "backend (llama.cpp / Ollama / cloud provider) is running "
                    "and reachable, then retry."
                )
            return "Task completed with no tool executions."

        files = _files_from_tool_results(state)
        successes = len([r for r in state.tool_results if r.status == "ok"])
        failures = len([r for r in state.tool_results if r.status == "error"])

        stopped_at_limit = state.exceeds_max() and not state.validation_passed
        if stopped_at_limit or state.status == "failed":
            output = "Stopped before completion."
            if stopped_at_limit:
                output += " Reached the maximum number of iterations."
            if state.validation_notes:
                output += f" Last validation issue: {state.validation_notes}"
        elif files:
            output = "Done. Changed: " + ", ".join(files[:8]) + "."
        elif successes:
            output = f"Done. Completed {successes} tool action(s)."
        else:
            output = "Nothing was successfully changed."

        if failures:
            output += f" {failures} tool action(s) reported errors."

        log("improver_summary", {"source": "deterministic", "output": output[:200]})
        return output