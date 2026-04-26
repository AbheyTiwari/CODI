# TODO: Instant Boilerplate + Edit Feature

## Goal
When user says "create portfolio website in html css", CODI should:
1. **Instantly** create boilerplate files (user sees files in < 1 second)
2. **Then** run agent loop to edit/enrich those files with content

## Steps
- [x] Step 1: Modify `main.py` — detect creation tasks, instantly create boilerplate files BEFORE agent loop with visible terminal feedback
- [x] Step 2: Modify `core/improver.py` — update planning to prefer EDIT steps when files already exist
- [x] Step 3: Clean up `agent.py` — remove dead code, keep agent focused on execution loop
- [x] Step 4: Test the flow

## How It Works

**Before (slow):**
```
User: "create portfolio website in html css"
→ Agent loop starts (invisible)
→ LLM Call 1: plan create index.html      (30-60s, nothing on screen)
→ LLM Call 2: plan edit index.html        (30-60s)
→ LLM Call 3: plan create styles.css      (30-60s)
→ ... 5-10 minutes total
```

**After (fast):**
```
User: "create portfolio website in html css"
→ INSTANTLY creates index.html + styles.css  (< 1s)
→ Terminal shows:
    ✓ created index.html
    ✓ created styles.css
    → enriching with content...
→ Files open in VS Code immediately
→ Agent loop starts with EDIT-focused plan
→ LLM Call 1: edit index.html with portfolio (30-60s)
→ LLM Call 2: edit styles.css with styling   (30-60s)
→ ... 2-4 minutes total, files visible from second 1
```

## Files Modified
1. **`main.py`** — Added `_instant_boilerplate_create()` that runs BEFORE agent loop
2. **`core/improver.py`** — Updated `_PLAN_PROMPT` to recognize boilerplate and plan EDIT steps
3. **`agent.py`** — Cleaned up, removed dead code

## Key Design Decision
The boilerplate creation happens in `main.py` (not inside the agent) so the user gets **immediate terminal feedback** even when the agent loop is slow in local mode. The existing templates from `core/quick_actions.py` are reused — no duplication.
