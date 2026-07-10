# football_theme.py
# ─────────────────────────────────────────────────────────────────────────────
# Purely cosmetic, purely hardcoded football theming for the CLI.
# No LLM calls, no network — just keyword lookups and a frame counter, so
# this never costs tokens, latency, or a request.
# ─────────────────────────────────────────────────────────────────────────────

import itertools

# Simple bouncing-ball animation frames for the live status panel title.
BALL_FRAMES = [
    "⚽      ",
    " ⚽     ",
    "  ⚽    ",
    "   ⚽   ",
    "    ⚽  ",
    "     ⚽ ",
    "    ⚽  ",
    "   ⚽   ",
    "  ⚽    ",
    " ⚽     ",
]

_frame_cycle = itertools.cycle(BALL_FRAMES)


def next_frame() -> str:
    """Advance and return the next bouncing-ball animation frame."""
    return next(_frame_cycle)


# Ordered (keyword, pun) pairs — first match against the lowercased status
# text wins. Keep these football-specific and short; they're appended to
# real status lines, not a replacement for them.
_PUN_RULES = [
    ("plan ready",       "Tactics board is set"),
    ("plan",              "Setting up the formation"),
    ("reading project",   "Studying the match footage"),
    ("context",           "Scouting the pitch before kickoff"),
    ("read",              "Reading the play"),
    ("search",            "Making a searching run"),
    ("validat",           "VAR check in progress"),
    ("correction",        "Regrouping at half-time"),
    ("repair",            "Physio's on the pitch"),
    ("fix",               "Physio's on the pitch"),
    ("fail",              "That one hit the crossbar"),
    ("error",             "Yellow card — going again"),
    ("permission",        "Checking with the referee"),
    ("external shell",    "Bringing on a substitute"),
    ("max iteration",     "Full time — final whistle"),
    ("complete",          "GOAL! Back of the net"),
    ("done",              "GOAL! Back of the net"),
    ("navigat",           "Making a run down the wing"),
    ("browser",           "Taking it wide down the flank"),
    ("command",           "Taking a set piece"),
    ("edit",              "A quick one-touch pass"),
    ("write",             "Striking it clean into the net"),
    ("create",            "Building up from the back"),
    ("delete",            "Clean tackle — that's gone"),
    ("remove",            "Clean tackle — that's gone"),
    ("step",              "Passing it forward"),
]

_FALLBACK_PUNS = [
    "Keeping possession",
    "Playing it out from the back",
    "Finding space between the lines",
    "Working the channel",
    "Tracking back to help out",
    "Composed on the ball",
]
_fallback_cycle = itertools.cycle(_FALLBACK_PUNS)


def pun_for(text: str) -> str:
    """Return a hardcoded football pun matching keywords found in `text`."""
    lowered = (text or "").lower()
    for keyword, pun in _PUN_RULES:
        if keyword in lowered:
            return pun
    return next(_fallback_cycle)


def themed_status_line(text: str) -> str:
    """Decorate a status line with a matching football pun.

    e.g. 'Creating an execution plan.'
      -> 'Creating an execution plan.  —  Setting up the formation'
    """
    text = (text or "").strip()
    if not text:
        return text
    return f"{text}  —  {pun_for(text)}"