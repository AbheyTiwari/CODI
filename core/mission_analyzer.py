from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_refiner_llm
from logger import log


@dataclass
class MissionAnalysis:
    goal: str = ""
    confidence: float = 0.0
    requires_context: bool = True
    assumptions: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    files_needed: List[str] = field(default_factory=list)
    symbols_needed: List[str] = field(default_factory=list)
    summary: str = ""


# Kept dense and JSON-first on purpose. The earlier version put a blank line
# between every numbered question and every schema field. Small quantized
# models (this project runs qwen2.5-coder-7b via llama.cpp) pattern-match
# formatting, not just semantics — the same failure mode already documented
# for the executor_system_prompt "relative/path" placeholder-copying bug.
# Loose, prose-like spacing in a JSON example measurably increases the odds
# the model echoes that same loose spacing back with non-JSON content mixed
# in. Every field is now packed on one line with no room to improvise.
MISSION_PROMPT = """You are CODI's Mission Analyzer. You never write code, never make a plan, never call tools. Your only job is to understand the task and output structured JSON describing it.
Determine: (1) the user's real goal, (2) assumptions being made, (3) missing information, (4) which files are probably required, (5) which functions/classes are probably relevant, (6) whether planning can safely begin.
Output ONLY valid JSON, nothing else — no prose before or after, no markdown fences:
{"goal":"","confidence":0.0,"requires_context":true,"assumptions":[],"unknowns":[],"files_needed":[],"symbols_needed":[],"summary":""}
Confidence scale: 0.0=know nothing, 0.5=partial understanding, 0.8=almost enough, 0.95=enough to plan, 1.0=complete understanding. If confidence is below 0.9, requires_context MUST be true.
If the task describes a NEW capability (e.g. "let users upload PDFs and chat over them") that has no existing file to point to, files_needed should list the file(s) you'd expect this to LIVE IN even if they don't exist yet (e.g. "ingest.py", "document_parser.py") — do not leave files_needed empty just because nothing matching exists yet.
JSON only:"""


# Capability keywords used ONLY as a deterministic fallback when the LLM's
# JSON response fails to parse. This mirrors the same fallback pattern
# already used in core/improver.py's _deterministic_requirements — a failed
# LLM call should degrade to a keyword-based guess, not to an empty mission.
_CAPABILITY_HINTS: dict[str, list[str]] = {
    "pdf":       ["pdf", "PyPDF2", "pypdf", "document parsing"],
    "docx":      ["docx", "word document", "python-docx"],
    "xlsx":      ["xlsx", "excel", "spreadsheet", "openpyxl"],
    "upload":    ["file upload", "upload endpoint"],
    "chat":      ["chatbot", "chat interface", "retrieval"],
    "embed":     ["embedding", "vector store"],
    "auth":      ["authentication", "login"],
}

_UNKNOWN_WORD_RE = re.compile(r"[a-z]{3,}")


def _deterministic_fallback(user_input: str, reason: str) -> MissionAnalysis:
    """
    Used only when the LLM's mission JSON fails to parse. Never returns a
    blank mission — extracts whatever capability keywords are literally
    present in the user's own words, so a single bad completion doesn't
    propagate an empty mission through ContextBuilder and Improver.
    """
    lowered = (user_input or "").lower()
    assumptions: list[str] = []
    for keyword, hints in _CAPABILITY_HINTS.items():
        if keyword in lowered:
            assumptions.append(f"Task likely involves: {', '.join(hints)}")

    return MissionAnalysis(
        goal=user_input,
        confidence=0.2,
        requires_context=True,
        assumptions=assumptions,
        unknowns=[f"Mission JSON parse failed ({reason}); using keyword fallback."],
        files_needed=[],
        symbols_needed=[],
        summary="Mission parsing failed — deterministic keyword fallback used instead of a blank mission.",
    )


class MissionAnalyzer:
    def __init__(self):
        self.llm = get_refiner_llm()

    def analyze(self, user_input: str) -> MissionAnalysis:
        messages = [
            SystemMessage(content=MISSION_PROMPT),
            HumanMessage(content=user_input),
        ]

        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()

            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"no JSON object found in response: {content[:200]!r}")

            data = json.loads(content[start:end + 1])

        except Exception as e:
            log("mission_parse_error", {"error": str(e), "raw_sample": (locals().get("content") or "")[:200]})
            fallback = _deterministic_fallback(user_input, str(e)[:120])
            log("mission_analysis", {
                "goal": fallback.goal[:160],
                "confidence": fallback.confidence,
                "files": fallback.files_needed,
                "unknowns": fallback.unknowns,
                "source": "deterministic_fallback",
            })
            return fallback

        analysis = MissionAnalysis(
            goal=data.get("goal", user_input),
            confidence=float(data.get("confidence", 0.0)),
            requires_context=bool(data.get("requires_context", True)),
            assumptions=data.get("assumptions", []),
            unknowns=data.get("unknowns", []),
            files_needed=data.get("files_needed", []),
            symbols_needed=data.get("symbols_needed", []),
            summary=data.get("summary", ""),
        )

        log("mission_analysis", {
            "goal": analysis.goal[:160],
            "confidence": analysis.confidence,
            "files": analysis.files_needed,
            "unknowns": analysis.unknowns,
            "source": "llm_json",
        })

        return analysis