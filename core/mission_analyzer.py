from __future__ import annotations

import json
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


MISSION_PROMPT = """
You are CODI's Mission Analyzer.

You NEVER write code.

You NEVER make a plan.

You NEVER call tools.

Your only job is to understand the task.

Think like a senior engineer.

Determine:

1. What is the user's real goal?

2. What assumptions are being made?

3. What information is missing?

4. Which files are probably required?

5. Which functions/classes are probably relevant?

6. Can planning safely begin?

Output ONLY valid JSON.

Schema:

{
    "goal":"",

    "confidence":0.0,

    "requires_context":true,

    "assumptions":[],

    "unknowns":[],

    "files_needed":[],

    "symbols_needed":[],

    "summary":""
}

Confidence:

0.0 = know nothing

0.5 = partial understanding

0.8 = almost enough

0.95 = enough to plan

1.0 = complete understanding

If confidence is below 0.9,
requires_context MUST be true.
"""


class MissionAnalyzer:

    def __init__(self):

        self.llm = get_refiner_llm()

    def analyze(self, user_input: str) -> MissionAnalysis:

        messages = [

            SystemMessage(content=MISSION_PROMPT),

            HumanMessage(content=user_input)

        ]

        try:

            response = self.llm.invoke(messages)

            content = response.content.strip()

            start = content.find("{")
            end = content.rfind("}")

            if start != -1 and end != -1:
                content = content[start:end + 1]

            data = json.loads(content)

        except Exception as e:

            log("mission_parse_error", {

                "error": str(e)

            })

            return MissionAnalysis(

                goal=user_input,

                confidence=0.2,

                requires_context=True,

                summary="Mission parsing failed."

            )

        analysis = MissionAnalysis(

            goal=data.get("goal", user_input),

            confidence=float(data.get("confidence", 0.0)),

            requires_context=bool(data.get("requires_context", True)),

            assumptions=data.get("assumptions", []),

            unknowns=data.get("unknowns", []),

            files_needed=data.get("files_needed", []),

            symbols_needed=data.get("symbols_needed", []),

            summary=data.get("summary", "")

        )

        log(

            "mission_analysis",

            {

                "goal": analysis.goal,

                "confidence": analysis.confidence,

                "files": analysis.files_needed,

                "unknowns": analysis.unknowns

            }

        )

        return analysis