from __future__ import annotations

import json
from types import SimpleNamespace

from core.improver import _static_ecommerce_steps, classify_plan_risk
from core.validator import Validator
from state.temp_db import RunState


class _StaticReviewLLM:
    def __init__(self, payload: dict):
        self.payload = payload

    def invoke(self, _messages):
        return SimpleNamespace(content=json.dumps(self.payload))


def _review(tmp_path, *, score: float, risk: str, confidence: float) -> dict:
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Plan\n\n1. Create index.html\n", encoding="utf-8")
    validator = Validator()
    validator._get_llm = lambda: _StaticReviewLLM({
        "score": score,
        "notes": "The plan is concrete.",
        "suggested_edits": "",
    })
    return validator.validate_plan_file(RunState(user_input="Build a landing page"), str(plan_path), risk, confidence)


def test_low_risk_medium_score_plan_requires_explicit_user_review(tmp_path):
    review = _review(tmp_path, score=8.5, risk="low", confidence=0.55)

    assert review["approved"] is False
    assert review["requires_user_review"] is True
    assert review["policy"] == "medium_score_low_risk_requires_user_review"


def test_medium_risk_medium_score_plan_requires_user_review_not_three_retries(tmp_path):
    review = _review(tmp_path, score=8.5, risk="medium", confidence=0.75)

    assert review["approved"] is False
    assert review["requires_user_review"] is True
    assert review["policy"] == "medium_score_medium_risk_requires_user_review"


def test_low_scoring_plan_still_requires_revision(tmp_path):
    review = _review(tmp_path, score=6.5, risk="low", confidence=1.0)

    assert review["approved"] is False


def test_ecommerce_plan_names_the_required_architecture_and_file_creates(tmp_path, monkeypatch):
    monkeypatch.setenv("CODI_WORKING_DIR", str(tmp_path))
    steps = _static_ecommerce_steps([])
    state = RunState(user_input="Build an ecommerce website", plan_steps=steps)

    risk = classify_plan_risk(state)

    joined = "\n".join(steps).lower()
    for capability in ("navigation", "catalog", "cart", "checkout", "filter", "responsive"):
        assert capability in joined
    assert set(risk["files_to_create"]) == {"index.html", "styles.css", "products.json", "script.js"}
    assert all(step.startswith("Create") for step in steps)
