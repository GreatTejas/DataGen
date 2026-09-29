"""Explicit handoff for the pending Person 4 RAG and reasoning work."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClinicalReasoningResult:
    status: str
    message: str
    explanation: str | None
    recommendation: str | None
    retrieved_evidence: list[dict[str, Any]]
    citations: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "retrieved_evidence": list(self.retrieved_evidence),
            "citations": list(self.citations),
        }


class ClinicalReasoningPlaceholder:
    """Return an honest pending status until Person 4 supplies RAG/reasoning."""

    def run(self, evidence: Any) -> ClinicalReasoningResult:
        del evidence
        return ClinicalReasoningResult(
            status="not_implemented",
            message="Person 4's clinical knowledge retrieval and reasoning are pending.",
            explanation=None,
            recommendation=None,
            retrieved_evidence=[],
            citations=[],
        )
