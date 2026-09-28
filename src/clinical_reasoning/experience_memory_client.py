"""Client interface to the Experience Memory store (owned by Person 2/3).

Per the team's division of work, Person 2 owns "Experience-memory
creation/update" and Person 3 owns "Experience Memory - retrieval
side" (similar-case retrieval, experience confidence). The Clinical
Reasoning Agent (Person 4) is the consumer: it calls
`retrieve_similar_cases` as a tool during reasoning.

Until that store exists, `StubExperienceMemoryClient` returns
hand-built cases from an in-memory list so the reasoning agent's tool
loop can be developed and tested end to end today. Swap in a real
client that talks to Person 2/3's store by implementing the same
`ExperienceMemoryClient` interface — nothing in `reasoning_agent.py`
needs to change.
"""
from __future__ import annotations

from typing import Protocol

from .schema import SimilarCase


class ExperienceMemoryClient(Protocol):
    def retrieve_similar_cases(
        self, physiological_pattern: dict, top_k: int = 3
    ) -> list[SimilarCase]: ...


class StubExperienceMemoryClient:
    """Standalone stand-in. Replace with the real Person 2/3 client."""

    def __init__(self, cases: list[SimilarCase] | None = None) -> None:
        self._cases = cases if cases is not None else _default_stub_cases()

    def retrieve_similar_cases(
        self, physiological_pattern: dict, top_k: int = 3
    ) -> list[SimilarCase]:
        # No real similarity search here — this is a placeholder that
        # returns the stub cases most likely to be topically relevant
        # by a crude overlap heuristic, purely so the reasoning agent
        # has something non-trivial to reason over during dev/testing.
        def overlap(case: SimilarCase) -> int:
            keys_a = set(physiological_pattern.keys())
            keys_b = set(case.physiological_pattern.keys())
            return len(keys_a & keys_b)

        ranked = sorted(self._cases, key=overlap, reverse=True)
        return ranked[:top_k]


def _default_stub_cases() -> list[SimilarCase]:
    return [
        SimilarCase(
            case_id="STUB-CASE-0001",
            similarity=0.0,
            physiological_pattern={
                "heart_rate": "rising",
                "systolic_bp": "falling",
            },
            anomaly_confidence=0.81,
            detected_trends=["falling blood pressure", "rising heart rate"],
            clinician_decision="accept",
            clinician_feedback=(
                "Confirmed early haemodynamic instability; started fluid "
                "bolus and escalated to senior review."
            ),
            experience_confidence=0.7,
        ),
        SimilarCase(
            case_id="STUB-CASE-0002",
            similarity=0.0,
            physiological_pattern={
                "spo2": "falling",
                "respiratory_rate": "rising",
            },
            anomaly_confidence=0.74,
            detected_trends=["falling SpO2", "rising respiratory rate"],
            clinician_decision="accept",
            clinician_feedback="Genuine respiratory decline, not an artifact.",
            experience_confidence=0.65,
        ),
        SimilarCase(
            case_id="STUB-CASE-0003",
            similarity=0.0,
            physiological_pattern={"heart_rate": "brief_spike"},
            anomaly_confidence=0.55,
            detected_trends=["isolated single-reading spike"],
            clinician_decision="dismiss",
            clinician_feedback="Patient was ambulating; artifact, not real.",
            experience_confidence=0.4,
        ),
    ]
