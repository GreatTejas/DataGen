"""Data contracts for the Clinical Knowledge (RAG) + Clinical Reasoning component (Person 4).

`EventEnvelope` is copied field-for-field from
`clinical_data_gen.schema.EventEnvelope` (Person 1) so anything this
component publishes stays wire-compatible with the rest of the pipeline
without importing across team boundaries.

`ConsolidatedEvidence` is this project's PROPOSED contract for what
Person 3's Evidence Fusion stage will hand to the Clinical Reasoning
Agent. It does not exist upstream yet (Evidence Fusion is not built),
so this schema is our working assumption, built from:
  - the report's description of Evidence Fusion (Sec. 3.4): combine
    trend/anomaly/risk outputs + patient context, handle conflicting
    or missing agent outputs, produce a consolidated summary.
  - the real fields Person 1's generator actually emits in
    patient_context.jsonl (age, sex, relevant_history,
    current_medications, recent_labs) and vitals_observed.jsonl
    (heart_rate, spo2, respiratory_rate, systolic_bp, diastolic_bp,
    map, fio2).

When Person 3 ships the real Evidence Fusion output, only
`adapters.py::evidence_from_fusion_output` needs to change — nothing
in `knowledge_store.py` or `reasoning_agent.py` should need to.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


SCHEMA_VERSION = "0.1.0-p4-draft"


@dataclass(frozen=True)
class EventEnvelope:
    """Identical shape to clinical_data_gen.schema.EventEnvelope."""

    schema_version: str
    event_id: str
    event_time: str
    patient_id: str
    run_id: str
    source: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrendSummary:
    """Stand-in for Person 1's Trend Analyzer output."""

    trend_score: float
    persistent_deterioration: bool
    per_vital_slope: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class AnomalySummary:
    """Stand-in for Person 2's Anomaly Agent output."""

    anomaly_confidence: float
    flagged_parameters: list[str] = field(default_factory=list)


@dataclass
class RiskSummary:
    """Stand-in for Person 3's Risk Assessor output."""

    risk_score: float
    risk_category: str  # "low" | "medium" | "high"
    per_vital_risk: dict[str, str] = field(default_factory=dict)


@dataclass
class PatientContext:
    """Mirrors the payload of Person 1's patient.context stream exactly."""

    age: Optional[int] = None
    sex: Optional[str] = None
    relevant_history: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    recent_labs: dict[str, float] = field(default_factory=dict)


@dataclass
class ConsolidatedEvidence:
    """PROPOSED contract: Evidence Fusion (Person 3) -> Clinical Reasoning (Person 4).

    Not yet confirmed against Person 3's real implementation — see
    module docstring. `evidence_id` should be stable and idempotent
    per (patient_id, window_end) so re-running reasoning on the same
    evidence is safe and auditable.
    """

    schema_version: str
    evidence_id: str
    patient_id: str
    window_start: str
    window_end: str
    current_vitals: dict[str, float]
    trend: TrendSummary
    anomaly: AnomalySummary
    risk: RiskSummary
    patient_context: PatientContext
    conflicts_or_missing: list[str] = field(default_factory=list)
    fusion_confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "patient_id": self.patient_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "current_vitals": self.current_vitals,
            "trend": asdict(self.trend),
            "anomaly": asdict(self.anomaly),
            "risk": asdict(self.risk),
            "patient_context": asdict(self.patient_context),
            "conflicts_or_missing": self.conflicts_or_missing,
            "fusion_confidence": self.fusion_confidence,
        }


@dataclass
class KnowledgeDocument:
    """Mirrors the fields actually present in knowledge_corpus.jsonl."""

    document_id: str
    title: str
    source_class: str
    version: str
    source_url: str
    locator: str
    applicability: str
    text: str


@dataclass
class RetrievedChunk:
    document: KnowledgeDocument
    chunk_text: str
    chunk_index: int
    score: float


@dataclass
class SimilarCase:
    """PROPOSED contract: Experience Memory (Person 2/3) -> Person 4.

    Based on report Sec. 4.3: each stored case has a physiological
    pattern, anomaly confidence, detected trends, the clinician's
    decision, and their stated feedback/reason.
    """

    case_id: str
    similarity: float
    physiological_pattern: dict[str, Any]
    anomaly_confidence: float
    detected_trends: list[str]
    clinician_decision: str
    clinician_feedback: str
    experience_confidence: float


@dataclass
class Citation:
    document_id: str
    version: str
    locator: str
    source_url: str


@dataclass
class ReasoningOutput:
    """What this component hands to Person 5 (Alert Policy + Audit Trail)."""

    schema_version: str
    patient_id: str
    evidence_id: str
    generated_at: str
    explanation: str
    contributing_factors: list[str]
    recommended_action: str
    citations: list[Citation]
    similar_cases_referenced: list[str]
    confidence: float
    model: str
    tool_call_count: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d
