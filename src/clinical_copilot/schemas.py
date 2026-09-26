"""Data models and schemas for Clinical Copilot."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AlertDecision(str, Enum):
    ALERT = "ALERT"
    DEFER = "DEFER"
    SUPPRESS = "SUPPRESS"


class AlertPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ClinicianActionType(str, Enum):
    ACCEPT = "ACCEPT"
    DISMISS = "DISMISS"
    DEFER = "DEFER"
    INVESTIGATE = "INVESTIGATE"


@dataclass
class VitalReading:
    heart_rate: float
    spo2: float
    respiratory_rate: float
    systolic_bp: float
    diastolic_bp: float
    map: float
    fio2: float = 0.21
    quality: str = "normal"
    timestamp: str = ""
    sequence: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any], timestamp: str = "", sequence: int = 0, quality: str = "normal") -> VitalReading:
        v = data.get("vitals", data)
        return cls(
            heart_rate=float(v.get("heart_rate", 75.0)),
            spo2=float(v.get("spo2", 98.0)),
            respiratory_rate=float(v.get("respiratory_rate", 16.0)),
            systolic_bp=float(v.get("systolic_bp", 120.0)),
            diastolic_bp=float(v.get("diastolic_bp", 80.0)),
            map=float(v.get("map", (v.get("systolic_bp", 120.0) + 2 * v.get("diastolic_bp", 80.0)) / 3.0)),
            fio2=float(v.get("fio2", 0.21)),
            quality=data.get("quality", quality),
            timestamp=timestamp,
            sequence=data.get("sequence", sequence),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PatientContext:
    patient_id: str
    age: int
    sex: str
    relevant_history: list[str] = field(default_factory=list)
    current_medications: list[str] = field(default_factory=list)
    recent_labs: dict[str, float] = field(default_factory=dict)
    baseline_vitals: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_event(cls, event: dict[str, Any]) -> PatientContext:
        pid = event.get("patient_id", "")
        payload = event.get("payload", {})
        return cls(
            patient_id=pid,
            age=int(payload.get("age", 60)),
            sex=str(payload.get("sex", "unknown")),
            relevant_history=list(payload.get("relevant_history", [])),
            current_medications=list(payload.get("current_medications", [])),
            recent_labs=dict(payload.get("recent_labs", {})),
            baseline_vitals=dict(payload.get("baseline", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGCitation:
    document_id: str
    title: str
    version: str
    locator: str
    source_url: str
    applicability: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlertRecommendation:
    actions: list[str]
    rationale: str
    citations: list[RAGCitation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": self.actions,
            "rationale": self.rationale,
            "citations": [c.to_dict() for c in self.citations],
        }


@dataclass
class AlertEvaluationResult:
    patient_id: str
    timestamp: str
    sequence: int
    decision: AlertDecision
    priority: Optional[AlertPriority]
    news2_score: int
    qsofa_score: int
    shock_index: float
    syndrome: str
    contributing_factors: list[str]
    explanation: str
    recommendation: Optional[AlertRecommendation]
    is_artifact: bool = False
    cooldown_active: bool = False
    deduplicated: bool = False
    episode_id: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "decision": self.decision.value,
            "priority": self.priority.value if self.priority else None,
            "news2_score": self.news2_score,
            "qsofa_score": self.qsofa_score,
            "shock_index": round(self.shock_index, 2),
            "syndrome": self.syndrome,
            "contributing_factors": self.contributing_factors,
            "explanation": self.explanation,
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
            "is_artifact": self.is_artifact,
            "cooldown_active": self.cooldown_active,
            "deduplicated": self.deduplicated,
            "episode_id": self.episode_id,
            "confidence": round(self.confidence, 3),
        }


@dataclass
class ClinicianFeedback:
    feedback_id: str
    alert_id: str
    patient_id: str
    timestamp: str
    clinician_id: str
    action: ClinicianActionType
    feedback_category: str
    clinical_notes: str = ""
    override_priority: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "clinician_id": self.clinician_id,
            "action": self.action.value,
            "feedback_category": self.feedback_category,
            "clinical_notes": self.clinical_notes,
            "override_priority": self.override_priority,
        }


@dataclass
class AuditRecord:
    index: int
    entry_id: str
    timestamp: str
    event_type: str
    patient_id: str
    payload: dict[str, Any]
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
