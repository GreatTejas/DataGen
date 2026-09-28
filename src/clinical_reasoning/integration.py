"""Wire P4 into the real P1 (Trend) and P2 (Patient State + Anomaly) code.

As of this update, P1's `TrendAgent` and P2's `PatientStateManager` /
`AnomalyAgent` exist and are importable. P3 (Risk Assessor + Evidence
Fusion + Experience Memory retrieval) is still not built. This module
therefore does real integration for the two pieces that exist and
keeps a clearly-labelled, swappable placeholder for the one that
doesn't — see `PLACEHOLDER: Risk / Evidence Fusion` below.

Import boundary
----------------
This module is the ONLY place in the P4 package that imports P1/P2
code (`clinical_data_gen.*`, `trend_agent.*`). Everything else in this
package (`reasoning_agent.py`, `knowledge_store.py`, ...) only knows
about this package's own `schema.py` types. That keeps the rest of P4
usable even in a checkout where P1/P2's packages aren't installed
(see `mock_evidence.py` for that fully-standalone path).

Two real naming/shape mismatches between P1 and P2 surfaced while
wiring this up. Both are called out below rather than silently
papered over, since resolving them isn't P4's call to make:

1. **Vital-name vocabulary.** P1's `TrendAgent` and the raw
   `vitals_observed.jsonl` stream use generator-native names
   (`heart_rate`, `spo2`, `respiratory_rate`, `systolic_bp`,
   `diastolic_bp`, `map`, `fio2`). P2's `PatientState` /
   `AnomalyAgent` use PhysioNet-style `CORE_VITALS` names (`HR`,
   `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`). There's also no
   generator field for `Temp`, so it is always reported missing to
   P2's pipeline today. `CORE_VITAL_FROM_GENERATED` below is the
   mapping P4 uses at the P2 boundary; `ConsolidatedEvidence.
   current_vitals` itself stays in generator-native names throughout
   this package, since that's what `query_builder.py` and the rest of
   P4 already key off of.
2. **Patient context richness.** P2's `StaticContext` (age, gender,
   unit1, unit2, hosp_adm_time) doesn't carry `relevant_history`,
   `current_medications`, or `recent_labs`, which the report's Sec.
   3.2 says the Clinical Reasoning Agent needs and which the raw
   `patient_context.jsonl` payload already has. This module reads
   patient context from the raw payload, not from `state.
   static_context`, until that's reconciled.

Nothing here depends on P3 existing: `build_consolidated_evidence`
takes an optional `risk` argument and only falls back to the local
placeholder heuristic (also used by `mock_evidence.py`) when none is
supplied.
"""
from __future__ import annotations

from typing import Any, Optional

from .schema import (
    AnomalySummary,
    ConsolidatedEvidence,
    PatientContext,
    RiskSummary,
    TrendSummary,
)

# Generator-native vital name -> P2 CORE_VITALS name.
# No generator source exists for "Temp"; no CORE_VITALS destination
# exists for "fio2". Both are intentionally left unmapped.
CORE_VITAL_FROM_GENERATED: dict[str, str] = {
    "heart_rate": "HR",
    "spo2": "O2Sat",
    "respiratory_rate": "Resp",
    "systolic_bp": "SBP",
    "diastolic_bp": "DBP",
    "map": "MAP",
}


def generated_vitals_to_core_vitals(vitals: dict[str, float]) -> dict[str, float]:
    """Translate one vitals_observed.jsonl payload into P2's vital names."""
    return {
        CORE_VITAL_FROM_GENERATED[name]: value
        for name, value in vitals.items()
        if name in CORE_VITAL_FROM_GENERATED and value is not None
    }


def update_patient_state(
    state_manager: Any,  # clinical_data_gen.state.patient_state_manager.PatientStateManager
    patient_id: str,
    event_time: str,
    generated_vitals: dict[str, float],
) -> Any:  # -> PatientState
    """Feed one real generated vitals reading into P2's PatientStateManager.

    Thin wrapper so callers don't need to know about the name mapping.
    """
    core_vitals = generated_vitals_to_core_vitals(generated_vitals)
    return state_manager.update(
        patient_id=patient_id,
        timestamp=event_time,
        vitals=core_vitals,
    )


def trend_summary_from_trend_agent_output(trend_output: dict[str, Any]) -> TrendSummary:
    """Convert P1 TrendAgent.observe()'s real output into a TrendSummary.

    Verified against trend_agent/trend_agent.py and
    trend_agent/mock/trend_analysis.jsonl: the real keys are
    candidate_deterioration_trend, concurrent_worsening_signals,
    trends (list of {vital, direction, change, ...}), explanation.
    """
    per_vital_slope = {
        item["vital"]: item["change"]
        for item in trend_output.get("trends", [])
        if item.get("direction") != "stable"
    }
    return TrendSummary(
        trend_score=(
            1.0
            if trend_output.get("candidate_deterioration_trend")
            else (0.5 if trend_output.get("concurrent_worsening_signals") else 0.0)
        ),
        persistent_deterioration=bool(
            trend_output.get("candidate_deterioration_trend", False)
        ),
        per_vital_slope=per_vital_slope,
        notes=[trend_output.get("explanation", "")] if trend_output.get("explanation") else [],
    )


def anomaly_summary_from_anomaly_result(anomaly_result: Any) -> AnomalySummary:
    """Convert P2's real AnomalyResult into an AnomalySummary.

    Verified against src/clinical_data_gen/anomaly/anomaly_agent.py:
    AnomalyResult has probability, is_anomalous, contributing_features
    (feature names like "HR.value", "HR.delta", not bare vital names).
    We keep the raw feature names in flagged_parameters rather than
    reverse-mapping them to generator vital names, since collapsing
    e.g. "HR.value" and "HR.delta" both to "heart_rate" would lose
    real information about *why* the feature was flagged.
    """
    return AnomalySummary(
        anomaly_confidence=float(getattr(anomaly_result, "probability", 0.0)),
        flagged_parameters=list(getattr(anomaly_result, "contributing_features", [])),
    )


# ---------------------------------------------------------------------
# PLACEHOLDER: Risk / Evidence Fusion (Person 3 — not built yet)
# ---------------------------------------------------------------------
# Same heuristic as mock_evidence.py's _compute_risk, duplicated here
# (not imported from mock_evidence) so this module has no dependency
# on the fully-synthetic mock path. Delete this function and always
# require a real `risk` argument once P3 ships.

_RISK_RULES: dict[str, tuple[float, float]] = {
    "heart_rate": (110.0, 130.0),
    "spo2": (94.0, 90.0),  # inverted: lower is worse
    "respiratory_rate": (24.0, 30.0),
    "systolic_bp": (95.0, 85.0),  # inverted: lower is worse
}


def placeholder_risk_from_vitals(current_vitals: dict[str, float]) -> RiskSummary:
    def risk_for(vital: str, value: float) -> str:
        medium, high = _RISK_RULES[vital]
        inverted = vital in ("spo2", "systolic_bp")
        if inverted:
            if value <= high:
                return "high"
            if value <= medium:
                return "medium"
            return "low"
        if value >= high:
            return "high"
        if value >= medium:
            return "medium"
        return "low"

    per_vital_risk = {
        vital: risk_for(vital, current_vitals[vital])
        for vital in _RISK_RULES
        if vital in current_vitals
    }
    order = {"low": 0, "medium": 1, "high": 2}
    worst = max(per_vital_risk.values(), key=lambda c: order[c], default="low")
    return RiskSummary(
        risk_score=order[worst] / 2.0, risk_category=worst, per_vital_risk=per_vital_risk
    )


def patient_context_from_raw_payload(payload: dict[str, Any]) -> PatientContext:
    """Read patient context from the raw patient_context.jsonl payload.

    See module docstring point 2 for why this reads the raw payload
    rather than P2's StaticContext.
    """
    return PatientContext(
        age=payload.get("age"),
        sex=payload.get("sex"),
        relevant_history=payload.get("relevant_history", []),
        current_medications=payload.get("current_medications", []),
        recent_labs=payload.get("recent_labs", {}),
    )


def build_consolidated_evidence(
    evidence_id: str,
    patient_id: str,
    window_start: str,
    window_end: str,
    generated_vitals: dict[str, float],
    trend_output: dict[str, Any],
    anomaly_result: Any,
    patient_context_payload: dict[str, Any],
    risk: Optional[RiskSummary] = None,
    conflicts_or_missing: Optional[list[str]] = None,
) -> ConsolidatedEvidence:
    """Assemble ConsolidatedEvidence from real P1 + P2 outputs.

    `risk` is optional and should be supplied once P3's Risk Assessor
    exists; until then this falls back to `placeholder_risk_from_vitals`
    and the returned evidence is marked with a lower fusion_confidence
    and a note in conflicts_or_missing so downstream consumers (and the
    audit trail) can see the risk assessment isn't real yet.
    """
    conflicts = list(conflicts_or_missing or [])
    if risk is None:
        risk = placeholder_risk_from_vitals(generated_vitals)
        conflicts.append(
            "risk_category is a P4 placeholder heuristic; "
            "Person 3's Risk Assessor has not shipped yet"
        )

    return ConsolidatedEvidence(
        schema_version="0.1.0-p4-draft",
        evidence_id=evidence_id,
        patient_id=patient_id,
        window_start=window_start,
        window_end=window_end,
        current_vitals=generated_vitals,
        trend=trend_summary_from_trend_agent_output(trend_output),
        anomaly=anomaly_summary_from_anomaly_result(anomaly_result),
        risk=risk,
        patient_context=patient_context_from_raw_payload(patient_context_payload),
        conflicts_or_missing=conflicts,
        fusion_confidence=0.75 if "risk_category is a P4 placeholder" in " ".join(conflicts) else 0.9,
    )
