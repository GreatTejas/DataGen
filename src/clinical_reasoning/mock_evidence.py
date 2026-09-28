"""Build a ConsolidatedEvidence object from Person 1's real generated data.

Persons 2 and 3 (Patient State/Trend/Anomaly, Risk/Evidence Fusion)
have not shipped yet. To develop and demo the RAG + Clinical Reasoning
component independently, this module computes a deliberately simple,
clearly-labelled stand-in for their combined output directly from
Person 1's actual files:

    data/generated/vitals_observed.jsonl
    data/generated/patient_context.jsonl

This is NOT the trend/anomaly/risk logic those components will
ultimately implement — it exists only so `ConsolidatedEvidence` is
populated with real, patient-specific numbers instead of hand-typed
fixtures. When Persons 2/3 ship real components, replace calls to
`build_mock_evidence` with `adapters.evidence_from_fusion_output`
(same return type, so nothing downstream changes).
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .schema import (
    AnomalySummary,
    ConsolidatedEvidence,
    PatientContext,
    RiskSummary,
    TrendSummary,
)

_TRACKED_VITALS = (
    "heart_rate",
    "spo2",
    "respiratory_rate",
    "systolic_bp",
    "diastolic_bp",
    "map",
)

# Rough, non-clinical risk thresholds used only to make the mock
# evidence internally consistent for demo purposes. These are NOT a
# substitute for Person 3's Risk Assessor.
_RISK_RULES: dict[str, tuple[float, float]] = {
    # vital: (medium_threshold_delta, high_threshold_delta) as "worse than"
    "heart_rate": (110.0, 130.0),
    "spo2": (94.0, 90.0),  # inverted: lower is worse
    "respiratory_rate": (24.0, 30.0),
    "systolic_bp": (95.0, 85.0),  # inverted: lower is worse
}


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _patient_context(data_dir: Path, patient_id: str) -> PatientContext:
    for row in _load_jsonl(data_dir / "patient_context.jsonl"):
        if row["patient_id"] == patient_id:
            payload = row["payload"]
            return PatientContext(
                age=payload.get("age"),
                sex=payload.get("sex"),
                relevant_history=payload.get("relevant_history", []),
                current_medications=payload.get("current_medications", []),
                recent_labs=payload.get("recent_labs", {}),
            )
    return PatientContext()


def _risk_for_vital(vital: str, value: float) -> str:
    if vital not in _RISK_RULES:
        return "low"
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


def build_mock_evidence(
    data_dir: str | Path,
    patient_id: str,
    as_of: str,
    window_minutes: int = 30,
    baseline_readings: int = 10,
) -> ConsolidatedEvidence:
    """Build ConsolidatedEvidence for `patient_id` as of `as_of` (ISO 8601).

    `window_minutes` sets the trend-analysis window (one reading/minute
    in the generated data). `baseline_readings` is how many of the
    patient's earliest readings are treated as their personal baseline
    for the anomaly z-score.
    """
    data_dir = Path(data_dir)
    all_rows = [
        row
        for row in _load_jsonl(data_dir / "vitals_observed.jsonl")
        if row["patient_id"] == patient_id
    ]
    if not all_rows:
        raise ValueError(f"no vitals found for patient_id={patient_id!r}")
    all_rows.sort(key=lambda r: r["event_time"])

    up_to = [r for r in all_rows if r["event_time"] <= as_of]
    if not up_to:
        raise ValueError(f"no readings for {patient_id} at or before {as_of}")

    latest = up_to[-1]
    current_vitals = dict(latest["payload"]["vitals"])

    window = up_to[-window_minutes:]
    baseline = all_rows[:baseline_readings]

    trend = _compute_trend(window)
    anomaly = _compute_anomaly(latest["payload"]["vitals"], baseline)
    risk = _compute_risk(current_vitals)

    conflicts: list[str] = []
    if len(window) < window_minutes:
        conflicts.append(
            f"trend window truncated to {len(window)} readings "
            f"(requested {window_minutes}); insufficient history"
        )

    evidence_id = f"MOCKEVID-{patient_id}-{latest['event_id'][:8]}"

    return ConsolidatedEvidence(
        schema_version="0.1.0-p4-draft",
        evidence_id=evidence_id,
        patient_id=patient_id,
        window_start=window[0]["event_time"],
        window_end=latest["event_time"],
        current_vitals=current_vitals,
        trend=trend,
        anomaly=anomaly,
        risk=risk,
        patient_context=_patient_context(data_dir, patient_id),
        conflicts_or_missing=conflicts,
        fusion_confidence=0.9 if not conflicts else 0.6,
    )


def _compute_trend(window: list[dict[str, Any]]) -> TrendSummary:
    if len(window) < 2:
        return TrendSummary(trend_score=0.0, persistent_deterioration=False)

    slopes: dict[str, float] = {}
    for vital in _TRACKED_VITALS:
        series = [
            row["payload"]["vitals"].get(vital)
            for row in window
            if row["payload"]["vitals"].get(vital) is not None
        ]
        if len(series) < 2:
            continue
        xs = list(range(len(series)))
        mean_x = sum(xs) / len(xs)
        mean_y = sum(series) / len(series)
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, series))
        den = sum((x - mean_x) ** 2 for x in xs) or 1e-9
        slopes[vital] = num / den

    worsening_directions = {
        "heart_rate": 1,
        "respiratory_rate": 1,
        "spo2": -1,
        "systolic_bp": -1,
        "map": -1,
    }
    worsening_count = sum(
        1
        for vital, slope in slopes.items()
        if vital in worsening_directions
        and slope * worsening_directions[vital] > 0.05
    )
    trend_score = min(1.0, worsening_count / max(1, len(worsening_directions)))
    persistent = worsening_count >= 2 and len(window) >= 10

    notes = [
        f"{vital} {'rising' if slope > 0 else 'falling'} "
        f"({slope:+.2f}/reading)"
        for vital, slope in slopes.items()
        if abs(slope) > 0.05
    ]

    return TrendSummary(
        trend_score=trend_score,
        persistent_deterioration=persistent,
        per_vital_slope=slopes,
        notes=notes,
    )


def _compute_anomaly(
    current: dict[str, float], baseline_rows: list[dict[str, Any]]
) -> AnomalySummary:
    flagged: list[str] = []
    z_scores: list[float] = []
    for vital in _TRACKED_VITALS:
        baseline_series = [
            row["payload"]["vitals"].get(vital)
            for row in baseline_rows
            if row["payload"]["vitals"].get(vital) is not None
        ]
        value = current.get(vital)
        if value is None or len(baseline_series) < 2:
            continue
        mean = statistics.mean(baseline_series)
        stdev = statistics.pstdev(baseline_series) or 1e-6
        z = (value - mean) / stdev
        z_scores.append(abs(z))
        if abs(z) >= 2.0:
            flagged.append(vital)

    confidence = min(1.0, max(z_scores) / 4.0) if z_scores else 0.0
    return AnomalySummary(anomaly_confidence=confidence, flagged_parameters=flagged)


def _compute_risk(current: dict[str, float]) -> RiskSummary:
    per_vital_risk = {
        vital: _risk_for_vital(vital, current[vital])
        for vital in _RISK_RULES
        if vital in current
    }
    order = {"low": 0, "medium": 1, "high": 2}
    worst = max(per_vital_risk.values(), key=lambda c: order[c], default="low")
    score = order[worst] / 2.0
    return RiskSummary(
        risk_score=score, risk_category=worst, per_vital_risk=per_vital_risk
    )
