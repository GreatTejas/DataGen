"""Evaluation metrics calculator for Clinical Copilot."""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EvaluationMetricsResult:
    total_patients: int
    total_patient_hours: float
    total_vital_events: int
    total_alerts_fired: int
    total_alerts_deferred: int
    total_alerts_suppressed: int
    alerts_per_patient_hour: float

    # Detection performance on ground-truth episodes
    ground_truth_episodes_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    episode_recall: float
    episode_precision: float
    episode_f1: float

    # Lead time
    median_lead_time_minutes: float
    mean_lead_time_minutes: float

    # Noise & fatigue reduction
    duplicate_suppression_count: int
    duplicate_suppression_rate: float
    artifact_episodes_count: int
    artifact_false_positives: int
    artifact_false_positive_rate: float

    # Evidence & Audit
    rag_citation_coverage: float
    audit_chain_verified: bool
    audit_records_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_iso(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def compute_metrics(
    ground_truth_episodes: list[dict[str, Any]],
    evaluation_labels: list[dict[str, Any]],
    alert_results: list[dict[str, Any]],
    total_patients: int,
    monitoring_minutes: int,
    audit_chain_valid: bool,
    audit_records_count: int,
) -> EvaluationMetricsResult:
    """Computes comprehensive evaluation benchmark metrics against ground truth."""
    total_patient_hours = (total_patients * monitoring_minutes) / 60.0
    total_vital_events = len(evaluation_labels)

    total_alerts_fired = sum(1 for a in alert_results if a["decision"] == "ALERT")
    total_alerts_deferred = sum(1 for a in alert_results if a["decision"] == "DEFER")
    total_alerts_suppressed = sum(1 for a in alert_results if a["decision"] == "SUPPRESS")
    alerts_per_patient_hour = total_alerts_fired / max(total_patient_hours, 0.01)

    # Deduplication suppression
    duplicate_suppressions = sum(1 for a in alert_results if a.get("cooldown_active") or a.get("deduplicated"))
    total_opportunities = total_alerts_fired + duplicate_suppressions
    duplicate_suppression_rate = (duplicate_suppressions / total_opportunities) if total_opportunities > 0 else 0.0

    # Group alerts by patient
    patient_alerts: dict[str, list[dict[str, Any]]] = {}
    for a in alert_results:
        if a["decision"] == "ALERT":
            patient_alerts.setdefault(a["patient_id"], []).append(a)

    # Ground truth evaluation
    gt_by_patient = {e["patient_id"]: e for e in ground_truth_episodes}
    gt_count = len(ground_truth_episodes)

    tp = 0
    fn = 0
    lead_times_minutes: list[float] = []

    for gt in ground_truth_episodes:
        pid = gt["patient_id"]
        onset = parse_iso(gt["onset_time"])
        window_start = parse_iso(gt["escalation_window_start"])

        alerts = patient_alerts.get(pid, [])
        # Check if an alert was raised around onset or during escalation window
        valid_alerts = [a for a in alerts if parse_iso(a["timestamp"]) >= onset]

        if valid_alerts:
            tp += 1
            first_alert_ts = parse_iso(valid_alerts[0]["timestamp"])
            lead_time_min = (window_start - first_alert_ts).total_seconds() / 60.0
            lead_times_minutes.append(lead_time_min)
        else:
            fn += 1

    # False Positives: patients without ground-truth episodes who received an alert
    patients_with_alerts = set(patient_alerts.keys())
    fp = sum(1 for pid in patients_with_alerts if pid not in gt_by_patient)

    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    median_lead = statistics.median(lead_times_minutes) if lead_times_minutes else 0.0
    mean_lead = statistics.mean(lead_times_minutes) if lead_times_minutes else 0.0

    # Artifact False Positive Rate
    # Evaluates whether injected artifact events triggered a false alert within the artifact window
    artifact_events = [l for l in evaluation_labels if l.get("artifact_injected")]
    artifact_count = len(artifact_events)
    artifact_fps = 0
    for art in artifact_events:
        pid = art["patient_id"]
        art_ts = parse_iso(art["event_time"])
        # Check if an alert was fired for this patient within [-60s, +300s] of the artifact
        for a in patient_alerts.get(pid, []):
            a_ts = parse_iso(a["timestamp"])
            diff_secs = (a_ts - art_ts).total_seconds()
            if -60 <= diff_secs <= 300:
                artifact_fps += 1
                break
    artifact_fpr = (artifact_fps / artifact_count) if artifact_count > 0 else 0.0

    # RAG Citation Coverage
    # % of fired alerts with at least 1 citation
    fired_alerts = [a for a in alert_results if a["decision"] == "ALERT"]
    with_citation = sum(
        1 for a in fired_alerts if a.get("recommendation") and a["recommendation"].get("citations")
    )
    rag_coverage = (with_citation / len(fired_alerts)) if fired_alerts else 1.0

    return EvaluationMetricsResult(
        total_patients=total_patients,
        total_patient_hours=round(total_patient_hours, 1),
        total_vital_events=total_vital_events,
        total_alerts_fired=total_alerts_fired,
        total_alerts_deferred=total_alerts_deferred,
        total_alerts_suppressed=total_alerts_suppressed,
        alerts_per_patient_hour=round(alerts_per_patient_hour, 2),
        ground_truth_episodes_count=gt_count,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        episode_recall=round(recall, 4),
        episode_precision=round(precision, 4),
        episode_f1=round(f1, 4),
        median_lead_time_minutes=round(median_lead, 1),
        mean_lead_time_minutes=round(mean_lead, 1),
        duplicate_suppression_count=duplicate_suppressions,
        duplicate_suppression_rate=round(duplicate_suppression_rate, 4),
        artifact_episodes_count=artifact_count,
        artifact_false_positives=artifact_fps,
        artifact_false_positive_rate=round(artifact_fpr, 4),
        rag_citation_coverage=round(rag_coverage, 4),
        audit_chain_verified=audit_chain_valid,
        audit_records_count=audit_records_count,
    )
