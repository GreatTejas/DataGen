"""Turn ConsolidatedEvidence into a text query the RAG retriever can use.

This is a small, transparent rule-based bridge between numeric vitals
and the clinical vocabulary used in the knowledge corpus (e.g. a low
systolic BP becomes "hypotension"). It is deliberately simple and
easy to audit/extend; it is not a substitute for a real clinical NLP
component, and every threshold here should be reviewed by someone
qualified before any use beyond this hackathon demo.
"""
from __future__ import annotations

from .schema import ConsolidatedEvidence

# (vital_name, comparator, threshold, keyword phrase)
_RULES: list[tuple[str, str, float, str]] = [
    ("heart_rate", ">", 100.0, "tachycardia compensatory tachycardia rising heart rate"),
    ("heart_rate", "<", 50.0, "bradycardia"),
    ("respiratory_rate", ">", 22.0, "tachypnea rising respiratory rate"),
    ("respiratory_rate", "<", 8.0, "respiratory depression"),
    ("spo2", "<", 92.0, "hypoxemia falling oxygen saturation"),
    ("systolic_bp", "<", 100.0, "hypotension falling blood pressure shock"),
    ("map", "<", 65.0, "low mean arterial pressure organ hypoperfusion"),
]


def build_query(evidence: ConsolidatedEvidence) -> str:
    terms: list[str] = []

    for vital, comparator, threshold, phrase in _RULES:
        value = evidence.current_vitals.get(vital)
        if value is None:
            continue
        triggered = value > threshold if comparator == ">" else value < threshold
        if triggered:
            terms.append(phrase)

    if evidence.trend.persistent_deterioration:
        terms.append("sustained multi-reading trend persistent deterioration")

    if evidence.anomaly.flagged_parameters:
        terms.append(" ".join(evidence.anomaly.flagged_parameters))

    if evidence.risk.risk_category in ("medium", "high"):
        terms.append("organ dysfunction suspected infection sepsis")

    if not terms:
        terms.append("routine vital sign monitoring stable patient")

    return " ".join(terms)
