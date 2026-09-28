import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clinical_reasoning.experience_memory_client import StubExperienceMemoryClient
from clinical_reasoning.fake_llm import FakeAnthropicClient
from clinical_reasoning.knowledge_store import RAGRetriever
from clinical_reasoning.reasoning_agent import ClinicalReasoningAgent
from clinical_reasoning.schema import (
    AnomalySummary,
    ConsolidatedEvidence,
    PatientContext,
    RiskSummary,
    TrendSummary,
)

CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "knowledge_corpus.jsonl"
)


def _sample_evidence() -> ConsolidatedEvidence:
    return ConsolidatedEvidence(
        schema_version="0.1.0-p4-draft",
        evidence_id="TEST-0001",
        patient_id="SYN-TEST",
        window_start="2026-09-18T09:30:00+00:00",
        window_end="2026-09-18T10:00:00+00:00",
        current_vitals={
            "heart_rate": 128.0,
            "spo2": 91.0,
            "respiratory_rate": 26.0,
            "systolic_bp": 88.0,
            "diastolic_bp": 55.0,
            "map": 66.0,
        },
        trend=TrendSummary(
            trend_score=0.8,
            persistent_deterioration=True,
            per_vital_slope={"heart_rate": 1.2, "systolic_bp": -1.5},
            notes=["heart_rate rising", "systolic_bp falling"],
        ),
        anomaly=AnomalySummary(
            anomaly_confidence=0.85, flagged_parameters=["heart_rate", "systolic_bp"]
        ),
        risk=RiskSummary(
            risk_score=1.0,
            risk_category="high",
            per_vital_risk={"heart_rate": "high", "systolic_bp": "high"},
        ),
        patient_context=PatientContext(age=64, sex="female"),
        conflicts_or_missing=[],
        fusion_confidence=0.9,
    )


def test_reasoning_agent_produces_grounded_output_offline():
    retriever = RAGRetriever.from_jsonl(CORPUS_PATH)
    agent = ClinicalReasoningAgent(
        retriever=retriever,
        experience_client=StubExperienceMemoryClient(),
        client=FakeAnthropicClient(),
    )

    output = agent.run(_sample_evidence())

    assert output.patient_id == "SYN-TEST"
    assert output.evidence_id == "TEST-0001"
    assert output.explanation
    assert output.recommended_action
    assert output.tool_call_count == 3  # search + retrieve_similar_cases + submit

    # Every citation returned must be a document that actually exists
    # in the corpus (no invented document_ids).
    from clinical_reasoning.knowledge_store import load_corpus

    corpus_ids = {d.document_id for d in load_corpus(CORPUS_PATH)}
    for citation in output.citations:
        assert citation.document_id in corpus_ids
        assert citation.version
