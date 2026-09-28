import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

DATAGEN_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "DataGen-main", "DataGen-main"
)
DATA_DIR = os.path.join(DATAGEN_ROOT, "data", "generated")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "vitals_observed.jsonl"))
    or not os.path.exists(os.path.join(DATAGEN_ROOT, "trend_agent", "trend_agent.py")),
    reason="Person 1/2's real checkout (TrendAgent, PatientStateManager, AnomalyAgent) "
    "is not available next to this repo",
)


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def real_modules():
    sys.path.insert(0, os.path.join(DATAGEN_ROOT, "src"))
    sys.path.insert(0, DATAGEN_ROOT)
    from clinical_data_gen.anomaly.anomaly_agent import AnomalyAgent
    from clinical_data_gen.state.patient_state_manager import PatientStateManager
    from trend_agent.trend_agent import TrendAgent

    return TrendAgent, PatientStateManager, AnomalyAgent


def test_real_pipeline_produces_consolidated_evidence(real_modules):
    TrendAgent, PatientStateManager, AnomalyAgent = real_modules

    from clinical_reasoning.integration import (
        build_consolidated_evidence,
        update_patient_state,
    )

    vitals_rows = sorted(
        (r for r in _load_jsonl(os.path.join(DATA_DIR, "vitals_observed.jsonl"))
         if r["patient_id"] == "SYN-0002"),
        key=lambda r: r["event_time"],
    )[:10]
    context_row = next(
        r for r in _load_jsonl(os.path.join(DATA_DIR, "patient_context.jsonl"))
        if r["patient_id"] == "SYN-0002"
    )

    trend_agent = TrendAgent()
    trend_agent.observe_context({"patient_id": "SYN-0002"})
    trend_output = None
    for row in vitals_rows:
        trend_output = trend_agent.observe(row)

    state_manager = PatientStateManager()
    anomaly_agent = AnomalyAgent()
    state = None
    for row in vitals_rows:
        state = update_patient_state(
            state_manager, "SYN-0002", row["event_time"], row["payload"]["vitals"]
        )
    anomaly_result = anomaly_agent.predict(state)

    evidence = build_consolidated_evidence(
        evidence_id="TEST-REAL-0001",
        patient_id="SYN-0002",
        window_start=vitals_rows[0]["event_time"],
        window_end=vitals_rows[-1]["event_time"],
        generated_vitals=vitals_rows[-1]["payload"]["vitals"],
        trend_output=trend_output,
        anomaly_result=anomaly_result,
        patient_context_payload=context_row["payload"],
    )

    assert evidence.patient_id == "SYN-0002"
    assert evidence.trend.notes  # real explanation string from TrendAgent
    assert 0.0 <= evidence.anomaly.anomaly_confidence <= 1.0
    # risk is still the documented placeholder until P3 ships
    assert any("placeholder" in note for note in evidence.conflicts_or_missing)


def test_core_vital_name_mapping_matches_p2_constants(real_modules):
    _, _, _ = real_modules
    from clinical_data_gen.preprocessing.constants import CORE_VITALS

    from clinical_reasoning.integration import CORE_VITAL_FROM_GENERATED

    for core_name in CORE_VITAL_FROM_GENERATED.values():
        assert core_name in CORE_VITALS
