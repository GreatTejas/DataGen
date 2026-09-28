import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from clinical_reasoning.mock_evidence import build_mock_evidence

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "DataGen-main", "DataGen-main", "data", "generated"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "vitals_observed.jsonl")),
    reason="Person 1's generated sample data is not available in this checkout",
)


def test_build_mock_evidence_for_shock_episode():
    # SYN-0002 / haemodynamic_shock episode, escalation window start
    # from ground_truth_episodes.jsonl (evaluation-only file; used here
    # only to pick a realistic timestamp for the test, never read by
    # the reasoning agent itself).
    evidence = build_mock_evidence(
        data_dir=DATA_DIR,
        patient_id="SYN-0002",
        as_of="2026-09-18T10:07:00+00:00",
    )
    assert evidence.patient_id == "SYN-0002"
    assert set(evidence.current_vitals.keys()) >= {
        "heart_rate",
        "spo2",
        "respiratory_rate",
        "systolic_bp",
    }
    assert 0.0 <= evidence.risk.risk_score <= 1.0
    assert evidence.risk.risk_category in ("low", "medium", "high")
    assert 0.0 <= evidence.anomaly.anomaly_confidence <= 1.0
    assert evidence.patient_context.sex in ("male", "female", None)


def test_build_mock_evidence_unknown_patient_raises():
    with pytest.raises(ValueError):
        build_mock_evidence(
            data_dir=DATA_DIR,
            patient_id="SYN-9999",
            as_of="2026-09-18T10:07:00+00:00",
        )
