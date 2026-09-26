"""Unit tests for Alert Policy, Cooldown, Deduplication, and Escalation Overrides."""
import unittest
from clinical_copilot.policy.alert_policy import AlertPolicyEngine
from clinical_copilot.policy.cooldown import CooldownManager
from clinical_copilot.policy.thresholds import compute_news2, compute_qsofa, compute_shock_index, identify_clinical_syndrome
from clinical_copilot.schemas import AlertDecision, AlertPriority, PatientContext, VitalReading


class TestAlertPolicy(unittest.TestCase):
    def setUp(self):
        self.engine = AlertPolicyEngine(cooldown_seconds=1200)
        self.context = PatientContext(
            patient_id="SYN-TEST-01",
            age=68,
            sex="male",
            relevant_history=["hypertension", "COPD"],
            current_medications=["beta blocker", "antihypertensive"],
            baseline_vitals={"heart_rate": 72.0, "systolic_bp": 120.0, "spo2": 91.0, "respiratory_rate": 16.0},
        )
        self.engine.register_context(self.context)

    def test_stable_vitals_suppressed(self):
        reading = VitalReading(
            heart_rate=72.0,
            spo2=91.0,  # normal for COPD baseline
            respiratory_rate=16.0,
            systolic_bp=120.0,
            diastolic_bp=75.0,
            map=90.0,
            quality="normal",
            timestamp="2026-09-18T08:00:00+00:00",
            sequence=0,
        )
        res = self.engine.evaluate_reading(reading, "SYN-TEST-01")
        self.assertEqual(res.decision, AlertDecision.SUPPRESS)
        self.assertEqual(res.syndrome, "stable")

    def test_artifact_quality_suspect_suppressed(self):
        reading = VitalReading(
            heart_rate=135.0,
            spo2=82.0,
            respiratory_rate=28.0,
            systolic_bp=90.0,
            diastolic_bp=55.0,
            map=66.0,
            quality="suspect",
            timestamp="2026-09-18T08:05:00+00:00",
            sequence=5,
        )
        res = self.engine.evaluate_reading(reading, "SYN-TEST-01")
        self.assertEqual(res.decision, AlertDecision.SUPPRESS)
        self.assertTrue(res.is_artifact)

    def test_deferral_on_first_deterioration_reading(self):
        # First reading with acute deterioration
        reading1 = VitalReading(
            heart_rate=80.0,  # beta-blocked
            spo2=90.0,
            respiratory_rate=23.0,
            systolic_bp=82.0,  # severe hypotension
            diastolic_bp=48.0,
            map=59.0,
            quality="normal",
            timestamp="2026-09-18T08:10:00+00:00",
            sequence=10,
        )
        res1 = self.engine.evaluate_reading(reading1, "SYN-TEST-01")
        # Should defer on first abnormal reading to confirm trajectory
        self.assertEqual(res1.decision, AlertDecision.DEFER)

        # Second consecutive reading confirming trajectory
        reading2 = VitalReading(
            heart_rate=82.0,
            spo2=89.5,
            respiratory_rate=24.0,
            systolic_bp=80.0,
            diastolic_bp=46.0,
            map=57.0,
            quality="normal",
            timestamp="2026-09-18T08:11:00+00:00",
            sequence=11,
        )
        res2 = self.engine.evaluate_reading(reading2, "SYN-TEST-01")
        self.assertEqual(res2.decision, AlertDecision.ALERT)
        self.assertIn(res2.priority, [AlertPriority.HIGH, AlertPriority.CRITICAL])
        self.assertIsNotNone(res2.recommendation)
        self.assertGreater(len(res2.recommendation.citations), 0)

    def test_cooldown_and_escalation_override(self):
        cd = CooldownManager(cooldown_seconds=1200)
        # 1. Initial Alert fires
        should_alert, is_esc, reason, ep_id = cd.check_alert(
            "SYN-TEST-02", "2026-09-18T08:00:00+00:00", AlertPriority.MEDIUM, "sepsis_like"
        )
        self.assertTrue(should_alert)
        self.assertFalse(is_esc)

        # 2. Subsequent alert within 5 minutes at SAME priority -> SUPPRESSED by cooldown
        should_alert2, is_esc2, reason2, _ = cd.check_alert(
            "SYN-TEST-02", "2026-09-18T08:05:00+00:00", AlertPriority.MEDIUM, "sepsis_like"
        )
        self.assertFalse(should_alert2)
        self.assertIn("cooldown", reason2.lower())

        # 3. Escalation: condition worsens to HIGH -> BREAKS COOLDOWN
        should_alert3, is_esc3, reason3, _ = cd.check_alert(
            "SYN-TEST-02", "2026-09-18T08:07:00+00:00", AlertPriority.HIGH, "sepsis_like"
        )
        self.assertTrue(should_alert3)
        self.assertTrue(is_esc3)
        self.assertIn("escalation", reason3.lower())


if __name__ == "__main__":
    unittest.main()
