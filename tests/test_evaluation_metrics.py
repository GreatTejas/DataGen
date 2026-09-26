"""Unit tests for Evaluation Metrics calculation."""
import unittest
from clinical_copilot.evaluation.metrics import compute_metrics


class TestEvaluationMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        gt_episodes = [
            {
                "episode_id": "EP-01",
                "patient_id": "SYN-0001",
                "onset_time": "2026-09-18T08:20:00+00:00",
                "escalation_window_start": "2026-09-18T09:00:00+00:00",
                "scenario": "sepsis_like",
            }
        ]
        eval_labels = [
            {"patient_id": "SYN-0001", "event_time": "2026-09-18T08:25:00+00:00", "artifact_injected": False},
            {"patient_id": "SYN-0002", "event_time": "2026-09-18T08:25:00+00:00", "artifact_injected": True},
        ]
        alert_results = [
            # True positive alert on SYN-0001
            {
                "patient_id": "SYN-0001",
                "timestamp": "2026-09-18T08:25:00+00:00",
                "decision": "ALERT",
                "priority": "HIGH",
                "recommendation": {"citations": [{"document_id": "sepsis3-qsofa-2016"}]},
            },
            # Cooldown suppressed alert on SYN-0001
            {
                "patient_id": "SYN-0001",
                "timestamp": "2026-09-18T08:26:00+00:00",
                "decision": "SUPPRESS",
                "priority": "HIGH",
                "cooldown_active": True,
            },
        ]

        metrics = compute_metrics(
            ground_truth_episodes=gt_episodes,
            evaluation_labels=eval_labels,
            alert_results=alert_results,
            total_patients=2,
            monitoring_minutes=60,
            audit_chain_valid=True,
            audit_records_count=10,
        )

        self.assertEqual(metrics.true_positives, 1)
        self.assertEqual(metrics.false_negatives, 0)
        self.assertEqual(metrics.episode_recall, 1.0)
        self.assertEqual(metrics.median_lead_time_minutes, 35.0)  # 09:00:00 - 08:25:00 = 35 min
        self.assertEqual(metrics.rag_citation_coverage, 1.0)
        self.assertEqual(metrics.artifact_false_positive_rate, 0.0)
        self.assertTrue(metrics.audit_chain_verified)


if __name__ == "__main__":
    unittest.main()
