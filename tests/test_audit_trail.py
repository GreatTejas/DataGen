"""Unit tests for Audit Trail tamper-evident cryptographic chaining."""
import tempfile
import unittest
from pathlib import Path
from clinical_copilot.audit.audit_trail import AuditTrail


class TestAuditTrail(unittest.TestCase):
    def test_hash_chain_integrity(self):
        audit = AuditTrail()
        # Record observations and decisions
        audit.record("VITAL_OBSERVATION", "SYN-0001", {"hr": 75, "bp": "120/80"})
        audit.record("ALERT_DECISION", "SYN-0001", {"decision": "ALERT", "priority": "HIGH"})
        audit.record("CLINICIAN_ACTION", "SYN-0001", {"action": "ACCEPT", "clinician": "Dr. Vance"})

        # Integrity must be valid
        valid, msg, count = audit.verify_integrity()
        self.assertTrue(valid)
        self.assertEqual(count, 3)

    def test_tamper_detection(self):
        audit = AuditTrail()
        audit.record("VITAL_OBSERVATION", "SYN-0001", {"hr": 75, "bp": "120/80"})
        audit.record("ALERT_DECISION", "SYN-0001", {"decision": "ALERT", "priority": "HIGH"})

        # Intentionally tamper with the first record's payload
        audit.records[0].payload["hr"] = 150

        # Verification must fail and detect tampering
        valid, msg, count = audit.verify_integrity()
        self.assertFalse(valid)
        self.assertIn("tampered", msg.lower())

    def test_export_and_query(self):
        audit = AuditTrail()
        audit.record("ALERT_DECISION", "SYN-0001", {"val": 1})
        audit.record("ALERT_DECISION", "SYN-0002", {"val": 2})
        audit.record("CLINICIAN_ACTION", "SYN-0001", {"val": 3})

        res = audit.query(patient_id="SYN-0001")
        self.assertEqual(len(res), 2)

        res_actions = audit.query(event_type="CLINICIAN_ACTION")
        self.assertEqual(len(res_actions), 1)

        with tempfile.TemporaryDirectory() as td:
            p_jsonl = Path(td) / "audit.jsonl"
            p_csv = Path(td) / "audit.csv"
            audit.export_jsonl(p_jsonl)
            audit.export_csv(p_csv)
            self.assertTrue(p_jsonl.exists())
            self.assertTrue(p_csv.exists())


if __name__ == "__main__":
    unittest.main()
