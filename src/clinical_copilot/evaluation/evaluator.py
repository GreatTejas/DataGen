"""End-to-end evaluation pipeline comparing copilot alerts against ground truth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..audit.audit_trail import AuditTrail
from ..policy.alert_policy import AlertPolicyEngine
from ..schemas import PatientContext, VitalReading
from .metrics import EvaluationMetricsResult, compute_metrics


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class CopilotEvaluator:
    """Evaluates the Alert Policy & Escalation engine against held-out ground truth data."""

    def __init__(self, data_dir: str | Path, cooldown_seconds: int = 1200) -> None:
        self.data_dir = Path(data_dir)
        self.cooldown_seconds = cooldown_seconds
        self.engine = AlertPolicyEngine(
            cooldown_seconds=cooldown_seconds,
            corpus_path=self.data_dir / "knowledge_corpus.jsonl",
        )
        self.audit = AuditTrail()

    def run_evaluation(self, output_report_path: Optional[str | Path] = None) -> EvaluationMetricsResult:
        contexts_data = load_jsonl(self.data_dir / "patient_context.jsonl")
        vitals_data = load_jsonl(self.data_dir / "vitals_observed.jsonl")
        episodes_data = load_jsonl(self.data_dir / "ground_truth_episodes.jsonl")
        labels_data = load_jsonl(self.data_dir / "evaluation_labels.jsonl")

        # 1. Register contexts
        patient_ids = set()
        for c in contexts_data:
            ctx = PatientContext.from_event(c)
            self.engine.register_context(ctx)
            patient_ids.add(ctx.patient_id)
            self.audit.record(
                event_type="PATIENT_CONTEXT_INITIALIZED",
                patient_id=ctx.patient_id,
                payload=ctx.to_dict(),
                timestamp=c.get("event_time"),
            )

        # 2. Replay vitals through the policy engine
        alert_results = []
        for v_event in vitals_data:
            pid = v_event.get("patient_id")
            ts = v_event.get("event_time", "")
            payload = v_event.get("payload", {})
            v_dict = payload.get("vitals", {})
            seq = payload.get("sequence", 0)
            quality = payload.get("quality", "normal")

            reading = VitalReading.from_dict(v_dict, timestamp=ts, sequence=seq, quality=quality)
            reading.patient_id = pid

            # Log observation in audit trail
            self.audit.record(
                event_type="VITAL_OBSERVATION",
                patient_id=pid,
                payload={"vitals": reading.to_dict(), "sequence": seq, "quality": quality},
                timestamp=ts,
            )

            # Evaluate through policy engine
            res = self.engine.evaluate_reading(reading, patient_id=pid)
            res_dict = res.to_dict()
            alert_results.append(res_dict)

            # If ALERT or DEFER, log decision and RAG citations to audit trail
            if res.decision.value in {"ALERT", "DEFER"}:
                self.audit.record(
                    event_type="ALERT_DECISION",
                    patient_id=pid,
                    payload=res_dict,
                    timestamp=ts,
                )

        # 3. Verify audit trail cryptographic integrity
        valid, msg, rec_count = self.audit.verify_integrity()

        # 4. Compute duration / patient hours
        minutes = 180
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text(encoding="utf-8"))
                minutes = m.get("minutes", 180)
            except Exception:
                pass

        results = compute_metrics(
            ground_truth_episodes=episodes_data,
            evaluation_labels=labels_data,
            alert_results=alert_results,
            total_patients=len(patient_ids) or 60,
            monitoring_minutes=minutes,
            audit_chain_valid=valid,
            audit_records_count=rec_count,
        )

        if output_report_path:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(results.to_dict(), f, indent=2)

        return results
