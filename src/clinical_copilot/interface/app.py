"""FastAPI web application serving the Clinician Copilot Dashboard and API endpoints."""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..audit.audit_trail import AuditTrail
from ..evaluation.evaluator import CopilotEvaluator, load_jsonl
from ..policy.alert_policy import AlertPolicyEngine
from ..schemas import (
    AlertDecision,
    AlertPriority,
    ClinicianActionType,
    ClinicianFeedback,
    PatientContext,
    VitalReading,
)


app = FastAPI(title="Clinical Deterioration Copilot", version="0.1.0")

# Mount templates and static directories
interface_dir = Path(__file__).parent
templates = Jinja2Templates(directory=str(interface_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(interface_dir / "static")), name="static")

# Shared state
def find_data_dir() -> Path:
    candidates = [
        Path("data/generated"),
        Path("DataGen-main/data/generated"),
        Path(__file__).resolve().parents[3] / "data" / "generated",
    ]
    for c in candidates:
        if (c / "patient_context.jsonl").exists():
            return c
    return Path("data/generated")

data_dir = find_data_dir()
engine: Optional[AlertPolicyEngine] = None
audit: Optional[AuditTrail] = None
patient_records: dict[str, dict[str, Any]] = {}
vitals_history: dict[str, list[dict[str, Any]]] = {}
latest_evaluation: dict[str, dict[str, Any]] = {}
ground_truth_map: dict[str, dict[str, Any]] = {}
simulation_cursor: int = 0
all_vitals_events: list[dict[str, Any]] = []


class ClinicianActionRequest(BaseModel):
    patient_id: str
    alert_id: Optional[str] = None
    action: str  # ACCEPT, DISMISS, DEFER, INVESTIGATE
    feedback_category: str
    clinical_notes: str = ""
    clinician_id: str = "Dr. C. Vance, MD (ICU Attending)"


def init_system() -> None:
    global engine, audit, patient_records, vitals_history, latest_evaluation, ground_truth_map, all_vitals_events
    if engine is not None:
        return

    engine = AlertPolicyEngine(cooldown_seconds=1200, corpus_path=data_dir / "knowledge_corpus.jsonl")
    audit = AuditTrail()

    # Load patient contexts
    contexts = load_jsonl(data_dir / "patient_context.jsonl")
    for c in contexts:
        ctx = PatientContext.from_event(c)
        engine.register_context(ctx)
        patient_records[ctx.patient_id] = {
            "context": ctx.to_dict(),
            "status": "STABLE",
            "active_alert": None,
            "latest_vitals": None,
            "latest_news2": 0,
            "latest_si": 0.0,
        }
        vitals_history[ctx.patient_id] = []
        audit.record("PATIENT_CONTEXT_INITIALIZED", ctx.patient_id, ctx.to_dict(), c.get("event_time"))

    # Load ground truth episodes
    gt_episodes = load_jsonl(data_dir / "ground_truth_episodes.jsonl")
    for gt in gt_episodes:
        ground_truth_map[gt["patient_id"]] = gt

    # Load all vitals events
    all_vitals_events = load_jsonl(data_dir / "vitals_observed.jsonl")


@app.on_event("startup")
def on_startup() -> None:
    init_system()


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/patients")
def get_patients() -> list[dict[str, Any]]:
    init_system()
    res = []
    for pid, pdata in patient_records.items():
        ctx = pdata["context"]
        eval_res = latest_evaluation.get(pid)
        res.append({
            "patient_id": pid,
            "age": ctx.get("age"),
            "sex": ctx.get("sex"),
            "history": ctx.get("relevant_history", []),
            "medications": ctx.get("current_medications", []),
            "status": pdata["status"],
            "latest_vitals": pdata["latest_vitals"],
            "latest_news2": pdata["latest_news2"],
            "latest_si": round(pdata["latest_si"], 2),
            "active_alert": pdata["active_alert"],
            "has_ground_truth": pid in ground_truth_map,
            "gt_scenario": ground_truth_map[pid]["scenario"] if pid in ground_truth_map else None,
        })
    # Sort: active alerts first (by priority), then by news2 score descending
    priority_weight = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    res.sort(
        key=lambda x: (
            priority_weight.get(x["active_alert"]["priority"], 0) if x["active_alert"] else 0,
            x["latest_news2"],
        ),
        reverse=True,
    )
    return res


@app.get("/api/patients/{patient_id}")
def get_patient_detail(patient_id: str) -> dict[str, Any]:
    init_system()
    if patient_id not in patient_records:
        raise HTTPException(status_code=404, detail="Patient not found")

    pdata = patient_records[patient_id]
    history_readings = vitals_history.get(patient_id, [])
    active_alert = pdata["active_alert"]
    gt = ground_truth_map.get(patient_id)

    return {
        "patient_id": patient_id,
        "context": pdata["context"],
        "status": pdata["status"],
        "latest_vitals": pdata["latest_vitals"],
        "latest_news2": pdata["latest_news2"],
        "latest_si": round(pdata["latest_si"], 2),
        "active_alert": active_alert,
        "vitals_history": history_readings[-60:],  # last 60 minutes of telemetry
        "ground_truth": gt,
    }


@app.post("/api/patients/{patient_id}/action")
def record_clinician_action(patient_id: str, req: ClinicianActionRequest) -> dict[str, Any]:
    init_system()
    if patient_id not in patient_records:
        raise HTTPException(status_code=404, detail="Patient not found")

    pdata = patient_records[patient_id]
    action_type = req.action.upper()
    feedback_id = f"FB-{uuid.uuid4().hex[:8]}"
    alert_id = req.alert_id or (pdata["active_alert"]["episode_id"] if pdata["active_alert"] else f"ALERT-{patient_id}")
    now_ts = datetime.now(UTC).isoformat()

    feedback = ClinicianFeedback(
        feedback_id=feedback_id,
        alert_id=alert_id,
        patient_id=patient_id,
        timestamp=now_ts,
        clinician_id=req.clinician_id,
        action=ClinicianActionType(action_type),
        feedback_category=req.feedback_category,
        clinical_notes=req.clinical_notes,
    )

    # 1. Update audit trail
    assert audit is not None
    audit.record("CLINICIAN_ACTION", patient_id, {"action": action_type, "alert_id": alert_id, "clinician_id": req.clinician_id}, timestamp=now_ts)
    audit.record("CLINICIAN_FEEDBACK", patient_id, feedback.to_dict(), timestamp=now_ts)

    # 2. Inform cooldown manager
    assert engine is not None
    engine.cooldown_manager.record_clinician_action(patient_id, action_type)

    # 3. Update patient dashboard status
    if action_type == "ACCEPT":
        pdata["status"] = "ACCEPTED (ESCALATED)"
    elif action_type == "DISMISS":
        pdata["status"] = "DISMISSED"
        pdata["active_alert"] = None
    elif action_type == "DEFER":
        pdata["status"] = "DEFERRED (MONITORING)"
    elif action_type == "INVESTIGATE":
        pdata["status"] = "INVESTIGATING"

    return {"status": "success", "patient_status": pdata["status"], "feedback_id": feedback_id}


@app.get("/api/audit")
def get_audit_trail(
    patient_id: Optional[str] = Query(None),
    event_type: Optional[str] = Query(None),
    limit: int = Query(100),
) -> dict[str, Any]:
    init_system()
    assert audit is not None
    records = audit.query(patient_id=patient_id, event_type=event_type, limit=limit)
    valid, msg, count = audit.verify_integrity()
    return {
        "chain_valid": valid,
        "integrity_message": msg,
        "total_records": len(audit.records),
        "returned_records": [r.to_dict() for r in records],
        "head_hash": audit.records[-1].entry_hash if audit.records else None,
    }


@app.get("/api/audit/verify")
def verify_audit_chain() -> dict[str, Any]:
    init_system()
    assert audit is not None
    valid, msg, count = audit.verify_integrity()
    return {
        "chain_valid": valid,
        "message": msg,
        "records_count": count,
        "head_hash": audit.records[-1].entry_hash if audit.records else None,
    }


@app.get("/api/metrics")
def get_evaluation_metrics() -> dict[str, Any]:
    report_file = data_dir / "evaluation_report.json"
    if report_file.exists():
        try:
            return json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Run on demand if report is not found
    ev = CopilotEvaluator(data_dir)
    res = ev.run_evaluation(report_file)
    return res.to_dict()


@app.post("/api/simulate/step")
def simulate_step(count: int = 60) -> dict[str, Any]:
    """Steps the streaming simulation forward by `count` events."""
    global simulation_cursor
    init_system()
    assert engine is not None
    assert audit is not None

    if simulation_cursor >= len(all_vitals_events):
        return {"status": "complete", "message": "End of dataset reached", "processed": 0}

    chunk = all_vitals_events[simulation_cursor : simulation_cursor + count]
    simulation_cursor += len(chunk)

    alerts_triggered = 0
    for v_event in chunk:
        pid = v_event.get("patient_id")
        ts = v_event.get("event_time", "")
        payload = v_event.get("payload", {})
        v_dict = payload.get("vitals", {})
        seq = payload.get("sequence", 0)
        quality = payload.get("quality", "normal")

        reading = VitalReading.from_dict(v_dict, timestamp=ts, sequence=seq, quality=quality)
        reading.patient_id = pid

        vitals_history[pid].append(reading.to_dict())

        # Evaluate through policy engine
        res = engine.evaluate_reading(reading, patient_id=pid)
        res_dict = res.to_dict()
        latest_evaluation[pid] = res_dict

        # Update patient status
        pdata = patient_records[pid]
        pdata["latest_vitals"] = reading.to_dict()
        pdata["latest_news2"] = res.news2_score
        pdata["latest_si"] = res.shock_index

        if res.decision == AlertDecision.ALERT:
            alerts_triggered += 1
            pdata["status"] = f"ALERT ({res.priority.value if res.priority else 'HIGH'})"
            pdata["active_alert"] = res_dict
            audit.record("ALERT_DECISION", pid, res_dict, timestamp=ts)
        elif res.decision == AlertDecision.DEFER:
            if not pdata["status"].startswith("ALERT") and not pdata["status"].startswith("ACCEPTED"):
                pdata["status"] = "DEFERRED (MONITORING)"
                audit.record("ALERT_DECISION", pid, res_dict, timestamp=ts)
        elif res.decision == AlertDecision.SUPPRESS:
            if not pdata["status"].startswith("ACCEPTED") and not pdata["status"].startswith("INVESTIGATING"):
                if res.is_artifact:
                    audit.record("ARTIFACT_SUPPRESSED", pid, res_dict, timestamp=ts)
                elif res.cooldown_active:
                    audit.record("COOLDOWN_SUPPRESSED", pid, res_dict, timestamp=ts)

    return {
        "status": "success",
        "processed": len(chunk),
        "cursor": simulation_cursor,
        "total": len(all_vitals_events),
        "alerts_triggered": alerts_triggered,
    }


@app.post("/api/simulate/reset")
def simulate_reset() -> dict[str, Any]:
    global simulation_cursor, engine, audit, patient_records, vitals_history, latest_evaluation
    simulation_cursor = 0
    engine = None
    audit = None
    patient_records = {}
    vitals_history = {}
    latest_evaluation = {}
    init_system()
    return {"status": "success", "message": "Simulation reset to beginning"}
