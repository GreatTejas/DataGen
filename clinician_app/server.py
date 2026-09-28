"""Local Person 5 clinician interface, observed-only policy, and audit endpoints."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinician_app.evaluation import calculate_metrics
from clinician_app.policy import CHANNELS, evaluate_stream

DATA = ROOT / "data" / "generated"
AUDIT = ROOT / "data" / "evaluation" / "clinician_audit.jsonl"
FEEDBACK = ROOT / "data" / "evaluation" / "clinician_feedback.jsonl"
PORT = int(os.environ.get("CLINICIAN_APP_PORT", "8765"))
POLICY_VERSION = "person5-observed-policy-v6"
LAST_SUGGESTIONS: dict[str, dict[str, str]] = {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def input_data() -> tuple[list[dict[str, Any]], dict[str, dict], list[dict], dict[str, dict]]:
    events = read_jsonl(DATA / "vitals_observed.jsonl")
    contexts = {x["patient_id"]: x.get("payload", {}) for x in read_jsonl(DATA / "patient_context.jsonl")}
    episodes = read_jsonl(DATA / "ground_truth_episodes.jsonl")
    labels = {x["event_id"]: x for x in read_jsonl(DATA / "evaluation_labels.jsonl")}
    return events, contexts, episodes, labels


def trend_summary(history: list[dict[str, Any]], key: str) -> float:
    prior, recent = history[-20:-10], history[-10:]
    if not prior or not recent:
        return 0.0
    mean = lambda rows: sum(float(x.get(key, 0)) for x in rows) / len(rows)
    return round(mean(recent) - mean(prior), 1)


def make_audit_traces(events: list[dict], decisions: dict[str, list[dict]], contexts: dict[str, dict]) -> list[dict]:
    """Create one idempotent policy/observation trace per generated reading."""
    old_audit = read_jsonl(AUDIT)
    current_event_ids = {x.get("event_id") for x in events}
    stale = [x for x in old_audit if x.get("event_type") == "policy_evaluation"
             and x.get("event_id") in current_event_ids and x.get("policy_version") != POLICY_VERSION]
    if stale:
        # Replace obsolete outputs from an earlier version of this mock policy while preserving clinician actions.
        stale_ids = {x.get("event_id") for x in stale}
        preserved = [x for x in old_audit if not (x.get("event_type") == "policy_evaluation" and x.get("event_id") in stale_ids)]
        temporary = AUDIT.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in preserved), encoding="utf-8")
        temporary.replace(AUDIT)
        old_audit = preserved
    existing = {x.get("event_id") for x in old_audit if x.get("event_type") == "policy_evaluation" and x.get("policy_version") == POLICY_VERSION}
    traces = []
    event_groups: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        event_groups[event["patient_id"]].append(event)
    for patient_id, rows in event_groups.items():
        rows.sort(key=lambda x: x["event_time"])
        decision_by_id = {d.get("event_id"): d for d in decisions.get(patient_id, [])}
        recent = []
        for event in rows:
            event_id = event.get("event_id")
            if event_id in existing:
                recent = (recent + [{"time": event["event_time"], **event.get("payload", {}).get("vitals", {})}])[-20:]
                continue
            decision = decision_by_id.get(event_id)
            if not decision:
                continue
            payload = event.get("payload", {})
            observed = payload.get("vitals", {})
            recent = (recent + [{"time": event["event_time"], **observed}])[-20:]
            trend = {name: trend_summary(recent, name) for name in CHANNELS}
            traces.append({
                "event_type": "policy_evaluation", "timestamp": event["event_time"],
                "event_id": event_id, "patient_id": patient_id, "policy_version": POLICY_VERSION,
                "run_id": event.get("run_id"),
                "observation": {"vitals": observed, "quality": payload.get("quality", "unknown"), "schema_version": event.get("schema_version")},
                "patient_context": contexts.get(patient_id, {}),
                "agent_outputs": {"trend_summary_demo": trend, "person5_baseline_rule_score": decision["evidence"], "anomaly_agent": None, "risk_assessor": None},
                "retrieved_evidence": [], "citations": [],
                "reasoning": decision["policy_reason"],
                "alert": {"decision": decision["policy"], "priority": decision["priority"], "evidence": decision["evidence"].get("flags", [])},
                "source": "generated_mock_replay", "mock_data": True,
            })
    append_jsonl(AUDIT, traces)
    return traces


def dashboard() -> dict[str, Any]:
    events, contexts, episodes, labels = input_data()
    if not events:
        return {"cases": [], "metrics": {}, "audit": [], "mode": "Demo reasoning", "patient_count": 0,
                "error": "No generated mock data found. Generate a cohort first, then refresh."}
    decisions = evaluate_stream(events)
    make_audit_traces(events, decisions, contexts)
    event_groups: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        event_groups[event["patient_id"]].append(event)
    action_by_patient = {}
    clinician_records = [x for x in read_jsonl(AUDIT) if x.get("event_type") == "clinician_decision"]
    for row in clinician_records:
        action_by_patient[row.get("patient_id")] = row
    cases = []
    for patient_id, stream in event_groups.items():
        stream.sort(key=lambda x: x["event_time"])
        latest = stream[-1]
        latest_decision = next((x for x in reversed(decisions[patient_id]) if x.get("event_id") == latest.get("event_id")), decisions[patient_id][-1])
        history = [{"time": x["event_time"], **x["payload"]["vitals"]} for x in stream[-30:]]
        context = contexts.get(patient_id, {})
        baseline = latest_decision["baseline"]
        flags = latest_decision["evidence"].get("flags", [])
        cases.append({
            "patient_id": patient_id, "event_id": latest.get("event_id"), "event_time": latest["event_time"],
            "vitals": latest["payload"]["vitals"], "baseline": baseline, "history": history, "context": context,
            "priority": latest_decision["priority"], "policy": latest_decision["policy"],
            "policy_reason": latest_decision["policy_reason"], "factors": flags or ["No sustained multi-parameter change detected"],
            "explanation": "Observed-data summary: " + ("; ".join(flags) if flags else "recent vital signs do not meet the persistent escalation rule."),
            "recommendation": "Review the recent observations and follow local escalation procedures when clinically indicated.",
            "quality": latest.get("payload", {}).get("quality", "unknown"),
            "last_action": action_by_patient.get(patient_id),
        })
    cases.sort(key=lambda x: ({"High": 0, "Medium": 1, "Low": 2}.get(x["priority"], 3), x["patient_id"]))
    traces = [x for x in read_jsonl(AUDIT) if x.get("event_type") == "policy_evaluation"]
    metrics, _ = calculate_metrics(decisions, episodes, labels, traces)
    return {"cases": cases, "metrics": metrics, "audit": clinician_records[-12:],
            "mode": "Gemini connected" if os.getenv("GEMINI_API_KEY") else "Demo reasoning",
            "patient_count": len(event_groups)}


CLINICIAN_ACTIONS = ("Accept", "Dismiss", "Defer", "Investigate")


def gemini_suggestion(case: dict[str, Any]) -> dict[str, Any]:
    """Recommend one clinician-interface action from the visible patient evidence."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        policy = case.get("policy", "SUPPRESS")
        suggested = {"ALERT": "Accept", "DEFER": "Defer", "SUPPRESS": "Dismiss"}.get(policy, "Investigate")
        if policy == "ALERT" and case.get("priority") != "High":
            suggested = "Investigate"
        if case.get("quality") == "suspect":
            suggested = "Investigate"
        return {"suggested_action": suggested,
                "reason": f"Demo rule fallback: policy is {policy} ({case.get('priority', 'Low')} priority). "
                          f"{case.get('policy_reason', 'Review the displayed patient evidence before deciding.')}",
                "provider": "Demo rule fallback"}
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.7-flash")
    prompt = (
        "You help a clinician review generated mock patient observations. Use only the supplied patient data and alert evidence. "
        "Choose exactly one action from Accept, Dismiss, Defer, Investigate. Accept endorses the displayed alert for clinician review; "
        "Dismiss means the current evidence does not support it; Defer means wait for more readings; Investigate means verify data quality "
        "or request closer human review. Missing or suspect data should favor Investigate or Defer. Do not diagnose or prescribe. "
        "Return exactly two lines, with no markdown: ACTION: <one action> and REASON: <one concise evidence-based sentence>.\n\n"
        "Patient data:\n" + json.dumps(case, ensure_ascii=False) + "\n\nChoose the action now."
    )
    # Plain text avoids the JSON response-format enum rejected by this API endpoint.
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": 512,
                                            "thinkingConfig": {"thinkingLevel": "low"}}}).encode()
    # Retry transient capacity/rate errors on the primary, then use the currently
    # supported previous Flash model if the primary remains overloaded.
    attempts = [model, model] + ([fallback_model] if fallback_model != model else [])
    last_error: Exception | None = None
    for attempt_index, attempt_model in enumerate(attempts):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{attempt_model}:generateContent"
        request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                data = json.loads(response.read())
            answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            action_match = re.search(r"(?im)^\s*ACTION\s*:\s*(Accept|Dismiss|Defer|Investigate)\s*[.!]?\s*$", answer)
            reason_match = re.search(r"(?im)^\s*REASON\s*:\s*(.+?)\s*$", answer)
            if not action_match:
                raise ValueError("Gemini did not return a valid ACTION line. Try again.")
            reason = reason_match.group(1).strip() if reason_match else answer[:1000]
            return {"suggested_action": action_match.group(1), "reason": reason, "provider": attempt_model}
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            try:
                details = json.loads(details).get("error", {}).get("message", details)
            except json.JSONDecodeError:
                pass
            last_error = RuntimeError(f"Gemini HTTP {exc.code}: {details[:1000]}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt_index == len(attempts) - 1:
                raise last_error from exc
            time.sleep(1 if attempt_index == 0 else 0.5)
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
    raise RuntimeError(str(last_error or "Gemini request failed"))


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/api/dashboard":
            self.send_json(dashboard())
            return
        if self.path == "/":
            raw = (ROOT / "clinician_app" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return
        if self.path == "/api/action":
            action, patient_id = body.get("action"), body.get("patient_id")
            if action not in {"Accept", "Dismiss", "Defer", "Investigate"} or not patient_id:
                self.send_json({"error": "A patient and valid action are required"}, 400)
                return
            current = next((case for case in dashboard()["cases"] if case["patient_id"] == patient_id), None)
            if current is None:
                self.send_json({"error": "Patient is not in the current mock cohort"}, 404)
                return
            now = datetime.now().astimezone().isoformat()
            reason = str(body.get("reason", ""))[:1000]
            suggestion = body.get("gemini_suggestion") or LAST_SUGGESTIONS.get(patient_id)
            if not isinstance(suggestion, dict) or suggestion.get("suggested_action") not in CLINICIAN_ACTIONS:
                suggestion = None
            elif suggestion:
                suggestion = {key: suggestion.get(key) for key in ("suggested_action", "reason", "provider")}
            decision = {"event_type": "clinician_decision", "timestamp": now, "patient_id": patient_id,
                        "event_id": current["event_id"], "action": action, "reason": reason,
                        "observation": current["vitals"], "agent_outputs": {"alert_policy": {"decision": current["policy"], "priority": current["priority"]}},
                        "retrieved_evidence": [], "citations": [], "reasoning": current["explanation"],
                        "gemini_suggestion": suggestion,
                        "alert": {"decision": current["policy"], "priority": current["priority"]}, "source": "clinician_interface_demo", "mock_data": True}
            append_jsonl(AUDIT, [decision])
            feedback = {"timestamp": now, "patient_id": patient_id, "event_id": current["event_id"],
                        "features": current["vitals"], "alert_policy_decision": current["policy"],
                        "gemini_suggestion": suggestion, "clinician_action": action, "feedback": reason,
                        "source": "clinician_interface_demo", "mock_data": True}
            append_jsonl(FEEDBACK, [feedback])
            self.send_json({"ok": True, "record": decision})
            return
        if self.path == "/api/reason":
            try:
                case = body["case"]
                suggestion = gemini_suggestion(case)
                if case.get("patient_id"):
                    LAST_SUGGESTIONS[case["patient_id"]] = suggestion
                now = datetime.now().astimezone().isoformat()
                append_jsonl(AUDIT, [{"event_type": "reasoning_generated", "timestamp": now,
                    "event_id": case.get("event_id"), "patient_id": case.get("patient_id"),
                    "observation": case.get("vitals", {}), "agent_outputs": {"alert_policy": {"decision": case.get("policy"), "priority": case.get("priority")}},
                    "retrieved_evidence": [], "citations": [], "reasoning": suggestion["reason"],
                    "alert": {"decision": case.get("policy"), "priority": case.get("priority")},
                    "suggested_action": suggestion["suggested_action"], "model": suggestion["provider"],
                    "source": "clinician_interface_demo", "mock_data": True}])
                self.send_json({**suggestion, "mode": "Gemini" if os.getenv("GEMINI_API_KEY") else "Demo"})
            except (RuntimeError, KeyError) as exc:
                self.send_json({"error": str(exc)}, 502)
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[clinician-app] {fmt % args}")


if __name__ == "__main__":
    print(f"Clinician demo: http://127.0.0.1:{PORT}")
    print("Generated mock data only. Set GEMINI_API_KEY to enable Gemini summaries.")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
