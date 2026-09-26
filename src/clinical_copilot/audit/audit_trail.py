"""Tamper-evident, cryptographically chained Audit Trail for Clinical Decision Support."""
from __future__ import annotations

import csv
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from ..schemas import AuditRecord


def canonical_json(data: Any) -> str:
    """Produces deterministic canonical JSON serialization for hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AuditTrail:
    """
    Append-only audit ledger with cryptographic SHA-256 chaining.
    Guarantees after-the-fact traceability and compliance verification.
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, log_path: Optional[str | Path] = None) -> None:
        self.records: list[AuditRecord] = []
        self.log_path = Path(log_path) if log_path else None
        if self.log_path and self.log_path.exists():
            self.load_from_file(self.log_path)

    def _compute_hash(
        self,
        prev_hash: str,
        index: int,
        timestamp: str,
        event_type: str,
        patient_id: str,
        payload_str: str,
    ) -> str:
        h = hashlib.sha256()
        h.update(prev_hash.encode("utf-8"))
        h.update(str(index).encode("utf-8"))
        h.update(timestamp.encode("utf-8"))
        h.update(event_type.encode("utf-8"))
        h.update(patient_id.encode("utf-8"))
        h.update(payload_str.encode("utf-8"))
        return h.hexdigest()

    def record(
        self,
        event_type: str,
        patient_id: str,
        payload: dict[str, Any],
        timestamp: Optional[str] = None,
        entry_id: Optional[str] = None,
    ) -> AuditRecord:
        """
        Appends a new immutable record to the audit chain.
        """
        now_ts = timestamp or datetime.now(UTC).isoformat()
        eid = entry_id or str(uuid.uuid4())
        index = len(self.records)
        prev_hash = self.records[-1].entry_hash if self.records else self.GENESIS_HASH
        payload_json = canonical_json(payload)
        entry_hash = self._compute_hash(prev_hash, index, now_ts, event_type, patient_id, payload_json)

        rec = AuditRecord(
            index=index,
            entry_id=eid,
            timestamp=now_ts,
            event_type=event_type,
            patient_id=patient_id,
            payload=payload,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self.records.append(rec)

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec.to_dict()) + "\n")

        return rec

    def verify_integrity(self) -> tuple[bool, str, int]:
        """
        Verifies the complete cryptographic chain from genesis to head.
        Returns (is_valid, message, verified_count).
        """
        if not self.records:
            return True, "Audit trail is empty; integrity valid by definition.", 0

        prev_hash = self.GENESIS_HASH
        for idx, rec in enumerate(self.records):
            if rec.index != idx:
                return False, f"Broken index sequencing at position {idx} (expected {idx}, got {rec.index})", idx
            if rec.prev_hash != prev_hash:
                return False, f"Broken chain link at position {idx}: prev_hash does not match preceding entry_hash.", idx

            payload_json = canonical_json(rec.payload)
            expected_hash = self._compute_hash(
                prev_hash,
                rec.index,
                rec.timestamp,
                rec.event_type,
                rec.patient_id,
                payload_json,
            )
            if rec.entry_hash != expected_hash:
                return False, f"Hash mismatch at position {idx}: entry tampered or payload altered.", idx

            prev_hash = rec.entry_hash

        return True, f"Cryptographic integrity verified. {len(self.records)} records intact.", len(self.records)

    def query(
        self,
        patient_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[AuditRecord]:
        """Queries records with optional filtering."""
        results = []
        for r in reversed(self.records):
            if patient_id and r.patient_id != patient_id:
                continue
            if event_type and r.event_type != event_type:
                continue
            results.append(r)
            if limit and len(results) >= limit:
                break
        return list(reversed(results))

    def load_from_file(self, path: Path) -> None:
        self.records = []
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.records.append(
                        AuditRecord(
                            index=d["index"],
                            entry_id=d["entry_id"],
                            timestamp=d["timestamp"],
                            event_type=d["event_type"],
                            patient_id=d["patient_id"],
                            payload=d["payload"],
                            prev_hash=d["prev_hash"],
                            entry_hash=d["entry_hash"],
                        )
                    )

    def export_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict()) + "\n")

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "entry_id", "timestamp", "event_type", "patient_id", "entry_hash", "prev_hash", "payload_summary"])
            for r in self.records:
                summary = str(r.payload)[:120].replace("\n", " ")
                writer.writerow([r.index, r.entry_id, r.timestamp, r.event_type, r.patient_id, r.entry_hash, r.prev_hash, summary])
