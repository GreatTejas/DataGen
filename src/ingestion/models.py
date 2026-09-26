from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class VitalEvent(BaseModel):
    """
    Canonical event produced by the synthetic data generator
    and consumed from Kafka.

    This represents ONLY fields visible to the live copilot.
    Ground-truth/evaluation fields must not be included.
    """

    schema_version: str = "1.0"

    event_id: str
    event_time: datetime

    patient_id: str
    source: str
    run_id: str

    heart_rate: Optional[float] = Field(default=None)
    spo2: Optional[float] = Field(default=None)
    respiratory_rate: Optional[float] = Field(default=None)
    systolic_bp: Optional[float] = Field(default=None)
    diastolic_bp: Optional[float] = Field(default=None)

    @field_validator("event_id", "patient_id", "source", "run_id")
    @classmethod
    def must_not_be_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value

    @field_validator("event_time")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)


class ValidatedVitalEvent(VitalEvent):
    """
    Event emitted by the ingestion layer after validation/QC.
    """

    ingested_at: datetime
    ingestion_latency_ms: float

    quality_valid: bool = True
    quality_flags: list[str] = Field(default_factory=list)

    normalization_version: str = "1.0"
