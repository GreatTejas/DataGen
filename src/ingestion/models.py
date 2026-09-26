from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class VitalEvent(BaseModel):
    """
    Canonical live vital-sign event.

    These are the fields visible to the live copilot.
    Ground-truth/evaluation fields such as scenario,
    risk_label, trend_phase and episode_id must not
    be present here.
    """

    schema_version: str = "1.0"

    event_id: str
    event_time: datetime

    patient_id: str
    source: str
    run_id: str

    heart_rate: Optional[float] = None
    spo2: Optional[float] = None
    respiratory_rate: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None

    @field_validator(
        "event_id",
        "patient_id",
        "source",
        "run_id",
    )
    @classmethod
    def validate_string_fields(cls, value: str) -> str:

        value = value.strip()

        if not value:
            raise ValueError("field cannot be empty")

        return value

    @field_validator("event_time")
    @classmethod
    def normalize_event_time(
        cls,
        value: datetime,
    ) -> datetime:

        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)
