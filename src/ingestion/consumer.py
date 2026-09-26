import json
import signal
from datetime import datetime, timezone

from confluent_kafka import (
    Consumer,
    Producer,
    KafkaError,
)

from pydantic import ValidationError

from models import VitalEvent

KAFKA_BROKER = "localhost:9092"

INPUT_TOPIC = "vitals.raw"
DLQ_TOPIC = "vitals.dlq"

GROUP_ID = "clinical-ingestion-v1"

running = True


def shutdown_handler(signum, frame):

    global running

    print("\n[CONSUMER] Shutdown requested.")

    running = False


signal.signal(
    signal.SIGINT,
    shutdown_handler,
)

signal.signal(
    signal.SIGTERM,
    shutdown_handler,
)


# =========================================================
# Kafka
# =========================================================


def create_consumer() -> Consumer:

    return Consumer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            "group.id": GROUP_ID,
            # We commit manually after processing.
            "enable.auto.commit": False,
            # Useful for local development.
            "auto.offset.reset": "earliest",
            # Maximum time between poll calls.
            "max.poll.interval.ms": 300000,
        }
    )


def create_producer() -> Producer:

    return Producer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            "acks": "all",
            "compression.type": "snappy",
        }
    )


# =========================================================
# Validation
# =========================================================


def validate_ranges(
    event: VitalEvent,
) -> list[str]:

    errors = []

    # These are DATA QUALITY limits.
    #
    # They are NOT clinical risk thresholds.

    if event.heart_rate is not None:

        if event.heart_rate <= 0 or event.heart_rate > 300:

            errors.append(f"invalid heart_rate=" f"{event.heart_rate}")

    if event.spo2 is not None:

        if event.spo2 < 0 or event.spo2 > 100:

            errors.append(f"invalid spo2=" f"{event.spo2}")

    if event.respiratory_rate is not None:

        if event.respiratory_rate <= 0 or event.respiratory_rate > 150:

            errors.append(f"invalid respiratory_rate=" f"{event.respiratory_rate}")

    if event.systolic_bp is not None:

        if event.systolic_bp <= 0 or event.systolic_bp > 300:

            errors.append(f"invalid systolic_bp=" f"{event.systolic_bp}")

    if event.diastolic_bp is not None:

        if event.diastolic_bp <= 0 or event.diastolic_bp > 200:

            errors.append(f"invalid diastolic_bp=" f"{event.diastolic_bp}")

    return errors


def validate_timestamp(
    event: VitalEvent,
) -> list[str]:

    errors = []

    now = datetime.now(timezone.utc)

    future_seconds = (event.event_time - now).total_seconds()

    # Reject events more than 5 minutes
    # into the future.
    if future_seconds > 300:

        errors.append("event_time is too far in the future")

    return errors


# =========================================================
# Normalization
# =========================================================


def normalize_event(
    event: VitalEvent,
) -> VitalEvent:

    # Pydantic has already normalized event_time
    # to UTC.
    #
    # Keep this function as the boundary where
    # future source-specific normalization can go.

    return event


# =========================================================
# Quality Control
# =========================================================


def quality_control(
    event: VitalEvent,
) -> list[str]:

    flags = []

    # Missing values are not automatically invalid.
    #
    # A monitor may legitimately omit a channel.
    # Downstream components can decide how to handle
    # missing observations.

    if event.heart_rate is None:
        flags.append("MISSING_HEART_RATE")

    if event.spo2 is None:
        flags.append("MISSING_SPO2")

    if event.respiratory_rate is None:
        flags.append("MISSING_RESPIRATORY_RATE")

    if event.systolic_bp is None:
        flags.append("MISSING_SYSTOLIC_BP")

    if event.diastolic_bp is None:
        flags.append("MISSING_DIASTOLIC_BP")

    return flags


# =========================================================
# DLQ
# =========================================================


def publish_dlq(
    producer: Producer,
    raw_event,
    errors: list[str],
):

    payload = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "raw_event": raw_event,
    }

    patient_id = "unknown"

    if isinstance(raw_event, dict):

        patient_id = str(
            raw_event.get(
                "patient_id",
                "unknown",
            )
        )

    producer.produce(
        topic=DLQ_TOPIC,
        key=patient_id,
        value=json.dumps(payload).encode("utf-8"),
    )

    producer.poll(0)


# =========================================================
# Downstream interface
# =========================================================


def process_valid_event(
    event: VitalEvent,
):
    """
    Downstream handoff.

    For now this is just a placeholder.

    Later this should call the Patient State
    Manager / downstream component.

    Example:

        patient_state_manager.update(event)
    """

    print(
        "[DOWNSTREAM] "
        f"patient={event.patient_id} "
        f"event={event.event_id} "
        f"HR={event.heart_rate} "
        f"SpO2={event.spo2} "
        f"RR={event.respiratory_rate} "
        f"BP={event.systolic_bp}/"
        f"{event.diastolic_bp}"
    )


# =========================================================
# Main consumer loop
# =========================================================


def run():

    consumer = create_consumer()

    dlq_producer = create_producer()

    consumer.subscribe([INPUT_TOPIC])

    print(f"[CONSUMER] Broker: " f"{KAFKA_BROKER}")

    print(f"[CONSUMER] Input: " f"{INPUT_TOPIC}")

    print(f"[CONSUMER] DLQ: " f"{DLQ_TOPIC}")

    try:

        while running:

            message = consumer.poll(timeout=1.0)

            if message is None:
                continue

            if message.error():

                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue

                print(f"[KAFKA ERROR] " f"{message.error()}")

                continue

            # =================================================
            # 1. Deserialize
            # =================================================

            try:

                raw_event = json.loads(message.value().decode("utf-8"))

            except Exception as exc:

                errors = [f"invalid JSON: {exc}"]

                print("[INVALID JSON] " f"{errors}")

                publish_dlq(
                    producer=dlq_producer,
                    raw_event={
                        "raw": message.value().decode(
                            "utf-8",
                            errors="replace",
                        )
                    },
                    errors=errors,
                )

                # Poison message has been handled.
                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                continue

            # =================================================
            # 2. Schema validation
            # =================================================

            try:

                event = VitalEvent.model_validate(raw_event)

            except ValidationError as exc:

                errors = [
                    "schema validation failed",
                    str(exc),
                ]

                print(
                    "[SCHEMA INVALID] "
                    f"patient="
                    f"{raw_event.get('patient_id', 'unknown')} "
                    f"errors={errors}"
                )

                publish_dlq(
                    producer=dlq_producer,
                    raw_event=raw_event,
                    errors=errors,
                )

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                continue

            # =================================================
            # 3. Range validation
            # =================================================

            errors = validate_ranges(event)

            # =================================================
            # 4. Timestamp validation
            # =================================================

            errors.extend(validate_timestamp(event))

            if errors:

                print(
                    "[VALIDATION FAILED] "
                    f"patient={event.patient_id} "
                    f"event={event.event_id} "
                    f"errors={errors}"
                )

                publish_dlq(
                    producer=dlq_producer,
                    raw_event=raw_event,
                    errors=errors,
                )

                consumer.commit(
                    message=message,
                    asynchronous=False,
                )

                continue

            # =================================================
            # 5. Normalize
            # =================================================

            event = normalize_event(event)

            # =================================================
            # 6. Quality control
            # =================================================

            quality_flags = quality_control(event)

            if quality_flags:

                print(
                    "[QUALITY FLAGS] "
                    f"patient={event.patient_id} "
                    f"event={event.event_id} "
                    f"flags={quality_flags}"
                )

            # =================================================
            # 7. Downstream handoff
            # =================================================

            try:

                process_valid_event(event)

            except Exception as exc:

                # IMPORTANT:
                # Do NOT commit if downstream processing
                # fails.
                #
                # Kafka can then redeliver the message.

                print(
                    "[DOWNSTREAM ERROR] "
                    f"patient={event.patient_id} "
                    f"event={event.event_id} "
                    f"error={exc}"
                )

                continue

            # =================================================
            # 8. Commit after successful processing
            # =================================================

            consumer.commit(
                message=message,
                asynchronous=False,
            )

    finally:

        print("[CONSUMER] Flushing DLQ producer...")

        dlq_producer.flush()

        print("[CONSUMER] Closing consumer...")

        consumer.close()

        print("[CONSUMER] Shutdown complete.")


if __name__ == "__main__":
    run()
