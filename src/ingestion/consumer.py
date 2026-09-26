import json
import signal
from datetime import datetime, timezone

from confluent_kafka import Consumer, Producer, KafkaError

from models import VitalEvent, ValidatedVitalEvent

KAFKA_BROKER = "localhost:9092"

INPUT_TOPIC = "vitals.raw"
VALIDATED_TOPIC = "vitals.validated"
INVALID_TOPIC = "vitals.invalid"

GROUP_ID = "clinical-ingestion-v1"


running = True


def shutdown_handler(signum, frame):
    global running

    print("\n[CONSUMER] Shutdown requested.")

    running = False


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# ---------------------------------------------------------
# Kafka
# ---------------------------------------------------------


def create_consumer() -> Consumer:

    return Consumer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            "group.id": GROUP_ID,
            # We manually commit after successful processing.
            "enable.auto.commit": False,
            # Start from beginning for a new local consumer group.
            "auto.offset.reset": "earliest",
            # Prevent excessively large batches.
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


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------


def validate_ranges(event: VitalEvent) -> list[str]:
    """
    Data-quality validation.

    IMPORTANT:
    These are NOT clinical risk thresholds.

    A clinically concerning value can still be valid data.
    """

    errors = []

    if event.heart_rate is not None:
        if event.heart_rate <= 0 or event.heart_rate > 300:
            errors.append(f"invalid heart_rate={event.heart_rate}")

    if event.spo2 is not None:
        if event.spo2 < 0 or event.spo2 > 100:
            errors.append(f"invalid spo2={event.spo2}")

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


def validate_timestamp(event: VitalEvent) -> list[str]:

    errors = []

    now = datetime.now(timezone.utc)

    # Don't accept events more than 5 minutes
    # into the future.
    future_seconds = (event.event_time - now).total_seconds()

    if future_seconds > 300:
        errors.append("event_time is too far in the future")

    return errors


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------


def normalize_event(event: VitalEvent) -> VitalEvent:
    """
    Keep canonical representation in one place.

    For the current generator, most of the work is already
    done by the Pydantic model.
    """

    return event


# ---------------------------------------------------------
# Invalid event
# ---------------------------------------------------------


def publish_invalid(
    producer: Producer,
    raw_event: dict,
    errors: list[str],
) -> None:

    payload = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "errors": errors,
        "raw_event": raw_event,
    }

    producer.produce(
        topic=INVALID_TOPIC,
        key=str(
            raw_event.get(
                "patient_id",
                "unknown",
            )
        ),
        value=json.dumps(payload).encode("utf-8"),
    )

    producer.poll(0)


# ---------------------------------------------------------
# Valid event
# ---------------------------------------------------------


def publish_validated(
    producer: Producer,
    event: VitalEvent,
) -> None:

    ingested_at = datetime.now(timezone.utc)

    latency_ms = (ingested_at - event.event_time).total_seconds() * 1000

    validated_event = ValidatedVitalEvent(
        **event.model_dump(),
        ingested_at=ingested_at,
        ingestion_latency_ms=latency_ms,
        quality_valid=True,
        quality_flags=[],
        normalization_version="1.0",
    )

    producer.produce(
        topic=VALIDATED_TOPIC,
        key=validated_event.patient_id,
        value=validated_event.model_dump_json().encode("utf-8"),
    )

    producer.poll(0)


# ---------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------


def run():

    consumer = create_consumer()
    producer = create_producer()

    consumer.subscribe([INPUT_TOPIC])

    print(f"[CONSUMER] Kafka: {KAFKA_BROKER}")

    print(f"[CONSUMER] Input: {INPUT_TOPIC}")

    print(f"[CONSUMER] Validated: " f"{VALIDATED_TOPIC}")

    print(f"[CONSUMER] Invalid: " f"{INVALID_TOPIC}")

    try:

        while running:

            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():

                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue

                print(f"[KAFKA ERROR] " f"{msg.error()}")

                continue

            # -------------------------------------------------
            # 1. Deserialize
            # -------------------------------------------------

            try:

                raw_event = json.loads(msg.value().decode("utf-8"))

            except Exception as e:

                print(
                    f"[INVALID JSON] "
                    f"partition={msg.partition()} "
                    f"offset={msg.offset()} "
                    f"error={e}"
                )

                publish_invalid(
                    producer=producer,
                    raw_event={
                        "raw": msg.value().decode(
                            "utf-8",
                            errors="replace",
                        )
                    },
                    errors=[f"invalid JSON: {e}"],
                )

                # Commit because we intentionally
                # handled the poison message.
                consumer.commit(
                    message=msg,
                    asynchronous=False,
                )

                continue

            # -------------------------------------------------
            # 2. Schema validation
            # -------------------------------------------------

            try:

                event = VitalEvent.model_validate(raw_event)

            except Exception as e:

                errors = [f"schema validation failed: {e}"]

                print(f"[SCHEMA INVALID] " f"{errors}")

                publish_invalid(
                    producer=producer,
                    raw_event=raw_event,
                    errors=errors,
                )

                consumer.commit(
                    message=msg,
                    asynchronous=False,
                )

                continue

            # -------------------------------------------------
            # 3. Data-quality validation
            # -------------------------------------------------

            errors = []

            errors.extend(validate_ranges(event))

            errors.extend(validate_timestamp(event))

            if errors:

                print(
                    f"[VALIDATION FAILED] "
                    f"patient={event.patient_id} "
                    f"event={event.event_id} "
                    f"errors={errors}"
                )

                publish_invalid(
                    producer=producer,
                    raw_event=raw_event,
                    errors=errors,
                )

                consumer.commit(
                    message=msg,
                    asynchronous=False,
                )

                continue

            # -------------------------------------------------
            # 4. Normalize
            # -------------------------------------------------

            event = normalize_event(event)

            # -------------------------------------------------
            # 5. Publish validated event
            # -------------------------------------------------

            publish_validated(
                producer=producer,
                event=event,
            )

            # -------------------------------------------------
            # 6. Commit ONLY after successful processing
            # -------------------------------------------------

            consumer.commit(
                message=msg,
                asynchronous=False,
            )

            print(
                f"[VALID] "
                f"patient={event.patient_id} "
                f"event={event.event_id} "
                f"HR={event.heart_rate} "
                f"SpO2={event.spo2} "
                f"RR={event.respiratory_rate} "
                f"BP={event.systolic_bp}/"
                f"{event.diastolic_bp}"
            )

    finally:

        print("[CONSUMER] Flushing producer...")

        producer.flush()

        print("[CONSUMER] Closing consumer...")

        consumer.close()

        print("[CONSUMER] Shutdown complete.")


if __name__ == "__main__":
    run()
