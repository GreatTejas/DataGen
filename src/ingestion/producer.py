import argparse
import json
import time
from pathlib import Path

from confluent_kafka import Producer

KAFKA_BROKER = "localhost:9092"
TOPIC = "vitals.raw"


def delivery_report(err, msg):
    """Called by Kafka when a message has been delivered."""

    if err is not None:
        print(
            f"[PRODUCER ERROR] "
            f"topic={msg.topic()} "
            f"partition={msg.partition()} "
            f"error={err}"
        )
        return

    print(
        f"[PRODUCED] "
        f"topic={msg.topic()} "
        f"partition={msg.partition()} "
        f"offset={msg.offset()} "
        f"key={msg.key().decode() if msg.key() else None}"
    )


def create_producer() -> Producer:
    return Producer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            # Reliable delivery for local development.
            "acks": "all",
            # Allow batching while still being fast.
            "linger.ms": 5,
            # Compression helps once event volume increases.
            "compression.type": "snappy",
            # Retry transient broker failures.
            "retries": 5,
        }
    )


def publish_event(
    producer: Producer,
    event: dict,
) -> None:

    patient_id = event["patient_id"]

    producer.produce(
        topic=TOPIC,
        # Critical:
        # patient_id determines Kafka partitioning.
        key=patient_id,
        value=json.dumps(event).encode("utf-8"),
        callback=delivery_report,
    )

    # Serve delivery callbacks.
    producer.poll(0)


def replay_jsonl(
    input_file: str,
    speed: float = 60.0,
) -> None:

    producer = create_producer()

    path = Path(input_file)

    if not path.exists():
        raise FileNotFoundError(path)

    print(f"[PRODUCER] Reading {path}")
    print(f"[PRODUCER] Kafka: {KAFKA_BROKER}")
    print(f"[PRODUCER] Topic: {TOPIC}")
    print(f"[PRODUCER] Replay speed: {speed}x")

    previous_event_time = None

    with path.open("r") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[SKIP] Invalid JSON at line " f"{line_number}: {e}")
                continue

            # Preserve the generator's event-time spacing.
            #
            # If the generator emitted:
            # 10:00:00
            # 10:00:05
            #
            # speed=60 means:
            # 5 simulated seconds
            # =
            # 5/60 real seconds
            current_event_time = event.get("event_time")

            if previous_event_time is not None and current_event_time is not None:
                try:
                    from datetime import datetime

                    current_dt = datetime.fromisoformat(
                        current_event_time.replace("Z", "+00:00")
                    )

                    delta = (current_dt - previous_event_time).total_seconds()

                    if delta > 0:
                        time.sleep(delta / speed)

                    previous_event_time = current_dt

                except Exception:
                    pass

            elif current_event_time is not None:
                try:
                    from datetime import datetime

                    previous_event_time = datetime.fromisoformat(
                        current_event_time.replace("Z", "+00:00")
                    )
                except Exception:
                    previous_event_time = None

            publish_event(
                producer=producer,
                event=event,
            )

            # Prevent local producer queue from growing forever.
            producer.poll(0)

    print("[PRODUCER] Flushing remaining events...")

    producer.flush()

    print("[PRODUCER] Replay complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Replay generated vital events into Kafka."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to generated JSONL file",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="Replay speed multiplier",
    )

    args = parser.parse_args()

    replay_jsonl(
        input_file=args.input,
        speed=args.speed,
    )
