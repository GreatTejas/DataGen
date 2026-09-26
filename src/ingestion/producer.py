import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from confluent_kafka import Producer

KAFKA_BROKER = "localhost:9092"
VITALS_TOPIC = "vitals.raw"


def create_producer() -> Producer:

    return Producer(
        {
            "bootstrap.servers": KAFKA_BROKER,
            # Wait for all in-sync replicas.
            "acks": "all",
            # Small batching window.
            "linger.ms": 5,
            # Useful once the stream becomes larger.
            "compression.type": "snappy",
            # Retry transient failures.
            "retries": 5,
        }
    )


def delivery_report(err, message):

    if err is not None:

        print(
            f"[PRODUCER ERROR] "
            f"topic={message.topic()} "
            f"partition={message.partition()} "
            f"error={err}"
        )

        return

    print(
        f"[PRODUCED] "
        f"patient={message.key().decode()} "
        f"partition={message.partition()} "
        f"offset={message.offset()}"
    )


def publish_event(
    producer: Producer,
    event: dict,
):

    patient_id = event["patient_id"]

    producer.produce(
        topic=VITALS_TOPIC,
        # patient_id is the Kafka key.
        #
        # This ensures events for the same patient
        # are routed consistently to the same partition.
        key=patient_id,
        value=json.dumps(event).encode("utf-8"),
        callback=delivery_report,
    )

    # Process delivery callbacks.
    producer.poll(0)


def replay_jsonl(
    input_file: str,
    speed: float = 60.0,
):

    path = Path(input_file)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    producer = create_producer()

    print(f"[PRODUCER] Input: {path}")

    print(f"[PRODUCER] Broker: {KAFKA_BROKER}")

    print(f"[PRODUCER] Topic: {VITALS_TOPIC}")

    print(f"[PRODUCER] Replay speed: {speed}x")

    previous_event_time = None

    with path.open("r") as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                event = json.loads(line)

            except json.JSONDecodeError as exc:

                print(f"[SKIP] Invalid JSON " f"line={line_number}: {exc}")

                continue

            # -------------------------------------------------
            # Preserve event-time spacing during replay.
            # -------------------------------------------------

            current_event_time = event.get("event_time")

            if current_event_time is not None:

                try:

                    current_dt = datetime.fromisoformat(
                        current_event_time.replace(
                            "Z",
                            "+00:00",
                        )
                    )

                    if previous_event_time is not None:

                        delta = (current_dt - previous_event_time).total_seconds()

                        if delta > 0:

                            time.sleep(delta / speed)

                    previous_event_time = current_dt

                except Exception:

                    # Replay should continue even if
                    # timestamp parsing fails.
                    pass

            # -------------------------------------------------
            # Publish raw event.
            # -------------------------------------------------

            try:

                publish_event(
                    producer=producer,
                    event=event,
                )

            except BufferError:

                print("[PRODUCER] Local queue full. " "Waiting for delivery...")

                producer.flush()

                publish_event(
                    producer=producer,
                    event=event,
                )

    print("[PRODUCER] Flushing remaining messages...")

    producer.flush()

    print("[PRODUCER] Replay complete.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("Replay generated vital events " "into Kafka.")
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
