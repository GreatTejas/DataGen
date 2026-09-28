"""Run the REAL P1 (TrendAgent) + P2 (PatientStateManager, AnomalyAgent)
code against Person 1's generated data, then feed the result into P4's
RAG + Clinical Reasoning Agent. Risk/Evidence Fusion (P3) is still a
placeholder — see clinical_reasoning/integration.py.

This is the strongest available integration check today: it does not
re-implement trend or anomaly detection, it imports and calls the
actual P1/P2 modules.

Usage:
    python scripts/run_real_pipeline_demo.py \
        --datagen-root /path/to/DataGen-main \
        --patient-id SYN-0002 \
        --corpus knowledge/knowledge_corpus.jsonl \
        --fake-llm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datagen-root",
        required=True,
        help="Path to the DataGen-main checkout (contains src/, trend_agent/, data/generated/)",
    )
    parser.add_argument("--patient-id", default=None, help="Defaults to the first ground-truth episode's patient")
    parser.add_argument("--corpus", default="knowledge/knowledge_corpus.jsonl")
    parser.add_argument("--fake-llm", action="store_true")
    args = parser.parse_args()

    datagen_root = Path(args.datagen_root).resolve()
    sys.path.insert(0, str(datagen_root / "src"))
    sys.path.insert(0, str(datagen_root))

    src_root = Path(__file__).resolve().parent.parent / "src"
    sys.path.insert(0, str(src_root))

    from clinical_data_gen.anomaly.anomaly_agent import AnomalyAgent
    from clinical_data_gen.state.patient_state_manager import PatientStateManager
    from trend_agent.trend_agent import TrendAgent

    from clinical_reasoning.experience_memory_client import StubExperienceMemoryClient
    from clinical_reasoning.integration import build_consolidated_evidence
    from clinical_reasoning.knowledge_store import RAGRetriever
    from clinical_reasoning.reasoning_agent import ClinicalReasoningAgent

    data_dir = datagen_root / "data" / "generated"
    vitals_rows = _load_jsonl(data_dir / "vitals_observed.jsonl")
    context_rows = _load_jsonl(data_dir / "patient_context.jsonl")
    episodes = _load_jsonl(data_dir / "ground_truth_episodes.jsonl")

    patient_id = args.patient_id or (episodes[0]["patient_id"] if episodes else None)
    if patient_id is None:
        raise SystemExit("no --patient-id given and no ground-truth episodes found")

    episode = next((e for e in episodes if e["patient_id"] == patient_id), None)
    as_of = episode["escalation_window_start"] if episode else None

    patient_rows = sorted(
        (r for r in vitals_rows if r["patient_id"] == patient_id),
        key=lambda r: r["event_time"],
    )
    if as_of is not None:
        patient_rows = [r for r in patient_rows if r["event_time"] <= as_of]
    if not patient_rows:
        raise SystemExit(f"no readings for {patient_id}")

    context_row = next((r for r in context_rows if r["patient_id"] == patient_id), None)
    if context_row is None:
        raise SystemExit(f"no patient_context row for {patient_id}")

    # --- Real P1 TrendAgent, fed the real event envelopes ---
    trend_agent = TrendAgent()
    trend_agent.observe_context({"patient_id": patient_id})
    trend_output = None
    for row in patient_rows:
        trend_output = trend_agent.observe(row)

    # --- Real P2 PatientStateManager + AnomalyAgent ---
    state_manager = PatientStateManager()
    anomaly_agent = AnomalyAgent()
    from clinical_reasoning.integration import update_patient_state

    state = None
    for row in patient_rows:
        state = update_patient_state(
            state_manager, patient_id, row["event_time"], row["payload"]["vitals"]
        )
    anomaly_result = anomaly_agent.predict(state)

    print(f"# Real pipeline run: {patient_id}, {len(patient_rows)} readings, "
          f"as_of={patient_rows[-1]['event_time']}\n")
    print("## Real TrendAgent output (Person 1)")
    print(json.dumps(trend_output, indent=2, sort_keys=True))
    print("\n## Real AnomalyResult (Person 2) — untrained cold-start model, "
          "so probability is near 0.5 with zero prior feedback updates")
    print(json.dumps(
        {
            "patient_id": anomaly_result.patient_id,
            "probability": anomaly_result.probability,
            "is_anomalous": anomaly_result.is_anomalous,
            "contributing_features": anomaly_result.contributing_features,
            "model_version": anomaly_result.model_version,
        },
        indent=2,
    ))

    latest_vitals = patient_rows[-1]["payload"]["vitals"]
    evidence = build_consolidated_evidence(
        evidence_id=f"REALEVID-{patient_id}-{patient_rows[-1]['event_id'][:8]}",
        patient_id=patient_id,
        window_start=patient_rows[0]["event_time"],
        window_end=patient_rows[-1]["event_time"],
        generated_vitals=latest_vitals,
        trend_output=trend_output,
        anomaly_result=anomaly_result,
        patient_context_payload=context_row["payload"],
    )
    print("\n## ConsolidatedEvidence assembled by P4 (risk is still a placeholder — see conflicts_or_missing)")
    print(json.dumps(evidence.to_dict(), indent=2))

    retriever = RAGRetriever.from_jsonl(args.corpus)
    client = None
    if args.fake_llm:
        from clinical_reasoning.fake_llm import FakeAnthropicClient

        client = FakeAnthropicClient()
    agent = ClinicalReasoningAgent(
        retriever=retriever, experience_client=StubExperienceMemoryClient(), client=client
    )
    output = agent.run(evidence)
    print("\n## Clinical Reasoning Agent output (Person 4)")
    print(json.dumps(output.to_dict(), indent=2))


if __name__ == "__main__":
    main()
