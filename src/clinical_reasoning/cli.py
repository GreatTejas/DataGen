"""CLI for the Clinical Knowledge (RAG) + Clinical Reasoning component.

    clinical-reasoning query --corpus knowledge/knowledge_corpus.jsonl \
        --q "falling blood pressure rising heart rate" --top-k 3

    clinical-reasoning mock-evidence --data-dir data/generated \
        --patient-id SYN-0002 --at 2026-09-18T10:05:00+00:00

    clinical-reasoning reason --data-dir data/generated \
        --corpus knowledge/knowledge_corpus.jsonl \
        --patient-id SYN-0002 --at 2026-09-18T10:05:00+00:00 --fake-llm

    clinical-reasoning demo --data-dir data/generated \
        --corpus knowledge/knowledge_corpus.jsonl --fake-llm
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .knowledge_store import RAGRetriever
from .mock_evidence import build_mock_evidence
from .reasoning_agent import ClinicalReasoningAgent


def _cmd_query(args: argparse.Namespace) -> None:
    retriever = RAGRetriever.from_jsonl(args.corpus)
    results = retriever.retrieve(args.q, top_k=args.top_k)
    for r in results:
        print(
            json.dumps(
                {
                    "document_id": r.document.document_id,
                    "title": r.document.title,
                    "version": r.document.version,
                    "locator": r.document.locator,
                    "score": round(r.score, 4),
                    "chunk_text": r.chunk_text,
                }
            )
        )
    if not results:
        print("(no matches above zero score)", file=sys.stderr)


def _cmd_mock_evidence(args: argparse.Namespace) -> None:
    evidence = build_mock_evidence(
        data_dir=args.data_dir,
        patient_id=args.patient_id,
        as_of=args.at,
        window_minutes=args.window_minutes,
    )
    print(json.dumps(evidence.to_dict(), indent=2))


def _build_agent(args: argparse.Namespace) -> ClinicalReasoningAgent:
    retriever = RAGRetriever.from_jsonl(args.corpus)
    client = None
    if args.fake_llm:
        from .fake_llm import FakeAnthropicClient

        client = FakeAnthropicClient()
    return ClinicalReasoningAgent(retriever=retriever, client=client)


def _cmd_reason(args: argparse.Namespace) -> None:
    evidence = build_mock_evidence(
        data_dir=args.data_dir,
        patient_id=args.patient_id,
        as_of=args.at,
        window_minutes=args.window_minutes,
    )
    agent = _build_agent(args)
    output = agent.run(evidence)
    print(json.dumps(output.to_dict(), indent=2, default=asdict))


def _cmd_demo(args: argparse.Namespace) -> None:
    import glob
    import os

    from .mock_evidence import _load_jsonl  # noqa: SLF001 (demo convenience)

    patients = sorted(
        {
            row["patient_id"]
            for row in _load_jsonl(os.path.join(args.data_dir, "patient_context.jsonl"))
        }
    )
    episodes = _load_jsonl(os.path.join(args.data_dir, "ground_truth_episodes.jsonl"))
    if not episodes:
        print("no ground-truth episodes found for demo", file=sys.stderr)
        return
    episode = episodes[0]
    patient_id = episode["patient_id"]
    at = episode["escalation_window_start"]

    print(f"# Demo: {patient_id} at {at} (scenario={episode['scenario']!r})\n")
    evidence = build_mock_evidence(data_dir=args.data_dir, patient_id=patient_id, as_of=at)
    print("## Consolidated evidence (mock stand-in for Person 3)")
    print(json.dumps(evidence.to_dict(), indent=2))

    agent_args = argparse.Namespace(corpus=args.corpus, fake_llm=args.fake_llm)
    agent = _build_agent(agent_args)
    output = agent.run(evidence)
    print("\n## Clinical Reasoning Agent output")
    print(json.dumps(output.to_dict(), indent=2, default=asdict))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clinical-reasoning")
    sub = parser.add_subparsers(dest="command", required=True)

    p_query = sub.add_parser("query", help="Test the RAG retriever standalone")
    p_query.add_argument("--corpus", required=True)
    p_query.add_argument("--q", required=True)
    p_query.add_argument("--top-k", type=int, default=3)
    p_query.set_defaults(func=_cmd_query)

    p_evidence = sub.add_parser(
        "mock-evidence", help="Build ConsolidatedEvidence from Person 1's real data"
    )
    p_evidence.add_argument("--data-dir", required=True)
    p_evidence.add_argument("--patient-id", required=True)
    p_evidence.add_argument("--at", required=True, help="ISO 8601 event_time")
    p_evidence.add_argument("--window-minutes", type=int, default=30)
    p_evidence.set_defaults(func=_cmd_mock_evidence)

    p_reason = sub.add_parser(
        "reason", help="Run the Clinical Reasoning Agent end to end"
    )
    p_reason.add_argument("--data-dir", required=True)
    p_reason.add_argument("--corpus", required=True)
    p_reason.add_argument("--patient-id", required=True)
    p_reason.add_argument("--at", required=True)
    p_reason.add_argument("--window-minutes", type=int, default=30)
    p_reason.add_argument(
        "--fake-llm",
        action="store_true",
        help="Use the offline fake LLM client instead of a real Anthropic call",
    )
    p_reason.set_defaults(func=_cmd_reason)

    p_demo = sub.add_parser(
        "demo", help="Run the first ground-truth episode end to end"
    )
    p_demo.add_argument("--data-dir", required=True)
    p_demo.add_argument("--corpus", required=True)
    p_demo.add_argument("--fake-llm", action="store_true")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
