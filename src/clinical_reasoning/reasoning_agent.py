"""Clinical Reasoning Agent: LLM + tool calling over fused evidence.

Consumes a ConsolidatedEvidence object (Person 3's contract, see
schema.py), retrieves grounding material via two tools
(`search_clinical_knowledge`, `retrieve_similar_cases`), and forces a
structured final answer via a third tool (`submit_assessment`) rather
than parsing free-text JSON, so the output is always well-formed.

The Anthropic Python SDK's tool-use message shapes were verified
directly against the installed `anthropic==1.8.0` package
(`Message`, `ToolUseBlock`, `TextBlock` field names) rather than
assumed from memory.

Set ANTHROPIC_API_KEY in the environment to run against the real API.
For offline development/tests, pass any object implementing
`.messages.create(**kwargs) -> Message` as `client` (see
tests/test_reasoning_agent.py for a fully offline fake).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict
from typing import Any, Optional

from .experience_memory_client import ExperienceMemoryClient, StubExperienceMemoryClient
from .knowledge_store import RAGRetriever
from .query_builder import build_query
from .schema import Citation, ConsolidatedEvidence, ReasoningOutput

DEFAULT_MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are the Clinical Reasoning component of a bedside \
deterioration-escalation copilot. You receive fused evidence (trend, \
anomaly, risk, and patient context) for one patient.

Rules you must follow:
- You do not diagnose. You explain what the evidence shows and why it is \
or is not concerning, grounded in retrieved clinical guidance and similar \
past cases.
- Use the search_clinical_knowledge tool to retrieve relevant guideline \
passages before writing your explanation. Use retrieve_similar_cases to \
check how similar past cases were handled.
- Every clinical claim in your explanation that relies on a guideline must \
be traceable to a document you actually retrieved. Do not invent \
document IDs, versions, or citations.
- When you are ready, call submit_assessment exactly once with your final \
structured answer. Do not produce a final text answer outside that tool \
call.
- Local hospital policy and clinician judgement always govern actual care; \
say so if the recommended_action could be read as an order rather than a \
suggestion.
"""

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_clinical_knowledge",
        "description": (
            "Search the versioned clinical-knowledge corpus (Sepsis-3/qSOFA, "
            "NEWS2, NICE NG51 excerpts) for passages relevant to a query. "
            "Returns document_id, title, version, locator, source_url, and "
            "the matched text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_similar_cases",
        "description": (
            "Retrieve similar past patient cases from the experience-memory "
            "store, including the clinician's prior decision and feedback."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "physiological_pattern": {"type": "object"},
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["physiological_pattern"],
        },
    },
    {
        "name": "submit_assessment",
        "description": "Submit the final, structured clinical reasoning output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "contributing_factors": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "recommended_action": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                            "version": {"type": "string"},
                            "locator": {"type": "string"},
                            "source_url": {"type": "string"},
                        },
                        "required": ["document_id", "version", "locator"],
                    },
                },
                "similar_cases_referenced": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "confidence": {"type": "number"},
            },
            "required": [
                "explanation",
                "contributing_factors",
                "recommended_action",
                "citations",
                "confidence",
            ],
        },
    },
]


class ReasoningAgentError(RuntimeError):
    pass


class ClinicalReasoningAgent:
    def __init__(
        self,
        retriever: RAGRetriever,
        experience_client: Optional[ExperienceMemoryClient] = None,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        max_tool_iterations: int = 6,
    ) -> None:
        self._retriever = retriever
        self._experience_client = experience_client or StubExperienceMemoryClient()
        self._model = model
        self._max_tool_iterations = max_tool_iterations
        self._client = client  # lazily created in _get_client() if None

    def _get_client(self):
        if self._client is not None:
            return self._client
        import anthropic  # local import: only needed for the live path

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ReasoningAgentError(
                "ANTHROPIC_API_KEY not set and no client was injected. "
                "Set the env var for live calls, or pass a fake client "
                "for offline development/tests."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _run_tool(self, name: str, tool_input: dict[str, Any]) -> Any:
        if name == "search_clinical_knowledge":
            top_k = tool_input.get("top_k", 3)
            chunks = self._retriever.retrieve(tool_input["query"], top_k=top_k)
            return [
                {
                    "document_id": c.document.document_id,
                    "title": c.document.title,
                    "version": c.document.version,
                    "locator": c.document.locator,
                    "source_url": c.document.source_url,
                    "text": c.chunk_text,
                    "score": c.score,
                }
                for c in chunks
            ]
        if name == "retrieve_similar_cases":
            top_k = tool_input.get("top_k", 3)
            cases = self._experience_client.retrieve_similar_cases(
                tool_input.get("physiological_pattern", {}), top_k=top_k
            )
            return [asdict(c) for c in cases]
        raise ReasoningAgentError(f"unknown tool: {name}")

    def run(self, evidence: ConsolidatedEvidence) -> ReasoningOutput:
        client = self._get_client()
        suggested_query = build_query(evidence)

        user_message = (
            "Consolidated evidence for one patient (JSON):\n"
            f"{json.dumps(evidence.to_dict(), indent=2)}\n\n"
            f"A rule-based first-pass query you may use as a starting point "
            f"for search_clinical_knowledge (feel free to refine it): "
            f"{suggested_query!r}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]

        tool_call_count = 0
        for _ in range(self._max_tool_iterations):
            response = client.messages.create(
                model=self._model,
                max_tokens=1500,
                system=_SYSTEM_PROMPT,
                tools=_TOOLS,
                messages=messages,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                raise ReasoningAgentError(
                    "model ended without calling submit_assessment"
                )

            messages.append({"role": "assistant", "content": response.content})

            tool_results: list[dict[str, Any]] = []
            final_output: Optional[ReasoningOutput] = None
            for block in tool_use_blocks:
                tool_call_count += 1
                if block.name == "submit_assessment":
                    final_output = self._to_reasoning_output(
                        block.input, evidence, tool_call_count
                    )
                    # Anthropic requires a tool_result for every tool_use
                    # in the same turn, even the terminal one.
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "assessment received",
                        }
                    )
                    continue
                result = self._run_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

            if final_output is not None:
                return final_output

            messages.append({"role": "user", "content": tool_results})

        raise ReasoningAgentError(
            f"exceeded {self._max_tool_iterations} tool-call iterations "
            "without a submit_assessment call"
        )

    def _to_reasoning_output(
        self,
        submission: dict[str, Any],
        evidence: ConsolidatedEvidence,
        tool_call_count: int,
    ) -> ReasoningOutput:
        citations = [
            Citation(
                document_id=c["document_id"],
                version=c["version"],
                locator=c["locator"],
                source_url=c.get("source_url", ""),
            )
            for c in submission.get("citations", [])
        ]
        return ReasoningOutput(
            schema_version="0.1.0-p4-draft",
            patient_id=evidence.patient_id,
            evidence_id=evidence.evidence_id,
            generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            explanation=submission["explanation"],
            contributing_factors=submission.get("contributing_factors", []),
            recommended_action=submission["recommended_action"],
            citations=citations,
            similar_cases_referenced=submission.get("similar_cases_referenced", []),
            confidence=float(submission.get("confidence", 0.0)),
            model=self._model,
            tool_call_count=tool_call_count,
        )
