"""A deterministic, offline stand-in for anthropic.Anthropic().

Implements just enough of `client.messages.create(...)` to drive
ClinicalReasoningAgent's tool loop without any network access or API
key. It scripts a fixed, sensible sequence: search the knowledge base,
retrieve similar cases, then submit an assessment built from whatever
those two tool calls actually returned (not hard-coded text), so the
demo output still reflects the real evidence and retrieval content.

This is a development/demo aid, not a mock of model *behaviour* or
*quality* — it exists so the RAG + tool-plumbing can be exercised and
unit-tested reliably before a real ANTHROPIC_API_KEY is wired in.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass
class _Block:
    type: str
    id: str = ""
    name: str = ""
    input: dict[str, Any] | None = None
    text: str = ""


def _tool_use(name: str, tool_input: dict[str, Any]) -> _Block:
    return _Block(type="tool_use", id=f"fake_{uuid.uuid4().hex[:8]}", name=name, input=tool_input)


class FakeAnthropicClient:
    """Drop-in for the `client` argument of ClinicalReasoningAgent."""

    def __init__(self) -> None:
        self._step = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, system, tools, messages):
        self._step += 1

        if self._step == 1:
            query = self._extract_suggested_query(messages)
            content = [_tool_use("search_clinical_knowledge", {"query": query, "top_k": 3})]
            return SimpleNamespace(content=content, stop_reason="tool_use")

        if self._step == 2:
            evidence = self._extract_evidence(messages)
            pattern = evidence.get("current_vitals", {}) if evidence else {}
            content = [
                _tool_use(
                    "retrieve_similar_cases",
                    {"physiological_pattern": pattern, "top_k": 2},
                )
            ]
            return SimpleNamespace(content=content, stop_reason="tool_use")

        # Step 3: build submit_assessment from whatever the last two tool
        # results actually contained.
        knowledge_hits, similar_cases = self._extract_last_tool_results(messages)
        citations = [
            {
                "document_id": hit["document_id"],
                "version": hit["version"],
                "locator": hit["locator"],
                "source_url": hit.get("source_url", ""),
            }
            for hit in knowledge_hits[:2]
        ]
        contributing_factors = [hit["title"] for hit in knowledge_hits[:2]]
        similar_case_ids = [c["case_id"] for c in similar_cases[:2]]

        explanation = (
            "Fused evidence shows the flagged vital-sign pattern above "
            "clinically relevant thresholds. Retrieved guidance "
            + (
                f"({', '.join(citations_doc_ids(citations))}) "
                if citations
                else ""
            )
            + "supports treating this as a pattern warranting clinician "
            "review rather than dismissing it as noise. "
            + (
                f"Similar past cases ({', '.join(similar_case_ids)}) were "
                "previously accepted as genuine by a clinician."
                if similar_case_ids
                else "No closely similar past cases were found."
            )
        )

        submission = {
            "explanation": explanation,
            "contributing_factors": contributing_factors or ["insufficient retrieved evidence"],
            "recommended_action": (
                "Escalate for clinician review; confirm against local "
                "protocol before any treatment decision."
            ),
            "citations": citations,
            "similar_cases_referenced": similar_case_ids,
            "confidence": 0.6 if citations else 0.3,
        }
        content = [_tool_use("submit_assessment", submission)]
        return SimpleNamespace(content=content, stop_reason="tool_use")

    @staticmethod
    def _extract_suggested_query(messages: list[dict[str, Any]]) -> str:
        first_user = messages[0]["content"]
        marker = "starting point for search_clinical_knowledge (feel free to refine it): "
        if marker in first_user:
            return first_user.split(marker, 1)[1].strip().strip("'")
        return "deterioration"

    @staticmethod
    def _extract_evidence(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        first_user = messages[0]["content"]
        try:
            json_str = first_user.split("(JSON):\n", 1)[1].split("\n\nA rule-based", 1)[0]
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            return None

    @staticmethod
    def _extract_last_tool_results(
        messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        knowledge_hits: list[dict[str, Any]] = []
        similar_cases: list[dict[str, Any]] = []
        for msg in messages:
            if msg["role"] != "user" or not isinstance(msg["content"], list):
                continue
            for block in msg["content"]:
                if block.get("type") != "tool_result":
                    continue
                try:
                    parsed = json.loads(block["content"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(parsed, list) and parsed:
                    if "document_id" in parsed[0]:
                        knowledge_hits = parsed
                    elif "case_id" in parsed[0]:
                        similar_cases = parsed
        return knowledge_hits, similar_cases


def citations_doc_ids(citations: list[dict[str, Any]]) -> list[str]:
    return [c["document_id"] for c in citations]
