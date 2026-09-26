"""Retrieval of evidence-based clinical guidelines from the versioned RAG knowledge corpus."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from ..schemas import RAGCitation


class EvidenceRetriever:
    """Retrieves versioned clinical guideline documents matching physiological patterns."""

    def __init__(self, corpus_path: Optional[str | Path] = None) -> None:
        self.corpus: list[RAGCitation] = []
        if corpus_path:
            self.load_corpus(Path(corpus_path))
        else:
            base_pkg = Path(__file__).resolve().parents[3]
            default_paths = [
                Path("data/generated/knowledge_corpus.jsonl"),
                Path("DataGen-main/data/generated/knowledge_corpus.jsonl"),
                base_pkg / "data" / "generated" / "knowledge_corpus.jsonl",
                Path("knowledge/protocol_corpus.jsonl"),
                Path("DataGen-main/knowledge/protocol_corpus.jsonl"),
                base_pkg / "knowledge" / "protocol_corpus.jsonl",
                Path("../data/generated/knowledge_corpus.jsonl"),
            ]
            for p in default_paths:
                if p.exists():
                    self.load_corpus(p)
                    break

        # Fallback default clinical corpus if file is missing
        if not self.corpus:
            self._init_fallback_corpus()

    def load_corpus(self, path: Path) -> None:
        self.corpus = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.corpus.append(
                            RAGCitation(
                                document_id=data.get("document_id", ""),
                                title=data.get("title", ""),
                                version=data.get("version", "2016"),
                                locator=data.get("locator", ""),
                                source_url=data.get("source_url", ""),
                                applicability=data.get("applicability", ""),
                                text=data.get("text", ""),
                            )
                        )
        except Exception:
            self._init_fallback_corpus()

    def _init_fallback_corpus(self) -> None:
        self.corpus = [
            RAGCitation(
                document_id="sepsis3-qsofa-2016",
                title="Sepsis-3: qSOFA bedside criteria",
                version="2016",
                locator="JAMA 315(8), pp. 801-810; qSOFA criteria section",
                source_url="https://doi.org/10.1001/jama.2016.0287",
                applicability="Adults with suspected infection outside ICU",
                text="qSOFA is a bedside prompt for possible poor outcomes in suspected infection. Criteria: RR >= 22/min, altered mentation, systolic BP <= 100 mmHg.",
            ),
            RAGCitation(
                document_id="sepsis3-definition-2016",
                title="Sepsis-3 consensus definition",
                version="2016",
                locator="JAMA 315(8), pp. 801-810; New Terms and Definitions",
                source_url="https://doi.org/10.1001/jama.2016.0287",
                applicability="Adults with suspected infection",
                text="Defines sepsis as life-threatening organ dysfunction caused by a dysregulated host response to infection (SOFA increase >= 2).",
            ),
            RAGCitation(
                document_id="news2-rcp-2017",
                title="National Early Warning Score 2",
                version="2017",
                locator="RCP NEWS2 report, 2017; scoring charts and Scale 2 guidance",
                source_url="https://www.rcp.ac.uk/media/a4ibkkbf/news2-final-report_0_0.pdf",
                applicability="Adults; follow local adoption and exclusions",
                text="NEWS2 standardizes acute-illness severity assessment. Includes designated oxygen-saturation Scale 2 for patients with hypercapnic respiratory failure.",
            ),
            RAGCitation(
                document_id="nice-ng51-2016",
                title="NICE NG51 suspected sepsis",
                version="2016",
                locator="NICE NG51 guideline landing page",
                source_url="https://www.nice.org.uk/guidance/ng51",
                applicability="UK suspected sepsis pathways",
                text="Guidance covers recognition, diagnosis, and early management of suspected sepsis. Requires immediate clinical team review and antibiotic delivery within 1 hour.",
            ),
        ]

    def retrieve(self, syndrome: str, keywords: Optional[list[str]] = None) -> list[RAGCitation]:
        """Retrieves relevant citations based on syndrome and clinical keywords."""
        citations = []
        kw_list = [k.lower() for k in (keywords or [])]

        for doc in self.corpus:
            doc_text = (doc.title + " " + doc.text + " " + doc.applicability).lower()
            matched = False

            if syndrome in {"sepsis_like", "beta_blocked_sepsis"} and "sepsis" in doc.document_id:
                matched = True
            elif syndrome == "haemodynamic_shock" and ("qsofa" in doc.document_id or "news2" in doc.document_id):
                matched = True
            elif syndrome in {"respiratory_decline", "masked_hypoxemia", "opioid_respiratory_depression"} and "news2" in doc.document_id:
                matched = True
            elif any(k in doc_text for k in kw_list):
                matched = True

            if matched and doc not in citations:
                citations.append(doc)

        # Ensure at least NEWS2 or relevant general guidance is attached
        if not citations:
            for doc in self.corpus:
                if "news2" in doc.document_id:
                    citations.append(doc)
                    break

        return citations
