"""Clinical Knowledge (RAG) store: load, chunk, index, and retrieve.

Retrieval backend: TF-IDF + cosine similarity (scikit-learn), chosen
deliberately over a pretrained dense-embedding model because:
  - it needs no model download / external API at index-build time,
    so the hackathon demo doesn't depend on network access or an
    embedding API key being present;
  - the corpus is short, guideline-style clinical text where lexical
    overlap (vital-sign names, clinical terms) is a strong signal.

`EmbeddingBackend` is an explicit interface so a dense backend (e.g.
Voyage AI embeddings, which Anthropic recommends for RAG) can be
swapped in later without touching `RAGRetriever` or the reasoning
agent — see `TfidfBackend` for the shape a new backend must match.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import KnowledgeDocument, RetrievedChunk

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_corpus(path: str | Path) -> list[KnowledgeDocument]:
    """Load a knowledge_corpus.jsonl file into KnowledgeDocument records.

    Field names match Person 1's actual output exactly (verified against
    knowledge_corpus.jsonl): document_id, title, source_class, version,
    source_url, locator, applicability, text.
    """
    docs: list[KnowledgeDocument] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                docs.append(
                    KnowledgeDocument(
                        document_id=row["document_id"],
                        title=row["title"],
                        source_class=row["source_class"],
                        version=row["version"],
                        source_url=row["source_url"],
                        locator=row["locator"],
                        applicability=row["applicability"],
                        text=row["text"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path}:{line_no} missing required field {exc}"
                ) from exc
    return docs


def chunk_document(
    doc: KnowledgeDocument, max_chars: int = 500, overlap_sentences: int = 1
) -> list[str]:
    """Split a document's text into overlapping sentence-group chunks.

    Every document currently in the corpus is a single short paragraph
    (well under max_chars), so this is a no-op in practice today and
    each doc yields exactly one chunk. It exists so a future ingestion
    of full-length guideline PDFs (NICE NG51, NEWS2 report, etc.)
    doesn't require touching the retrieval pipeline.
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(doc.text.strip()) if s]
    if not sentences:
        return [doc.text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        if current and current_len + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            # start next chunk with `overlap_sentences` of trailing context
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
        i += 1
    if current:
        chunks.append(" ".join(current))
    return chunks


class EmbeddingBackend(Protocol):
    """Interface a retrieval backend must satisfy."""

    def fit(self, texts: list[str]) -> None: ...

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return a similarity score per text in `texts`, same order."""
        ...


class TfidfBackend:
    """Default, dependency-light lexical retrieval backend."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        )
        self._matrix = None

    def fit(self, texts: list[str]) -> None:
        self._matrix = self._vectorizer.fit_transform(texts)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self._matrix is None:
            raise RuntimeError("call fit() before score()")
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]
        return sims.tolist()


@dataclass
class _IndexedChunk:
    document: KnowledgeDocument
    chunk_text: str
    chunk_index: int


class RAGRetriever:
    """Loads a corpus, chunks it, indexes it, and answers top-k queries."""

    def __init__(
        self,
        documents: list[KnowledgeDocument],
        backend: EmbeddingBackend | None = None,
        max_chunk_chars: int = 500,
    ) -> None:
        self._backend = backend or TfidfBackend()
        self._chunks: list[_IndexedChunk] = []
        for doc in documents:
            for idx, chunk_text in enumerate(
                chunk_document(doc, max_chars=max_chunk_chars)
            ):
                self._chunks.append(_IndexedChunk(doc, chunk_text, idx))
        if not self._chunks:
            raise ValueError("corpus produced zero chunks")
        self._backend.fit([c.chunk_text for c in self._chunks])

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs) -> "RAGRetriever":
        return cls(load_corpus(path), **kwargs)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        texts = [c.chunk_text for c in self._chunks]
        scores = self._backend.score(query, texts)
        ranked = sorted(
            zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True
        )
        results = []
        for chunk, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    document=chunk.document,
                    chunk_text=chunk.chunk_text,
                    chunk_index=chunk.chunk_index,
                    score=float(score),
                )
            )
        return results
