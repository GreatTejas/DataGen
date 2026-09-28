import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clinical_reasoning.knowledge_store import RAGRetriever, chunk_document, load_corpus
from clinical_reasoning.schema import KnowledgeDocument

CORPUS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "knowledge", "knowledge_corpus.jsonl"
)


def test_load_corpus_has_expected_fields():
    docs = load_corpus(CORPUS_PATH)
    assert len(docs) >= 4
    doc = docs[0]
    assert doc.document_id
    assert doc.text
    assert doc.locator


def test_retrieval_finds_qsofa_for_sepsis_query():
    retriever = RAGRetriever.from_jsonl(CORPUS_PATH)
    results = retriever.retrieve("respiratory rate blood pressure suspected infection", top_k=2)
    assert results
    doc_ids = {r.document.document_id for r in results}
    assert "sepsis3-qsofa-2016" in doc_ids or "sepsis3-definition-2016" in doc_ids


def test_retrieval_returns_nothing_for_zero_score_irrelevant_query():
    retriever = RAGRetriever.from_jsonl(CORPUS_PATH)
    results = retriever.retrieve("zzz nonexistent gibberish term unrelated", top_k=3)
    # allowed to be empty; if not empty, scores must still be > 0
    assert all(r.score > 0 for r in results)


def test_chunk_document_short_text_is_one_chunk():
    doc = KnowledgeDocument(
        document_id="d1",
        title="t",
        source_class="s",
        version="v",
        source_url="u",
        locator="l",
        applicability="a",
        text="Short guideline text. Two sentences here.",
    )
    chunks = chunk_document(doc, max_chars=500)
    assert len(chunks) == 1


def test_chunk_document_long_text_splits():
    long_sentence = "This is a clinical guideline sentence about vitals. " * 20
    doc = KnowledgeDocument(
        document_id="d2",
        title="t",
        source_class="s",
        version="v",
        source_url="u",
        locator="l",
        applicability="a",
        text=long_sentence,
    )
    chunks = chunk_document(doc, max_chars=200)
    assert len(chunks) > 1
