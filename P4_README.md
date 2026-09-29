# Person 4: Clinical Knowledge RAG and Clinical Reasoning

P4 consumes consolidated P3 evidence, retrieves versioned clinical material and prior cases, and emits a structured explanation, contributing factors, cautious recommended action, confidence, and citations. It does not calculate P1 trends, P2 anomalies, P3 risk, or P5 alert policy.

## Local and free setup

Python 3.10 or newer is required. From the DataGen directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[p4]"
```

This installs CPU-capable local embeddings and the Groq client. The first dense-index build downloads `sentence-transformers/all-MiniLM-L6-v2`; subsequent embedding and retrieval runs use the local cached model and SQLite index without network access:

```powershell
python scripts/build_p4_index.py
```

The index is stored at `data/derived/p4_vector_index.sqlite3` and can be rebuilt from the checked-in JSONL corpus. To run retrieval and deterministic reasoning without an API key:

```powershell
python scripts/run_p4_demo.py --retrieval dense
```

Groq inference is optional. Create an API key in Groq Console, then set it in the current PowerShell session; do not put it in source files:

```powershell
# Supply GROQ_API_KEY through a secret manager or process environment.
python scripts/run_p4_demo.py --retrieval dense --groq
```

The Groq chat request is sent to Groq Cloud; only retrieval, the vector index, and tool execution run locally. The Groq Free plan has request/token limits that depend on the account and model; check the current [rate limits](https://console.groq.com/docs/rate-limits). The code uses Groq's documented local function-calling flow and defaults to `openai/gpt-oss-20b` ([tool calling](https://console.groq.com/docs/tool-use/local-tool-calling), [model list](https://console.groq.com/docs/models)).

This project contains synthetic patient data. Do not send real patient identifiers, protected health information, or other confidential clinical records to Groq. Groq calls are not local inference.

## Test

```powershell
python -m pip install pytest
python -m pytest tests/p4 -q
```

The P4 tests cover lexical retrieval, evidence-to-query mapping, deterministic reasoning, persistent dense-index behavior using a test encoder, and the Groq tool/citation flow using a mocked client. The actual Groq API path requires a valid key and network access.

## Implemented P4 scope

- Versioned JSONL corpus with Sepsis-3, qSOFA, NEWS2, historical NICE NG51, and NICE NG253/NG254/NG255 records. NG51 is marked superseded. The summaries preserve source links and locators.
- Corpus validation, duplicate-ID checks, whitespace cleanup, sentence chunking, and overlap.
- Local sentence-transformer embeddings and a persistent SQLite vector store; dependency-light TF-IDF remains available with `--retrieval tfidf`.
- Clinical query construction, top-k retrieval, versioned source citations, and citation validation against passages retrieved in that run.
- Groq tool-calling for local knowledge search, P3 experience-memory retrieval, and structured assessment submission.
- Adapters from P3 `RiskResult`, `FusedEvidence`, and `ExperienceMemory`; structured P4-to-P5 `ReasoningOutput`.
- Deterministic no-key mode for local demos and tests.

The checked-in corpus contains concise, provenance-linked summaries, not the full text of the guidelines or executable NEWS2/qSOFA calculators. It is a prototype retrieval corpus; clinical teams must verify the cited primary sources and locally adopted pathways before any real clinical use.

## P3 integration

```python
from clinical_reasoning.adapters import evidence_from_p3, ExperienceMemoryAdapter
from clinical_reasoning.pipeline import P4Service

evidence = evidence_from_p3(
    evidence_id=f"{patient_id}:{window_end}", patient_id=patient_id,
    window_start=window_start, window_end=window_end,
    current_vitals=generated_vitals, trend_result=trend_result,
    anomaly_result=anomaly_result, risk_result=risk_result,
    patient_context_payload=context_payload, experience_result=experience_result,
    fusion_confidence=fused.final_confidence,
)
service = P4Service(
    "knowledge/protocol_corpus.jsonl",
    retrieval_backend="dense",
    index_path="data/derived/p4_vector_index.sqlite3",
    live_llm=True,
    experience_client=ExperienceMemoryAdapter(experience_memory),
)
reasoning_output = service.reason(evidence).to_dict()
```

P5 can consume `ReasoningOutput.to_dict()`. Citations contain document ID, version, locator, source URL, retrieval score, and the retrieved passage.
