# Clinical Deterioration Copilot — Clinical Knowledge (RAG) + Clinical Reasoning

## What's actually built

- `knowledge_store.py` — loads `knowledge_corpus.jsonl`, chunks documents
  (a no-op today since the corpus is short excerpts, but ready for
  longer source guidelines later), and retrieves the top-k relevant
  chunks for a query via TF-IDF + cosine similarity.
- `query_builder.py` — a small, transparent rule-based bridge from
  numeric vitals/trend/risk to the clinical vocabulary used in the
  corpus (e.g. low systolic BP → "hypotension"), used as a first-pass
  retrieval query.
- `reasoning_agent.py` — the Clinical Reasoning Agent: an
  Anthropic-SDK tool-calling loop with three tools
  (`search_clinical_knowledge`, `retrieve_similar_cases`,
  `submit_assessment`). It forces the final answer through the
  `submit_assessment` tool call (structured JSON schema) instead of
  parsing free text, and every citation it returns must reference a
  document actually retrieved from the corpus, not one the model made up.
- `experience_memory_client.py` — the interface P4 needs from
  P2/P3's Experience Memory, plus `StubExperienceMemoryClient`, an
  in-memory placeholder so the agent has real (if fake) cases to
  reason over today.
- `integration.py` — **real** adapters onto Person 1's `TrendAgent`
  and Person 2's `PatientStateManager` / `AnomalyAgent`, now that
  those exist. See "Integration status" below.
- `mock_evidence.py` — a fully standalone fallback that builds
  `ConsolidatedEvidence` straight from Person 1's raw generated files
  with local heuristics, no dependency on the `clinical_data_gen`
  package being importable. Useful for quick dev/demo without both
  repos checked out side by side.
- `fake_llm.py` — a deterministic offline stand-in for
  `anthropic.Anthropic()` so the whole pipeline (retrieval → tool
  calls → structured output) can be run and unit-tested with no
  network access and no API key.
- `cli.py` — `query`, `mock-evidence`, `reason`, `demo` subcommands
  (all using the standalone `mock_evidence` path).
- `scripts/run_real_pipeline_demo.py` — runs the **actual** P1
  `TrendAgent` and P2 `PatientStateManager`/`AnomalyAgent` against
  Person 1's generated data, then feeds the real result into P4.
- `tests/` — 10 tests, all passing: retrieval, evidence building
  against real generated data, the full tool-calling loop offline,
  and (when the `DataGen-main` checkout is present as a sibling
  directory) the real P1/P2 integration path.

## Integration status

| Component                                    | Status                | What P4 does                                                                                                                                         |
| -------------------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1 Trend Analyzer (`trend_agent.TrendAgent`) | **Built, integrated** | `integration.trend_summary_from_trend_agent_output` consumes its real output                                                                         |
| P2 Patient State (`PatientStateManager`)     | **Built, integrated** | `integration.update_patient_state` feeds it real vitals                                                                                              |
| P2 Anomaly Agent (`AnomalyAgent`)            | **Built, integrated** | `integration.anomaly_summary_from_anomaly_result` consumes its real `AnomalyResult`                                                                  |
| P3 Risk Assessor + Evidence Fusion           | **Not built**         | `integration.placeholder_risk_from_vitals` stands in; every `ConsolidatedEvidence` built this way carries a note in `conflicts_or_missing` saying so |
| P2/P3 Experience Memory (retrieval side)     | **Not built**         | `StubExperienceMemoryClient` stands in                                                                                                               |
| P5 Alert Policy / Audit                      | **Not built**         | `ReasoningOutput` is the proposed hand-off shape                                                                                                     |

Run the real integration end to end:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/run_real_pipeline_demo.py \
  --datagen-root /path/to/DataGen-main \
  --corpus knowledge/knowledge_corpus.jsonl --fake-llm
```

This actually calls P1's and P2's real code (not a re-implementation)
against the generated `SYN-0002` haemodynamic-shock episode, then
runs it through P4's real RAG retrieval and reasoning agent. Verified
working end to end on the current `DataGen-main` checkout — see
`tests/test_integration.py`.

Two real mismatches surfaced while wiring this up, documented in
detail in `integration.py`'s module docstring rather than silently
patched over, since neither is P4's call to make:

1. **Vital-name vocabulary.** P1's `TrendAgent` uses generator-native
   names (`heart_rate`, `spo2`, ...); P2's `PatientState`/`AnomalyAgent`
   use PhysioNet-style `CORE_VITALS` names (`HR`, `O2Sat`, ...), and
   there's no generator field for `Temp` at all, so it's always
   reported missing to P2 today. `integration.CORE_VITAL_FROM_GENERATED`
   is the translation table.
2. **Patient context richness.** P2's `StaticContext` doesn't carry
   `relevant_history` / `current_medications` / `recent_labs`, which
   the report says the Clinical Reasoning Agent needs and which the
   raw `patient_context.jsonl` already has. P4 currently reads patient
   context from the raw payload rather than `state.static_context`.

Also worth flagging for P2: `p2_impl_details.md` describes
`anomaly_agent.update(state=state, feedback=feedback)` with an
`AnomalyFeedback` dataclass, but the shipped `AnomalyAgent.update()`
signature is `update(self, state, label: int)` — no `AnomalyFeedback`
type exists yet in `clinical_data_gen.anomaly`. Not a P4 blocker
(P4 only calls `.predict()`), but worth confirming with P2 before P5
wires up clinician feedback.

## Contracts with the rest of the team (proposed, not yet confirmed)

One upstream component still doesn't exist. Rather than block on it,
this component defines the shape it needs and isolates the
assumption in one place:

1. **`ConsolidatedEvidence`** (`schema.py`) — trend and anomaly are
   now populated from Person 1/2's real output (`integration.py`).
   Risk is still a P4 placeholder (`integration.placeholder_risk_from_vitals`)
   pending Person 3's Risk Assessor + Evidence Fusion. **When Person 3
   ships**, pass their real `RiskSummary` (or an adapter that builds
   one) as `build_consolidated_evidence(..., risk=...)` — everything
   else in the pipeline is unaffected, since it already only reads
   `evidence.risk`, never recomputes it.

2. **`ExperienceMemoryClient`** (`experience_memory_client.py`) — the
   one method P4 needs from P2/P3's Experience Memory:
   `retrieve_similar_cases(physiological_pattern, top_k) -> list[SimilarCase]`.
   **When that store ships**, implement this Protocol against it and
   pass it into `ClinicalReasoningAgent(experience_client=...)` instead
   of `StubExperienceMemoryClient()`.

3. **`ReasoningOutput`** (`schema.py`) is what this component hands to
   Person 5 (Alert Policy + Clinician Interface + Audit Trail):
   explanation, contributing factors, recommended action, citations
   (document_id/version/locator, for audit-record completeness — see
   report Sec. 5), similar cases referenced, and a confidence score.
   Person 5 should treat this as the current proposed shape and flag
   anything they need added.

If Person 5 wants this on a Kafka topic like the rest of the pipeline
(`vitals.raw`, `patient.context`), wrap `ReasoningOutput.to_dict()` in
the same `EventEnvelope` Person 1 already defined — it's reproduced
verbatim in `schema.py` so it's wire-compatible without an import
across team code.

## Running it

```bash
pip install -r requirements.txt
export PYTHONPATH=src

# Test retrieval alone
python -m clinical_reasoning.cli query \
  --corpus knowledge/knowledge_corpus.jsonl \
  --q "falling blood pressure rising heart rate" --top-k 3

# Build evidence from Person 1's real generated data for one patient
python -m clinical_reasoning.cli mock-evidence \
  --data-dir /path/to/DataGen-main/data/generated \
  --patient-id SYN-0002 --at 2026-09-18T10:07:00+00:00

# Run the full agent (offline, no API key needed)
python -m clinical_reasoning.cli reason \
  --data-dir /path/to/DataGen-main/data/generated \
  --corpus knowledge/knowledge_corpus.jsonl \
  --patient-id SYN-0002 --at 2026-09-18T10:07:00+00:00 --fake-llm

# Same, but against the real haemodynamic_shock ground-truth episode
python -m clinical_reasoning.cli demo \
  --data-dir /path/to/DataGen-main/data/generated \
  --corpus knowledge/knowledge_corpus.jsonl --fake-llm
```

Drop `--fake-llm` and set `ANTHROPIC_API_KEY` to run against the real
model (`claude-sonnet-4-6` by default, see `config/reasoning_config.json`).

Run tests:

```bash
pip install pytest
export PYTHONPATH=src
pytest -q
```

`tests/test_mock_evidence.py` and `tests/test_integration.py` look for
Person 1/2's checkout at `../DataGen-main/DataGen-main` relative to
this repo (i.e. both repos unzipped as siblings) and skip themselves
if that path isn't there — adjust once both repos live in their final
merged layout.

## Known gaps / what's next

- Retrieval is TF-IDF, not a dense embedding model — deliberate for
  now (see `knowledge_store.py` docstring) because it needs no model
  download or embedding API key. `EmbeddingBackend` is defined as an
  explicit interface so a dense backend (e.g. Voyage AI, which
  Anthropic recommends for RAG) can be swapped in without touching
  the retriever or the agent.
- `mock_evidence.py`'s trend/anomaly/risk logic is a placeholder for
  demo purposes only, not a proposal for how Persons 2/3 should build
  theirs.
- No persisted vector index yet — the corpus is small enough
  (currently 4 documents) that building the TF-IDF index at process
  start is effectively free; revisit if the real corpus grows a lot.
- Confluence with Person 5's Alert Policy on exactly which fields of
  `ReasoningOutput` it needs is still open.
