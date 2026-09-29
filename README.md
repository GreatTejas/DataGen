# Clinical Deterioration Copilot - Data Generation

This component creates a reproducible, event-time dataset for the Agentic Clinical Deterioration & Escalation Copilot. It is designed for demonstration and model/system evaluation - **not clinical care, validation, or deployment**.

The generator produces much more than random vital signs:

- Age/sex-stratified baselines fitted from the local PhysioNet Challenge 2019 training set.
- Real, baseline-normalised sepsis-onset windows time-warped into synthetic sepsis-like trajectories.
- Discriminating cases: beta-blocked sepsis, opioid respiratory depression, masked hypoxemia, and COPD low-baseline SpO2.
- Hard negatives: monitoring artifacts, post-physiotherapy, anxiety, and ambulation.
- Benign noisy/transient episodes, so downstream logic must suppress false alarms instead of reacting to one threshold crossing.
- Per-reading trajectory labels and episode-level ground truth, including the first clinically meaningful escalation window.
- Static patient context and a versioned clinical-knowledge corpus ready for a RAG service.
- A stable event envelope suitable for Kafka, file replay, or API ingestion.

## Architecture

```mermaid
flowchart LR
    D[PhysioNet 2019 PSV files] --> F[Parameter & template fitting]
    C[Seeded config] --> P[Patient profile factory]
    F --> P
    P --> B[Personalised baseline]
    B --> T[Trajectory engine]
    T --> N[Noise & artifact injector]
    N --> V[VitalEvent JSONL]
    T --> L[Label engine]
    L --> G[Ground-truth episodes]
    P --> X[Static patient context]
    V --> KAFKA[(Kafka topic: vitals.raw)]
    X --> KAFKA2[(Kafka topic: patient.context)]
    G --> EVAL[Offline evaluator]
    KAFKA --> INGEST[Validated Kafka consumer]
    KAFKA2 --> INGEST
    INGEST --> P1[P1 Trend]
    P1 --> P2[P2 State + Anomaly + Learning]
    P2 --> P3[P3 Risk + Evidence Fusion + Experience]
    P3 --> HANDOFF[patient.evidence]
    HANDOFF -. P4 consumer pending .-> P4[P4 RAG + Clinical Reasoning pending]
    P4 -. future integration .-> P5[P5 Alert Policy + Review UI]
    P5 --> AUDIT[Audit + clinician feedback]
    AUDIT -. feedback .-> P2
    FILES[Generated mock cohort] -. local demo input .-> P5
```

### Event and topic contract

| Stream | Kafka topic | Key | Purpose |
| --- | --- | --- | --- |
| Vital readings | `vitals.raw` | `patient_id` | Ordered, event-time patient telemetry |
| Static context | `patient.context` | `patient_id` | Compacted profile available before replay |
| Consolidated observed evidence | `patient.evidence` | `patient_id` | P1-P3 output; explicit P4-pending handoff |
| Invalid input | `vitals.dlq` | `patient_id` | Rejected JSON, schema, or quality events |
| Ground truth | `evaluation.ground_truth` | `episode_id` | Held-out labels for measuring detection and alert burden |
| Knowledge documents | RAG corpus | `document_id` | Versioned, citeable guideline snippets |

All events share `schema_version`, `event_id`, `event_time`, `patient_id`, `source`, and a deterministic `run_id`. The live vital payload contains observed values only. Latent values and labels are deliberately isolated in the evaluation outputs so an agent cannot cheat during demo or testing.

## Dataset construction pipeline

1. Fit age/sex baselines, serial-dependence diagnostics, residual correlation, and real sepsis-onset templates from PhysioNet Challenge 2019 PSV files. Separately fit community HR/BP baselines from CDC NHANES 2017-2018 for hard-negative baseline comparison.
2. Load the versioned fitted parameter artifact, generator configuration, and fixed seed. The generator consumes fitted baselines, real sepsis templates, and OU/correlation estimates through its correlated AR(1) residual-noise process.
3. Sample a static patient profile and derive its baseline from the age/sex stratum.
4. Time-warp a real baseline-normalised window for sepsis-like trajectories; use clearly labelled mechanism-based stress tests for the other archetypes.
5. Add measurement variability and short-lived artifacts. Artifacts can cross a threshold but do not become an episode label.
6. Validate physiology and temporal invariants, write event-time JSONL, then emit separate labels and quality metrics.
7. Replay vital events with their original inter-event timing to Kafka or a local consumer. Publish context first, keyed by patient ID.

This makes the construction pipeline auditable in a midterm report: each signal has a declared source, every synthetic label has a causal scenario, and the seed recreates the exact cohort.

## Quick start

Requires Python 3.10+. Install the project and its runtime dependencies from the repository root:

```powershell
python -m pip install -e .
python -m clinical_data_gen.cli fit-physionet --input data/training_setA --output data/derived/physionet_2019_params.json
python -m clinical_data_gen.cli fit-nhanes-baselines --bp data/secondary/NHANES_2017_2018_BPX_J.XPT --demographics data/secondary/NHANES_2017_2018_DEMO_J.XPT
python -m clinical_data_gen.cli generate --output data/generated --patients 60 --minutes 180 --seed 20260918 --params data/derived/physionet_2019_params.json --reference data/training_setA --fidelity-max-files 1000
python -m clinical_data_gen.cli validate --input data/generated
```

The Kafka consumer validates context and vitals, runs the shared P1-P3 workflow, and publishes consolidated observed evidence to `patient.evidence`. Start it in one terminal, then replay the generated cohort in another:

```powershell
$env:KAFKA_BROKER = "localhost:9092"
python -m ingestion.consumer
```

```powershell
python -m ingestion.producer --input data/generated/vitals_observed.jsonl --speed 60
```

The producer publishes `patient.context` before time-ordered `vitals.raw` readings; `--speed 60` sends one minute of simulated time per real second. The CLI file replay remains available for stdout-only inspection: `python -m clinical_data_gen.cli replay --input data/generated --speed 60`. The P5 dashboard reads the generated mock cohort locally; it does not yet consume Kafka output. P4 RAG retrieval and clinical reasoning are placeholders in both handoffs.

## Outputs

```
data/generated/
  manifest.json                 # seed, schema, generation parameters, hashes
  patient_context.jsonl         # compacted/static context
  vitals_observed.jsonl         # only fields permitted to the streaming copilot
  ground_truth_episodes.jsonl   # scenario, onset, escalation window, evidence
  evaluation_labels.jsonl       # held-out event-level labels and latent values
  data_quality_report.json      # invariant/coverage metrics
  knowledge_corpus.jsonl        # RAG-ready protocol snippets
```

## Important design decisions

### Leakage prevention

The live payload intentionally excludes `scenario`, `risk_label`, `latent`, `trend_phase`, and `episode_id`. These are in the evaluator-only files. Keep them in a separate bucket or topic ACL in the full project.

### Clinical realism without claiming clinical validity

Baselines, serial-dependence diagnostics, residual correlation, and sepsis-onset templates are fitted from the local PhysioNet Challenge 2019 training data. The generator consumes fitted baselines, real sepsis templates, and fitted correlated AR(1) residual noise. CDC NHANES 2017-2018 adds an independent community HR/BP baseline reference for hard-negative comparison; it is not used as a deterioration-trajectory source. Medication effects are intentionally not fitted or applied: Challenge-2019 has no medication history or administration field. The additional discriminating archetypes are mechanism-based stress tests, not claims of clinical validation.

### P4 status: RAG grounding pending

The repository includes a versioned protocol-corpus seed with document IDs, source URLs, versions, and locators. It is not yet chunked, embedded, indexed, or retrieved by a RAG service. The current P4 interface explicitly returns `not_implemented` and empty evidence/citation fields; it must not be described as providing clinical explanations or recommendations.

## Evaluation that makes the demo credible

Report metrics against `ground_truth_episodes.jsonl`, not only a risk score:

- Episode recall and precision within the labelled escalation window.
- Median lead time from first alert to labelled escalation window.
- Alerts per patient-hour and duplicate-alert suppression rate.
- Artifact false-positive rate.
- Retrieval citation coverage and audit-record completeness.
- The generated `data_quality_report.json` records KS distance and lag-1 ACF per channel, real-vs-synthetic correlation error, fitted-versus-realised residual-correlation error, and paired TSTR/TRTR AUROC using patient-level splits. It reports both a deterministic logistic-loss gradient-boosted-stump baseline and stronger depth-3 boosted trees over the five streamed vitals.

Use a patient-level split (not a random row split) so the same profile never appears in both train and evaluation cohorts.

## Project structure

```
src/clinical_data_gen/
  generator.py       # causal trajectory, artifact, labels and validation logic
  fit_physionet.py   # fit PhysioNet parameters and real sepsis templates
  fit_nhanes.py      # fit independent CDC NHANES HR/BP community baselines
  schema.py          # explicit event envelopes and schema version
  pipeline.py        # stateful P1-P3 workflow and P4 handoff
  clinical_reasoning.py # explicit P4 placeholder
  cli.py             # generate, validate, replay commands
src/ingestion/
  producer.py        # context-first event-time Kafka replay
  consumer.py        # validation, P1-P3 execution, evidence handoff, DLQ
clinician_app/       # local P5 mock-cohort review, policy and audit demo
knowledge/            # versioned RAG corpus seed
config/               # reproducible scenario mix and cadence
tests/                # deterministic and safety/invariant tests
```

## Demo sequence

1. Generate a fixed-seed cohort and show `data_quality_report.json`.
2. Publish contexts, then replay observations into `vitals.raw`.
3. Show an artifact that should be suppressed, then a labelled multi-parameter episode.
4. Show the P4-pending handoff (no retrieved evidence, citations, or recommendation) rather than claiming RAG or clinical reasoning is implemented.
5. Use the local P5 review UI to inspect the generated cohort, record clinician actions, and compare policy decisions with the held-out episode window.

## Governance

Do not use real patient identifiers. Generated IDs are synthetic. Preserve the manifest, source/config revision, and knowledge-corpus version with every experiment. A clinician should review all scenario assumptions before any use beyond software demonstration.

## Person 4 integration

Person 4's local RAG and clinical reasoning setup is documented in [P4_README.md](P4_README.md). Local embeddings and the persistent vector index run on this machine; optional Groq reasoning requires a `GROQ_API_KEY` and sends the evidence to Groq Cloud.
