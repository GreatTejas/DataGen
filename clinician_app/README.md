# Clinician review demo (Person 5)

This local dashboard completes the Person 5 portion of the Team 38 deterioration-copilot demo. It reads the repository's generated `data/generated` mock cohort, shows patient observations against a personal baseline and the alert evidence, and records clinician actions in a local JSONL audit file. No external package is required.

## Run

From the `DataGen-main` project folder:

```powershell
python clinician_app/server.py
```

Open `http://127.0.0.1:8765`. To use another local port, set `CLINICIAN_APP_PORT` first.

The **Ask Gemini to choose** button sends the selected patient's observed vitals, context, trend summary, and Person 5 policy result to Google's Gemini `generateContent` REST API. Gemini returns one structured suggestion from **Accept, Dismiss, Defer, or Investigate** plus an evidence-based reason. The clinician still chooses their own action; the suggestion is included in the audit/feedback record for later agreement analysis. Set `GEMINI_API_KEY` in the environment before starting the server; the key stays on the server and is not sent to the browser. The default model is `gemini-3.8-flash`; override with `GEMINI_MODEL` if needed. Without a key, the interface labels a deterministic Person 5 rule fallback, not a Gemini response.

```powershell
$env:GEMINI_API_KEY = "your-key"
python clinician_app/server.py
```

## Clinician actions and audit

Accept, Dismiss, Defer, and Investigate actions (including an optional reason) append to `data/evaluation/clinician_audit.jsonl`. A feedback export for the Person 2 learning component is written to `data/evaluation/clinician_feedback.jsonl`; the teammate's learner is not yet connected, so this is an integration handoff file rather than a live model update. The audit JSONL also records every replayed observation, demo trend and policy output, explanation, empty RAG evidence/citation fields, and clinician decision.

## Evaluation metrics (reference only)

Metrics are available from the `/api/dashboard` endpoint for reference; they are intentionally not shown in the clinician interface. To create saved reports, run `python scripts/evaluate_alert_policy.py`. This writes `data/evaluation/alert_policy/metrics.json`, `episode_predictions.jsonl`, and `alert_events.jsonl`. The policy uses the first 30 observations as a patient baseline, requires at least two readings with persistent cross-system evidence in a three-reading window, ignores suspect-quality readings, deduplicates alerts during an active episode, and applies a 30-minute cooldown before a new episode alert. Its decision code never reads ground-truth labels; the evaluator reads those separate files only after policy predictions are produced. Episode precision groups alerts for the same patient within 30 minutes as one alert episode.

The report includes episode precision and recall within the labelled escalation window, median lead time, alerts per patient-hour, duplicate suppression, artifact false-positive rate, RAG citation coverage, and audit completeness. These are demo metrics from a simple baseline-relative policy, not clinical validation results or metrics for the integrated multi-agent system. Citation coverage is zero until the RAG service is connected. Gemini action agreement can only be calculated after clinician actions have been recorded; agreement is a review signal and should not be presented as ground-truth clinical accuracy.

The generator's separate online anomaly-model evaluation is in `data/evaluation/online_lr/metrics.json`. It evaluates event-level anomaly predictions and should not be combined with the episode-level alert-policy cards.
