# P2 — Anomaly Detection Integration API

P2 owns:

- Patient State
- Anomaly Detection
- Online Learning

Other components should use the public interfaces below and should not access P2 internals such as the feature extractor or model directly.

---

## 1. Patient State

New observations should be submitted through the existing public `PatientStateManager` update API.

```python
state = patient_state_manager.update(...)
```

The returned object is:

```python
PatientState
```

Pass this `PatientState` to the Anomaly Agent.

> The exact arguments to `PatientStateManager.update()` are defined by its implementation. Do not create a separate state representation in other modules.

Other components should not directly modify `PatientState` internals.

---

## 2. Anomaly Prediction

Use:

```python
anomaly_result = anomaly_agent.predict(state)
```

Input:

```python
PatientState
```

Output:

```python
AnomalyResult
```

The result contains:

```python
@dataclass
class AnomalyResult:
    patient_id: str
    timestamp: Any
    probability: float
    is_anomalous: bool
    feature_vector: dict[str, float]
    contributing_features: list[str]
    model_version: int
```

### Meaning

- `probability` — anomaly probability produced by the model.
- `is_anomalous` — thresholded anomaly decision.
- `feature_vector` — features used for this prediction.
- `contributing_features` — features contributing to the prediction.
- `model_version` — model version that produced this result.

Do not assume a fixed feature count or feature ordering. Use the feature names provided in `feature_vector`.

---

## 3. Important: Anomaly ≠ Clinical Alert

An anomaly is physiological evidence, not the final clinical decision.

```text
PatientState
    ↓
Anomaly Detection
    ↓
AnomalyResult
    ↓
Risk / Evidence Fusion
    ↓
Clinical Reasoning
    ↓
Alert Policy
```

Therefore:

```python
is_anomalous == True
```

must not automatically be interpreted as:

```text
raise clinical alert
```

Downstream components decide clinical risk and actionability.

---

## 4. P3 / Downstream Integration

P3 should consume the `PatientState` and `AnomalyResult`.

Example:

```python
anomaly_result = anomaly_agent.predict(state)

# Pass both objects to the P3 risk/evidence module.
risk_result = risk_engine.process(
    patient_state=state,
    anomaly_result=anomaly_result,
)
```

The exact P3 function/class name is owned by P3. The important contract is that P3 receives:

```text
PatientState
+
AnomalyResult
```

---

## 5. Clinician Feedback / Online Learning

When explicit anomaly feedback is available:

```python
anomaly_agent.update(
    state=state,
    feedback=feedback,
)
```

Feedback should distinguish **anomaly correctness** from the **clinician's action/decision**.

Recommended structure:

```python
@dataclass
class AnomalyFeedback:
    patient_id: str
    timestamp: Any
    anomaly_label: bool | None
    decision: str | None
    reason: str | None
```

Allowed `decision` values:

```text
accept
dismiss
defer
investigate
```

### Important

Do not automatically interpret:

```text
dismiss → anomaly_label=False
```

A clinician may dismiss an alert because it is not clinically actionable even though the physiological anomaly is real.

If:

```python
feedback.anomaly_label is None
```

the anomaly model should not perform a supervised update from that feedback.

---

## 6. Model Version

Downstream components should preserve:

```python
anomaly_result.model_version
```

when storing alerts, feedback, evidence, or audit records.

This identifies which model version generated the anomaly result.

---

## 7. Ground Truth

`SepsisLabel` and other evaluation labels are **evaluation-only**.

They must not:

- be added to `PatientState`
- be sent as live clinician feedback
- be used as live anomaly labels

For offline/prequential evaluation, ground truth may be used by the evaluator to simulate delayed supervision, but this remains separate from the live runtime pipeline.

---

## 8. P2 Internal Implementation

The following are internal P2 implementation details and should not be accessed directly by other components:

```text
FeatureExtractor
feature normalization
feature count / ordering
model weights
learning rate
gradient updates
Online Logistic Regression implementation
patient history storage
```

This allows P2 to change its internal model or feature representation without requiring changes to other components.

---

## 9. Integration Flow

```text
P1 / Data Ingestion / Trend
            |
            v
  PatientStateManager
            |
            v
      PatientState
            |
            v
      AnomalyAgent
            |
            v
     AnomalyResult
            |
            v
    Risk / Evidence Fusion
            |
            v
    Clinical Reasoning
            |
            v
      Alert Policy / UI
            |
            v
    Clinician Feedback
            |
            v
      AnomalyAgent.update()
```

### Integration rule

Other components should integrate with P2 through:

```text
PatientStateManager
PatientState
AnomalyAgent.predict()
AnomalyResult
AnomalyAgent.update()
AnomalyFeedback
```

They should not depend on P2's internal feature extraction or model implementation.