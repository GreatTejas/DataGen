"""Online anomaly detection agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clinical_data_gen.anomaly.features import (
    FeatureVector,
    PatientFeatureExtractor,
)
from clinical_data_gen.learning.online_logistic_regression import (
    OnlineLogisticRegression,
)
from clinical_data_gen.state.patient_state import PatientState


@dataclass(frozen=True)
class AnomalyResult:
    """Result produced by the anomaly agent."""

    patient_id: str
    timestamp: Any
    probability: float
    is_anomalous: bool
    feature_vector: dict[str, float]
    contributing_features: list[str]
    model_version: int


class AnomalyAgent:
    """
    Online anomaly detection agent.

    Pipeline:

        PatientState
            ↓
        PatientFeatureExtractor
            ↓
        OnlineLogisticRegression
            ↓
        AnomalyResult

    The logistic-regression model is global.

    Patient-specific temporal context is supplied through
    PatientState.recent_history.
    """

    def __init__(
        self,
        *,
        history_window: int = 6,
        learning_rate: float = 0.05,
        l2: float = 1e-4,
        positive_class_weight: float = 1.0,
        gradient_clip: float | None = 5.0,
        threshold: float = 0.5,
        diminishing_learning_rate: bool = True,
    ) -> None:
        """
        Initialize the anomaly agent.

        Parameters
        ----------
        history_window:
            Number of previous observations used to construct temporal
            features.

        learning_rate:
            Initial learning rate for online logistic regression.

        l2:
            L2 regularization strength.

        positive_class_weight:
            Weight applied to positive examples during online updates.

        gradient_clip:
            Maximum absolute gradient component.

        threshold:
            Probability threshold used only by the anomaly agent to convert
            probability into a binary anomaly decision.

        diminishing_learning_rate:
            Whether the learning rate decreases as updates accumulate.
        """
        if history_window < 1:
            raise ValueError(
                "history_window must be at least 1"
            )

        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

        self.history_window = history_window
        self.threshold = threshold

        self.feature_extractor = PatientFeatureExtractor(
            history_window=history_window,
        )

        self.model = OnlineLogisticRegression(
            n_features=self.feature_extractor.feature_count,
            learning_rate=learning_rate,
            l2=l2,
            positive_class_weight=positive_class_weight,
            gradient_clip=gradient_clip,
            diminishing_learning_rate=diminishing_learning_rate,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        state: PatientState,
    ) -> AnomalyResult:
        """
        Predict anomaly probability for the current patient state.

        This method does not update the model.
        """
        features = self.feature_extractor.extract(state)

        prediction = self.model.predict(
            features.values,
        )

        return self._build_result(
            state=state,
            features=features,
            probability=prediction.probability,
            is_anomalous=(
                prediction.probability >= self.threshold
            ),
        )

    def predict_features(
        self,
        features: FeatureVector,
    ):
        """
        Predict directly from an already extracted feature vector.

        Useful for evaluation and debugging.
        """
        return self.model.predict(
            features.values,
        )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        state: PatientState,
        feedback: int | bool,
    ) -> AnomalyResult:
        """
        Update the online model using an explicit anomaly label.

        feedback must represent anomaly truth:

            1 / True  -> anomalous
            0 / False -> not anomalous

        Clinician actions such as accept/dismiss are intentionally not
        interpreted here as anomaly labels.
        """
        label = self._validate_label(
            feedback,
        )

        features = self.feature_extractor.extract(
            state,
        )

        prediction = self.model.predict(
            features.values,
        )

        self.model.update(
            features.values,
            label,
        )

        return self._build_result(
            state=state,
            features=features,
            probability=prediction.probability,
            is_anomalous=(
                prediction.probability >= self.threshold
            ),
        )

    def predict_and_update(
        self,
        *,
        state: PatientState,
        feedback: int | bool,
    ) -> AnomalyResult:
        """
        Predict and then update the model from an explicit anomaly label.
        """
        return self.update(
            state=state,
            feedback=feedback,
        )

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        *,
        state: PatientState,
        features: FeatureVector,
        probability: float,
        is_anomalous: bool,
    ) -> AnomalyResult:
        """Construct the public anomaly result."""
        patient_id = getattr(
            state,
            "patient_id",
            "",
        )

        timestamp = getattr(
            state,
            "timestamp",
            None,
        )

        return AnomalyResult(
            patient_id=str(patient_id),
            timestamp=timestamp,
            probability=float(probability),
            is_anomalous=bool(is_anomalous),
            feature_vector=features.as_dict(),
            contributing_features=(
                self._top_contributing_features(
                    features,
                )
            ),
            model_version=self.model.model_version,
        )

    # ------------------------------------------------------------------
    # Feature attribution
    # ------------------------------------------------------------------

    def _top_contributing_features(
        self,
        features: FeatureVector,
        top_k: int = 5,
    ) -> list[str]:
        """
        Return features with the largest absolute linear contribution.

        Contribution is approximated as:

            weight × feature_value
        """
        weights = self.model.weights

        if not weights:
            return []

        contributions: list[tuple[str, float]] = []

        for index, (name, value) in enumerate(
            zip(
                features.names,
                features.values,
            )
        ):
            if index >= len(weights):
                break

            contribution = (
                weights[index] * value
            )

            contributions.append(
                (
                    name,
                    contribution,
                )
            )

        contributions.sort(
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        return [
            name
            for name, _ in contributions[:top_k]
        ]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_label(
        feedback: int | bool,
    ) -> int:
        """Validate and normalize a binary anomaly label."""
        if isinstance(feedback, bool):
            return int(feedback)

        if isinstance(feedback, int) and feedback in (0, 1):
            return feedback

        raise ValueError(
            "feedback must be a binary anomaly label: "
            "0, 1, False, or True"
        )