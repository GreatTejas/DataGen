"""Feature extraction for the online anomaly detection model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clinical_data_gen.preprocessing.constants import CORE_VITALS
from clinical_data_gen.state.patient_state import PatientState


@dataclass(frozen=True)
class FeatureVector:
    """Numeric feature vector consumed by the anomaly model."""

    values: list[float]
    names: list[str]

    def as_dict(self) -> dict[str, float]:
        """Return feature names mapped to feature values."""
        return dict(zip(self.names, self.values))

    def __len__(self) -> int:
        return len(self.values)


class PatientFeatureExtractor:
    """
    Convert PatientState into a fixed-dimensional temporal feature vector.

    For each of the seven critical vitals, the model receives:

    - current normalized value
    - previous normalized values at t-1 ... t-history_window
    - current missing flag
    - current imputed flag

    Static patient/context features are also included.

    Historical positions are preserved. A missing observation at t-k does
    not cause older observations to shift forward into that lag.

    The history window contains previous patient states only.
    The current state is handled separately.
    """

    NORMALIZATION: dict[str, tuple[float, float]] = {
        "HR": (80.0, 30.0),
        "O2Sat": (97.0, 5.0),
        "Temp": (37.0, 1.0),
        "SBP": (120.0, 30.0),
        "MAP": (80.0, 20.0),
        "DBP": (70.0, 20.0),
        "Resp": (18.0, 8.0),
    }

    STATIC_FEATURES = ("Age", "Gender", "Unit1", "Unit2")

    def __init__(self, history_window: int = 6) -> None:
        if history_window < 1:
            raise ValueError("history_window must be at least 1")

        self.history_window = history_window

    def extract(self, state: PatientState) -> FeatureVector:
        """
        Extract the complete feature vector from the current patient state.

        Feature ordering is deterministic:

        For every critical vital:
            <vital>_current
            <vital>_lag_1
            ...
            <vital>_lag_N
            <vital>_missing
            <vital>_imputed

        Followed by:
            Age
            Gender
            Unit1
            Unit2
        """
        values: list[float] = []
        names: list[str] = []

        for vital_name in CORE_VITALS:
            current_value = self._get_current_value(
                state,
                vital_name,
            )

            current_missing = self._is_current_missing(
                state,
                vital_name,
            )

            current_imputed = self._is_current_imputed(
                state,
                vital_name,
            )

            # Current value.
            values.append(
                self._normalize(
                    vital_name,
                    current_value,
                )
            )
            names.append(f"{vital_name}_current")

            # Explicit historical lag values.
            history_values = self._get_history_values(
                state,
                vital_name,
            )

            for lag in range(1, self.history_window + 1):
                lag_value = self._get_lag_value(
                    history_values,
                    lag,
                )

                values.append(
                    self._normalize(
                        vital_name,
                        lag_value,
                    )
                )
                names.append(f"{vital_name}_lag_{lag}")

            # Current missingness / imputation indicators.
            values.append(float(current_missing))
            names.append(f"{vital_name}_missing")

            values.append(float(current_imputed))
            names.append(f"{vital_name}_imputed")

        # Static patient/context features.
        static_values = self._extract_static_context(state)

        for name, value in static_values:
            values.append(value)
            names.append(name)

        return FeatureVector(
            values=values,
            names=names,
        )

    # ------------------------------------------------------------------
    # Current-state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_value(
        state: PatientState,
        vital_name: str,
    ) -> float | None:
        observation = state.current_vitals.get(vital_name)

        if observation is None:
            return None

        if observation.value is None:
            return None

        try:
            return float(observation.value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_current_missing(
        state: PatientState,
        vital_name: str,
    ) -> bool:
        observation = state.current_vitals.get(vital_name)

        if observation is None:
            return True

        return (
            observation.value is None
            or observation.is_missing
        )

    @staticmethod
    def _is_current_imputed(
        state: PatientState,
        vital_name: str,
    ) -> bool:
        observation = state.current_vitals.get(vital_name)

        if observation is None:
            return False

        return observation.is_imputed

    # ------------------------------------------------------------------
    # Historical lag helpers
    # ------------------------------------------------------------------

    def _get_history_values(
        self,
        state: PatientState,
        vital_name: str,
    ) -> list[float | None]:
        """
        Return historical values in chronological order.

        The returned list contains the most recent history_window
        observations available in PatientState.recent_history.

        Missing values are retained as None so that lag positions remain
        aligned with the actual observation sequence.
        """
        history = list(state.recent_history)

        if not history:
            return []

        history = history[-self.history_window :]

        return [
            self._get_snapshot_value(
                snapshot,
                vital_name,
            )
            for snapshot in history
        ]

    @staticmethod
    def _get_lag_value(
        history_values: list[float | None],
        lag: int,
    ) -> float | None:
        """
        Return the value corresponding to a specific lag.

        history_values is chronological:
            [oldest, ..., previous]

        Therefore:
            lag=1 -> most recent previous observation
            lag=2 -> second most recent observation
            ...
        """
        if lag < 1:
            raise ValueError("lag must be at least 1")

        if lag > len(history_values):
            return None

        index = len(history_values) - lag
        return history_values[index]

    @staticmethod
    def _get_snapshot_value(
        snapshot: Any,
        vital_name: str,
    ) -> float | None:
        """
        Extract a vital value from a historical PatientState snapshot.

        Supports both PatientState-like snapshots and dictionary snapshots.
        """
        current_vitals = getattr(
            snapshot,
            "current_vitals",
            None,
        )

        if current_vitals is None and isinstance(snapshot, dict):
            current_vitals = snapshot.get("current_vitals")

        if not current_vitals:
            return None

        observation = current_vitals.get(vital_name)

        if observation is None:
            return None

        if isinstance(observation, dict):
            value = observation.get("value")

            if value is None:
                return None

            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        value = getattr(
            observation,
            "value",
            None,
        )

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Static context
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_static_context(
        state: PatientState,
    ) -> list[tuple[str, float]]:
        context = state.static_context

        return [
            (
                "Age",
                PatientFeatureExtractor._to_float(
                    getattr(context, "age", None),
                ),
            ),
            (
                "Gender",
                PatientFeatureExtractor._to_float(
                    getattr(context, "gender", None),
                ),
            ),
            (
                "Unit1",
                PatientFeatureExtractor._to_float(
                    getattr(context, "unit1", None),
                ),
            ),
            (
                "Unit2",
                PatientFeatureExtractor._to_float(
                    getattr(context, "unit2", None),
                ),
            ),
        ]

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None:
            return 0.0

        if isinstance(value, bool):
            return float(value)

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @classmethod
    def _normalize(
        cls,
        vital_name: str,
        value: float | None,
    ) -> float:
        """
        Normalize a vital using the configured physiological center/scale.

        Missing values are represented as 0.0 here. Their missingness is
        separately represented by the <vital>_missing feature.
        """
        if value is None:
            return 0.0

        center_scale = cls.NORMALIZATION.get(vital_name)

        if center_scale is None:
            return float(value)

        center, scale = center_scale

        if scale <= 0:
            raise ValueError(
                f"Normalization scale must be positive for {vital_name}"
            )

        return (float(value) - center) / scale

    # ------------------------------------------------------------------
    # Model dimensionality
    # ------------------------------------------------------------------

    @property
    def feature_count(self) -> int:
        """
        Number of features produced by this extractor.

        Per vital:
            1 current value
            history_window lag values
            1 missing flag
            1 imputed flag

        Plus four static features.
        """
        per_vital = (
            1
            + self.history_window
            + 2
        )

        return (
            len(CORE_VITALS) * per_vital
            + len(self.STATIC_FEATURES)
        )