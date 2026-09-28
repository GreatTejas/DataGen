"""Evaluate the online logistic-regression anomaly model on PSV data."""

from __future__ import annotations

import argparse
import json
import time
from math import log
from pathlib import Path

from clinical_data_gen.anomaly.anomaly_agent import AnomalyAgent
from clinical_data_gen.evaluation.online_metrics import OnlineMetricsEvaluator
from clinical_data_gen.evaluation.stream import (
    EvaluationSample,
    discover_psv_files,
    evaluate_psv_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate online logistic regression on PSV data "
            "with configurable temporal history."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="PSV file or directory containing PSV files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory where evaluation results will be written.",
    )

    parser.add_argument(
        "--history-window",
        type=int,
        default=6,
        help=(
            "Number of previous observations used to construct "
            "temporal features. Default: 6."
        ),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Initial online learning rate. Default: 0.05.",
    )

    parser.add_argument(
        "--l2",
        type=float,
        default=1e-4,
        help="L2 regularization strength. Default: 1e-4.",
    )

    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=1.0,
        help="Weight applied to positive examples. Default: 1.0.",
    )

    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=5.0,
        help="Maximum absolute gradient component. Default: 5.0.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly probability threshold. Default: 0.5.",
    )

    parser.add_argument(
        "--constant-learning-rate",
        action="store_true",
        help="Disable diminishing learning rate.",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help=(
            "Print progress after this many observations. "
            "Default: 10000."
        ),
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional maximum number of patients to evaluate.",
    )

    parser.add_argument(
        "--max-forward-fill",
        type=int,
        default=3,
        help=(
            "Maximum forward-fill length for supported vitals. "
            "Default: 3."
        ),
    )

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.history_window < 1:
        raise ValueError(
            "--history-window must be at least 1"
        )

    if args.max_forward_fill < 0:
        raise ValueError(
            "--max-forward-fill must be non-negative"
        )

    if args.progress_every < 1:
        raise ValueError(
            "--progress-every must be at least 1"
        )

    if args.max_patients is not None and args.max_patients < 1:
        raise ValueError(
            "--max-patients must be at least 1"
        )

    if args.positive_class_weight <= 0:
        raise ValueError(
            "--positive-class-weight must be greater than 0"
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be greater than 0"
        )

    if args.l2 < 0:
        raise ValueError(
            "--l2 must be non-negative"
        )

    if args.gradient_clip is not None and args.gradient_clip <= 0:
        raise ValueError(
            "--gradient-clip must be greater than 0"
        )

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1"
        )


def _create_agent(args: argparse.Namespace) -> AnomalyAgent:
    """Create the anomaly agent using CLI configuration."""
    return AnomalyAgent(
        history_window=args.history_window,
        learning_rate=args.learning_rate,
        l2=args.l2,
        positive_class_weight=args.positive_class_weight,
        gradient_clip=args.gradient_clip,
        threshold=args.threshold,
        diminishing_learning_rate=not args.constant_learning_rate,
    )


def _evaluate_patient(
    *,
    psv_file: Path,
    agent: AnomalyAgent,
    evaluator: OnlineMetricsEvaluator,
    history_window: int,
    max_forward_fill: int,
    progress_every: int,
    processed_observations: int,
    total_observations: int,
    start_time: float,
) -> tuple[int, int]:
    """
    Evaluate one patient sequentially.

    For every observation:

        1. construct PatientState
        2. predict using current state + historical context
        3. record prediction against evaluation-only label
        4. update online model using the label

    The label is used only for offline/prequential evaluation.
    """
    patient_observations = 0

    for sample in evaluate_psv_file(
        psv_file,
        history_window=history_window,
        max_forward_fill=max_forward_fill,
    ):
        _process_sample(
            sample=sample,
            agent=agent,
            evaluator=evaluator,
        )

        patient_observations += 1
        processed_observations += 1

        if processed_observations % progress_every == 0:
            _print_progress(
                evaluator=evaluator,
                processed_observations=processed_observations,
                total_observations=total_observations,
                start_time=start_time,
            )

    return patient_observations, processed_observations


def _binary_log_loss(
    probability: float,
    label: int,
) -> float:
    """
    Calculate binary cross-entropy for one prediction.

    This matches the definition used by OnlineMetricsEvaluator.
    """
    epsilon = 1e-15

    probability = min(
        max(probability, epsilon),
        1.0 - epsilon,
    )

    if label == 1:
        return -log(probability)

    return -log(1.0 - probability)


def _process_sample(
    *,
    sample: EvaluationSample,
    agent: AnomalyAgent,
    evaluator: OnlineMetricsEvaluator,
) -> None:
    """
    Perform one prequential prediction/update step.

    Prediction happens before the label is used for updating the model.
    """
    # ---------------------------------------------------------------
    # 1. Predict using the model BEFORE learning from this label.
    # ---------------------------------------------------------------
    result = agent.predict(sample.state)

    # Capture the model version associated with this prediction.
    model_version = agent.model.model_version

    # Calculate the loss of this prediction against the evaluation label.
    loss = _binary_log_loss(
        result.probability,
        sample.label,
    )

    # ---------------------------------------------------------------
    # 2. Record the prediction.
    #
    # The threshold is used only to derive the binary predicted label
    # for evaluation. It does not affect the probability itself.
    # ---------------------------------------------------------------
    evaluator.add_prediction(
        probability=result.probability,
        true_label=sample.label,
        patient_id=sample.state.patient_id,
        timestamp=sample.state.timestamp,
        loss=loss,
        model_version=model_version,
        threshold=agent.threshold,
    )

    # ---------------------------------------------------------------
    # 3. Update the online model AFTER recording the prediction.
    #
    # SepsisLabel is used here only as the simulated delayed label for
    # offline/prequential evaluation. It is NOT clinician feedback.
    # ---------------------------------------------------------------
    agent.model.update(
        agent.feature_extractor.extract(sample.state).values,
        sample.label,
    )


def _print_progress(
    *,
    evaluator: OnlineMetricsEvaluator,
    processed_observations: int,
    total_observations: int,
    start_time: float,
) -> None:
    """Print lightweight progress information."""
    elapsed = max(time.perf_counter() - start_time, 1e-9)

    observations_per_second = (
        processed_observations / elapsed
    )

    metrics = evaluator.compute()

    metrics = evaluator.compute()

    pr_auc = metrics.pr_auc
    precision = metrics.precision
    recall = metrics.recall
    f1 = metrics.f1
    mean_loss = metrics.mean_loss

    progress = (
        processed_observations / total_observations
        if total_observations > 0
        else 0.0
    )

    eta_seconds = (
        (total_observations - processed_observations)
        / observations_per_second
        if observations_per_second > 0
        else 0.0
    )

    print(
        f"[{progress:6.1%}] "
        f"obs={processed_observations:,}/{total_observations:,} "
        f"PR-AUC={_format_metric(pr_auc)} "
        f"P={_format_metric(precision)} "
        f"R={_format_metric(recall)} "
        f"F1={_format_metric(f1)} "
        f"loss={_format_metric(mean_loss)} "
        f"speed={observations_per_second:,.1f}/s "
        f"ETA={eta_seconds / 60.0:.1f}m"
    )


def _format_metric(value: object) -> str:
    """Format a metric safely for console output."""
    if value is None:
        return "N/A"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _count_observations(
    psv_files: list[Path],
) -> int:
    """Count non-empty data rows across the selected PSV files."""
    total = 0

    for path in psv_files:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            # Skip header.
            next(file, None)

            total += sum(
                1
                for line in file
                if line.strip()
            )

    return total


def _save_predictions(
    *,
    output_path: Path,
    evaluator: OnlineMetricsEvaluator,
) -> None:
    """Write prediction records to JSONL."""
    predictions_path = output_path / "predictions.jsonl"

    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in evaluator.records:
            file.write(
                json.dumps(
                    {
                        "patient_id": record.patient_id,
                        "timestamp": record.timestamp,
                        "probability": record.probability,
                        "predicted_label": record.predicted_label,
                        "true_label": record.true_label,
                        "loss": record.loss,
                        "model_version": record.model_version,
                    }
                )
                + "\n"
            )


def _save_results(
    *,
    output_path: Path,
    evaluator: OnlineMetricsEvaluator,
    args: argparse.Namespace,
) -> None:
    """Save metrics and experiment configuration."""
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = evaluator.compute()

    with (
        output_path / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics.as_dict(),
            file,
            indent=2,
        )

    _save_predictions(
        output_path=output_path,
        evaluator=evaluator,
    )

    config = {
        "input": str(args.input),
        "history_window": args.history_window,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "positive_class_weight": args.positive_class_weight,
        "gradient_clip": args.gradient_clip,
        "threshold": args.threshold,
        "diminishing_learning_rate": (
            not args.constant_learning_rate
        ),
        "max_forward_fill": args.max_forward_fill,
        "max_patients": args.max_patients,
    }

    with (
        output_path / "config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
        )


def main() -> None:
    args = parse_args()

    _validate_args(args)

    input_path = Path(args.input)
    output_path = Path(args.output)

    psv_files = discover_psv_files(
        input_path,
    )

    if args.max_patients is not None:
        psv_files = psv_files[
            : args.max_patients
        ]

    if not psv_files:
        raise ValueError(
            "No PSV files selected for evaluation."
        )

    total_observations = _count_observations(
        psv_files,
    )

    agent = _create_agent(args)

    evaluator = OnlineMetricsEvaluator()

    processed_observations = 0
    start_time = time.perf_counter()

    print()
    print("=" * 72)
    print("Online Logistic Regression Evaluation")
    print("=" * 72)
    print(f"Input              : {input_path}")
    print(f"Patients           : {len(psv_files):,}")
    print(f"Observations       : {total_observations:,}")
    print(f"History window     : {args.history_window}")
    print(f"Learning rate      : {args.learning_rate}")
    print(f"L2                 : {args.l2}")
    print(f"Positive weight    : {args.positive_class_weight}")
    print(f"Threshold          : {args.threshold}")
    print(
        "Learning rate mode : "
        + (
            "diminishing"
            if not args.constant_learning_rate
            else "constant"
        )
    )
    print("=" * 72)
    print()

    for patient_index, psv_file in enumerate(
        psv_files,
        start=1,
    ):
        _, processed_observations = _evaluate_patient(
            psv_file=psv_file,
            agent=agent,
            evaluator=evaluator,
            history_window=args.history_window,
            max_forward_fill=args.max_forward_fill,
            progress_every=args.progress_every,
            processed_observations=processed_observations,
            total_observations=total_observations,
            start_time=start_time,
        )

        if patient_index % max(
            1,
            min(100, len(psv_files)),
        ) == 0:
            print(
                f"Patients processed: "
                f"{patient_index:,}/{len(psv_files):,}"
            )

    elapsed = time.perf_counter() - start_time

    _save_results(
        output_path=output_path,
        evaluator=evaluator,
        args=args,
    )

    metrics = evaluator.compute()

    print()
    print("=" * 72)
    print("Final Results")
    print("=" * 72)

    print(
        f"Observations   : "
        f"{len(evaluator.records):,}"
    )

    print(
        f"Positive       : "
        f"{sum(r.true_label for r in evaluator.records):,}"
    )

    print(
        f"Negative       : "
        f"{sum(1 - r.true_label for r in evaluator.records):,}"
    )

    for name in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "negative_predictive_value",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "alerts_per_patient_hour",
    ):
        value = getattr(metrics, name)

        print(
            f"{name.replace('_', ' ').title():<28}: "
            f"{_format_metric(value)}"
        )

    print(
        f"Runtime          : "
        f"{elapsed / 60.0:.2f} minutes"
    )

    print(
        f"Output            : "
        f"{output_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()