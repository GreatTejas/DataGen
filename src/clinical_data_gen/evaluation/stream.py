"""Streaming evaluation utilities for PSV data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from clinical_data_gen.ingestion.psv_ingestor import PSVIngestor
from clinical_data_gen.state.patient_state_manager import PatientStateManager
from clinical_data_gen.state.patient_state import PatientState


@dataclass(frozen=True)
class EvaluationSample:
    """One state/ground-truth pair from the evaluation stream."""

    state: PatientState
    label: int


class PSVEvaluationStream:
    """
    Convert PSV files into an ordered stream of PatientState observations.

    The state manager retains enough previous observations for the configured
    feature-extraction history window.
    """

    def __init__(
        self,
        *,
        history_window: int = 6,
        max_forward_fill: int = 3,
    ) -> None:
        if history_window < 1:
            raise ValueError("history_window must be at least 1")

        if max_forward_fill < 0:
            raise ValueError("max_forward_fill must be non-negative")

        self.history_window = history_window
        self.max_forward_fill = max_forward_fill

    def samples(
        self,
        psv_path: str | Path,
    ) -> Iterator[EvaluationSample]:
        """
        Yield PatientState objects and their PSV SepsisLabel values.

        The label is evaluation-only ground truth. It is not stored inside
        PatientState.
        """
        state_manager = PatientStateManager(
            history_length=self.history_window,
            max_forward_fill=self.max_forward_fill,
        )

        ingestor = PSVIngestor(state_manager)

        for patient_id, row in self._iter_psv_rows(psv_path):
            state, label = ingestor.ingest_row(
                patient_id=patient_id,
                row=row,
            )

            yield EvaluationSample(
                state=state,
                label=int(label),
            )

    def _iter_psv_rows(
        self,
        psv_path: str | Path,
    ):
        """
        Read a PSV file and yield patient ID + parsed raw row.

        Patient ID is derived from the file name, matching the existing
        evaluation convention.
        """
        path = Path(psv_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"PSV file not found: {path}"
            )

        patient_id = path.stem

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            header_line = file.readline().strip()

            if not header_line:
                return

            headers = header_line.split("|")

            for line in file:
                line = line.strip()

                if not line:
                    continue

                values = line.split("|")

                if len(values) != len(headers):
                    raise ValueError(
                        f"Invalid PSV row in {path}: "
                        f"expected {len(headers)} columns, "
                        f"got {len(values)}"
                    )

                yield patient_id, dict(
                    zip(headers, values)
                )


def evaluate_psv_file(
    psv_path: str | Path,
    *,
    history_window: int = 6,
    max_forward_fill: int = 3,
) -> Iterator[EvaluationSample]:
    """
    Evaluate one PSV file as an ordered state stream.
    """
    stream = PSVEvaluationStream(
        history_window=history_window,
        max_forward_fill=max_forward_fill,
    )

    yield from stream.samples(psv_path)


def evaluate_psv_files(
    psv_files: list[str | Path],
    *,
    history_window: int = 6,
    max_forward_fill: int = 3,
) -> Iterator[EvaluationSample]:
    """
    Evaluate multiple PSV files sequentially.

    Each patient receives its own PatientStateManager, so history does not
    leak between patients.
    """
    for psv_file in psv_files:
        yield from evaluate_psv_file(
            psv_file,
            history_window=history_window,
            max_forward_fill=max_forward_fill,
        )


def discover_psv_files(
    input_path: str | Path,
) -> list[Path]:
    """
    Discover PSV files from either a single file or a directory.
    """
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() != ".psv":
            raise ValueError(
                f"Expected a .psv file: {path}"
            )

        return [path]

    if path.is_dir():
        files = sorted(
            path.glob("*.psv")
        )

        if not files:
            raise FileNotFoundError(
                f"No .psv files found in: {path}"
            )

        return files

    raise FileNotFoundError(
        f"Input path not found: {path}"
    )