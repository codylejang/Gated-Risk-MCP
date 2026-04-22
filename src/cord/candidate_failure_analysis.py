from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


CANDIDATE_LIST_COLUMNS = ("field", "candidates", "prediction", "ground_truth")
RECOVERABILITY_COLUMNS = (
    "field_name",
    "predicted_text",
    "gold_text",
    "recoverable_by_candidates",
    "correct",
)
FAILURE_MODEL = "model_mistake"
FAILURE_MISSING_CANDIDATE = "missing_candidate"
FAILURE_NONE = "correct"


Normalizer = Callable[[object], str]


@dataclass(frozen=True)
class FailureAnalysisResult:
    """Container for candidate-recovery and model-selection diagnostics."""

    annotated_rows: pd.DataFrame
    overall: pd.Series
    by_field: pd.DataFrame
    model_failures: pd.DataFrame
    missing_candidate_failures: pd.DataFrame


def normalize_receipt_value(value: object) -> str:
    """Normalize OCR labels enough for robust exact matching.

    This intentionally stays conservative: it trims whitespace, collapses
    repeated whitespace, and compares case-insensitively. Domain-specific
    canonicalization, such as date parsing or currency normalization, can be
    injected through ``analyze_candidate_failures(normalizer=...)``.
    """

    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().casefold().split())


def analyze_candidate_failures(
    df: pd.DataFrame,
    *,
    normalizer: Normalizer = normalize_receipt_value,
    example_rows: int = 5,
    print_examples: bool = True,
) -> FailureAnalysisResult:
    """Analyze whether extraction errors come from ranking or generation.

    Expected input columns:
    - ``field``: receipt field name, such as company, date, address, or total
    - ``candidates``: iterable of candidate strings generated for that field
    - ``prediction``: model-selected string
    - ``ground_truth``: labeled correct string

    Returns a ``FailureAnalysisResult`` with an annotated row-level DataFrame,
    overall metrics, per-field metrics, and example failure subsets.
    """

    _validate_input(df)

    annotated = df.copy()
    annotated["prediction_normalized"] = annotated["prediction"].map(normalizer)
    annotated["ground_truth_normalized"] = annotated["ground_truth"].map(normalizer)
    annotated["normalized_candidates"] = annotated["candidates"].map(
        lambda values: _normalize_candidates(values, normalizer)
    )

    annotated["is_correct"] = (
        annotated["prediction_normalized"] == annotated["ground_truth_normalized"]
    )
    annotated["ground_truth_in_candidates"] = annotated.apply(
        lambda row: row["ground_truth_normalized"] in row["normalized_candidates"],
        axis=1,
    )
    annotated["failure_type"] = annotated.apply(_failure_type, axis=1)

    overall = _summarize(annotated)
    by_field = _summarize_by_field(annotated)

    model_failures = _example_failures(annotated, FAILURE_MODEL, example_rows)
    missing_candidate_failures = _example_failures(
        annotated, FAILURE_MISSING_CANDIDATE, example_rows
    )

    if print_examples:
        print_failure_analysis(
            overall=overall,
            by_field=by_field,
            model_failures=model_failures,
            missing_candidate_failures=missing_candidate_failures,
        )

    return FailureAnalysisResult(
        annotated_rows=annotated,
        overall=overall,
        by_field=by_field,
        model_failures=model_failures,
        missing_candidate_failures=missing_candidate_failures,
    )


def analyze_prediction_outputs(
    df: pd.DataFrame,
    *,
    example_rows: int = 5,
    print_examples: bool = True,
) -> FailureAnalysisResult:
    """Analyze existing extraction output files.

    This supports repo outputs such as
    ``Outputs/sroie_extraction/*/validation_field_predictions.csv`` where the
    candidate list itself is not saved, but ``recoverable_by_candidates`` tells
    us whether the gold answer was present in the generated candidates.
    """

    _validate_recoverability_input(df)

    annotated = df.copy()
    annotated["field"] = annotated["field_name"]
    annotated["prediction"] = annotated["predicted_text"]
    annotated["ground_truth"] = annotated["gold_text"]
    annotated["is_correct"] = annotated["correct"].map(_parse_bool)
    annotated["ground_truth_in_candidates"] = annotated["recoverable_by_candidates"].map(
        _parse_bool
    )
    annotated["failure_type"] = annotated.apply(_failure_type, axis=1)

    overall = _summarize(annotated)
    by_field = _summarize_by_field(annotated)
    model_failures = _example_failures(annotated, FAILURE_MODEL, example_rows)
    missing_candidate_failures = _example_failures(
        annotated, FAILURE_MISSING_CANDIDATE, example_rows
    )

    if print_examples:
        print_failure_analysis(
            overall=overall,
            by_field=by_field,
            model_failures=model_failures,
            missing_candidate_failures=missing_candidate_failures,
        )

    return FailureAnalysisResult(
        annotated_rows=annotated,
        overall=overall,
        by_field=by_field,
        model_failures=model_failures,
        missing_candidate_failures=missing_candidate_failures,
    )


def print_failure_analysis(
    *,
    overall: pd.Series,
    by_field: pd.DataFrame,
    model_failures: pd.DataFrame,
    missing_candidate_failures: pd.DataFrame,
) -> None:
    """Print a readable console report for the failure analysis result."""

    print("Overall")
    print(overall.to_string())
    print()

    print("Breakdown by field")
    print(by_field.to_string(index=False))
    print()

    print("Example model failures")
    _print_examples(model_failures)
    print()

    print("Example missing-candidate failures")
    _print_examples(missing_candidate_failures)


def load_candidate_frame(path: Path, *, candidate_separator: str | None = None) -> pd.DataFrame:
    """Load a failure-analysis CSV and parse the candidates column.

    The ``candidates`` column may be saved as a JSON/Python list string, such as
    ``["A", "B"]``, or as a delimited string when ``candidate_separator`` is
    provided, such as ``A|B``.
    """

    df = pd.read_csv(path)
    if "candidates" not in df.columns:
        raise ValueError("Input CSV must include a 'candidates' column.")

    df["candidates"] = df["candidates"].map(
        lambda value: parse_candidates(value, separator=candidate_separator)
    )
    return df


def load_analysis_frame(
    path: Path,
    *,
    candidate_separator: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load either candidate-list input or existing prediction-output input."""

    df = pd.read_csv(path)
    if _has_columns(df, CANDIDATE_LIST_COLUMNS):
        df["candidates"] = df["candidates"].map(
            lambda value: parse_candidates(value, separator=candidate_separator)
        )
        return df, "candidate_list"
    if _has_columns(df, RECOVERABILITY_COLUMNS):
        return df, "recoverability"

    raise ValueError(
        "Input CSV does not match a supported format. Expected either "
        f"{list(CANDIDATE_LIST_COLUMNS)} or repo prediction-output columns "
        f"{list(RECOVERABILITY_COLUMNS)}."
    )


def save_failure_analysis(
    result: FailureAnalysisResult,
    output_dir: Path,
) -> None:
    """Write analysis artifacts to CSV/JSON files."""

    output_dir.mkdir(parents=True, exist_ok=True)

    result.overall.to_frame("value").to_csv(output_dir / "overall_metrics.csv")
    result.by_field.to_csv(output_dir / "field_breakdown.csv", index=False)
    result.model_failures.to_csv(output_dir / "model_failures_examples.csv", index=False)
    result.missing_candidate_failures.to_csv(
        output_dir / "missing_candidate_examples.csv", index=False
    )

    annotated = result.annotated_rows.copy()
    if "normalized_candidates" in annotated.columns:
        annotated["normalized_candidates"] = annotated["normalized_candidates"].map(
            lambda values: json.dumps(sorted(values))
        )
    annotated.to_csv(output_dir / "annotated_rows.csv", index=False)


def parse_candidates(value: object, *, separator: str | None = None) -> list[str]:
    """Parse candidate cells from CSV-friendly formats."""

    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if separator is not None:
        return [part.strip() for part in text.split(separator) if part.strip()]

    parsed = _parse_serialized_list(text)
    if parsed is not None:
        return [str(item) for item in parsed]

    raise ValueError(
        "Could not parse candidates cell. Store candidates as a JSON/Python list "
        "like '[\"A\", \"B\"]', or pass --candidate-separator for delimited text."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze OCR candidate extraction failures."
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "CSV with columns: field, candidates, prediction, ground_truth. "
            "Existing repo prediction outputs are also supported."
        ),
    )
    parser.add_argument(
        "--candidate-separator",
        default=None,
        help="Separator for delimited candidate strings, for example '|'.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of example failures to print and save.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for CSV outputs.",
    )
    args = parser.parse_args()

    df, input_mode = load_analysis_frame(
        args.input_csv,
        candidate_separator=args.candidate_separator,
    )

    if input_mode == "candidate_list":
        result = analyze_candidate_failures(df, example_rows=args.examples)
    else:
        result = analyze_prediction_outputs(df, example_rows=args.examples)

    if args.output_dir is not None:
        save_failure_analysis(result, args.output_dir)
        print()
        print(f"Saved analysis files to {args.output_dir}")


def _validate_input(df: pd.DataFrame) -> None:
    missing_columns = [column for column in CANDIDATE_LIST_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    invalid_candidates = ~df["candidates"].map(_is_candidate_iterable)
    if invalid_candidates.any():
        bad_indices = df.index[invalid_candidates].tolist()[:10]
        raise ValueError(
            "'candidates' must contain an iterable of strings for every row. "
            f"Invalid row indices: {bad_indices}"
        )


def _validate_recoverability_input(df: pd.DataFrame) -> None:
    missing_columns = [
        column for column in RECOVERABILITY_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _has_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> bool:
    return all(column in df.columns for column in columns)


def _is_candidate_iterable(value: object) -> bool:
    if value is None or isinstance(value, str):
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _normalize_candidates(values: Iterable[object], normalizer: Normalizer) -> set[str]:
    return {normalizer(value) for value in values}


def _failure_type(row: pd.Series) -> str:
    if row["is_correct"]:
        return FAILURE_NONE
    if row["ground_truth_in_candidates"]:
        return FAILURE_MODEL
    return FAILURE_MISSING_CANDIDATE


def _summarize(frame: pd.DataFrame) -> pd.Series:
    n_rows = len(frame)
    n_errors = int((~frame["is_correct"]).sum())
    model_errors = int((frame["failure_type"] == FAILURE_MODEL).sum())
    missing_candidate_errors = int(
        (frame["failure_type"] == FAILURE_MISSING_CANDIDATE).sum()
    )

    return pd.Series(
        {
            "rows": n_rows,
            "accuracy": _safe_rate(int(frame["is_correct"].sum()), n_rows),
            "errors": n_errors,
            "model_mistake_errors": model_errors,
            "missing_candidate_errors": missing_candidate_errors,
            "pct_errors_model_mistake": _safe_rate(model_errors, n_errors),
            "pct_errors_missing_candidate": _safe_rate(
                missing_candidate_errors, n_errors
            ),
            "candidate_recall": _safe_rate(
                int(frame["ground_truth_in_candidates"].sum()), n_rows
            ),
        }
    )


def _summarize_by_field(annotated: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"field": field, **_summarize(group).to_dict()}
            for field, group in annotated.groupby("field", dropna=False)
        ]
    ).sort_values(["accuracy", "field"], ascending=[True, True], ignore_index=True)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _example_failures(
    annotated: pd.DataFrame, failure_type: str, example_rows: int
) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "doc_id",
            "field",
            "prediction",
            "ground_truth",
            "candidates",
            "candidate_count",
            "confidence",
            "margin",
            "match_score",
            "failure_type",
        ]
        if column in annotated.columns
    ]
    return annotated.loc[annotated["failure_type"] == failure_type, columns].head(
        example_rows
    )


def _print_examples(examples: pd.DataFrame) -> None:
    if examples.empty:
        print("(none)")
        return
    print(examples.to_string(index=False))


def _parse_serialized_list(text: str) -> list[object] | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (SyntaxError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


if __name__ == "__main__":
    main()
