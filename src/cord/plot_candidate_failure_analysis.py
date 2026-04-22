from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_REPORT_FILES = ("field_breakdown.csv", "annotated_rows.csv")
FAILURE_TYPES = ("model_mistake", "missing_candidate")


def plot_failure_analysis(report_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Create diagnostic tables and curves from failure-analysis outputs."""

    _validate_report_dir(report_dir)
    output_dir = output_dir or report_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    field_breakdown = pd.read_csv(report_dir / "field_breakdown.csv")
    annotated_rows = pd.read_csv(report_dir / "annotated_rows.csv")
    annotated_rows = _prepare_annotated_rows(annotated_rows)

    paths = [
        _plot_field_diagnostic_table(field_breakdown, output_dir),
        _plot_field_failure_rates(field_breakdown, output_dir),
        _plot_confidence_error_curve(annotated_rows, output_dir),
        _plot_margin_error_curve(annotated_rows, output_dir),
        _plot_failure_examples_table(annotated_rows, output_dir),
    ]
    return paths


def _plot_field_diagnostic_table(field_breakdown: pd.DataFrame, output_dir: Path) -> Path:
    table = field_breakdown.copy()
    table["accuracy"] = table["accuracy"].map(_format_pct)
    table["candidate_recall"] = table["candidate_recall"].map(_format_pct)
    table["model_error_share"] = table["pct_errors_model_mistake"].map(_format_pct)
    table["missing_candidate_share"] = table["pct_errors_missing_candidate"].map(
        _format_pct
    )

    display = table[
        [
            "field",
            "rows",
            "errors",
            "accuracy",
            "candidate_recall",
            "model_mistake_errors",
            "missing_candidate_errors",
            "model_error_share",
            "missing_candidate_share",
        ]
    ].sort_values("errors", ascending=False)
    display[["rows", "errors", "model_mistake_errors", "missing_candidate_errors"]] = (
        display[["rows", "errors", "model_mistake_errors", "missing_candidate_errors"]]
        .astype(int)
        .astype(str)
    )

    fig, ax = plt.subplots(figsize=(14, 3.2))
    ax.axis("off")
    table_artist = ax.table(
        cellText=display.values,
        colLabels=[
            "Field",
            "Rows",
            "Errors",
            "Accuracy",
            "Cand. recall",
            "Model errs",
            "Missing errs",
            "Model share",
            "Missing share",
        ],
        loc="center",
        cellLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(9)
    table_artist.scale(1, 1.55)

    for (row, _column), cell in table_artist.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#333333")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f5f7")

    ax.set_title("Field Failure Diagnostics", weight="bold", pad=12)
    fig.tight_layout()

    path = output_dir / "field_diagnostic_table.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_field_failure_rates(field_breakdown: pd.DataFrame, output_dir: Path) -> Path:
    table = field_breakdown.copy()
    table["error_rate"] = 1.0 - table["accuracy"]
    table["model_error_rate"] = table["model_mistake_errors"] / table["rows"]
    table["missing_candidate_rate"] = table["missing_candidate_errors"] / table["rows"]
    table = table.sort_values("error_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        table["field"],
        table["error_rate"],
        marker="o",
        linewidth=2.5,
        label="All errors",
        color="#222222",
    )
    ax.plot(
        table["field"],
        table["model_error_rate"],
        marker="o",
        linewidth=2,
        label="Model mistakes",
        color="#4c78a8",
    )
    ax.plot(
        table["field"],
        table["missing_candidate_rate"],
        marker="o",
        linewidth=2,
        label="Missing candidates",
        color="#f58518",
    )

    ax.set_title("Failure Rate by Field")
    ax.set_xlabel("Field")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(0.05, table["error_rate"].max() * 1.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = output_dir / "field_failure_rates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_confidence_error_curve(annotated_rows: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "confidence_error_curve.png"
    if "confidence" not in annotated_rows.columns:
        return _save_missing_metric_plot(path, "confidence")

    thresholds = _thresholds(annotated_rows["confidence"])
    curve = _threshold_curve(annotated_rows, "confidence", thresholds)
    return _plot_threshold_curve(
        curve,
        x_column="threshold",
        title="Errors Above Confidence Threshold",
        xlabel="Minimum confidence",
        output_path=path,
    )


def _plot_margin_error_curve(annotated_rows: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "margin_error_curve.png"
    if "margin" not in annotated_rows.columns:
        return _save_missing_metric_plot(path, "margin")

    thresholds = _thresholds(annotated_rows["margin"])
    curve = _threshold_curve(annotated_rows, "margin", thresholds)
    return _plot_threshold_curve(
        curve,
        x_column="threshold",
        title="Errors Above Margin Threshold",
        xlabel="Minimum margin",
        output_path=path,
    )


def _plot_failure_examples_table(annotated_rows: pd.DataFrame, output_dir: Path) -> Path:
    failures = annotated_rows.loc[
        annotated_rows["failure_type"].isin(FAILURE_TYPES)
    ].copy()
    failures["error_priority"] = failures["failure_type"].map(
        {"model_mistake": 0, "missing_candidate": 1}
    )
    sort_columns = [
        column
        for column in ["error_priority", "field", "confidence", "candidate_count"]
        if column in failures.columns
    ]
    ascending = [True, True, False, False][: len(sort_columns)]
    if sort_columns:
        failures = failures.sort_values(sort_columns, ascending=ascending)

    columns = [
        column
        for column in [
            "doc_id",
            "field",
            "failure_type",
            "prediction",
            "ground_truth",
            "confidence",
            "margin",
            "candidate_count",
        ]
        if column in failures.columns
    ]
    display = failures[columns].head(12).copy()
    for column in ["confidence", "margin"]:
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.3f}"
            )
    if "candidate_count" in display.columns:
        display["candidate_count"] = display["candidate_count"].fillna("").map(
            lambda value: "" if value == "" else str(int(value))
        )

    fig, ax = plt.subplots(figsize=(15, max(3.5, 0.45 * len(display) + 1.4)))
    ax.axis("off")
    table_artist = ax.table(
        cellText=display.values,
        colLabels=[column.replace("_", " ").title() for column in display.columns],
        loc="center",
        cellLoc="left",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1, 1.35)

    for (row, _column), cell in table_artist.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#333333")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f5f7")

    ax.set_title("Highest-Priority Failure Examples", weight="bold", pad=12)
    fig.tight_layout()

    path = output_dir / "failure_examples_table.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _threshold_curve(
    annotated_rows: pd.DataFrame, score_column: str, thresholds: list[float]
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        subset = annotated_rows.loc[annotated_rows[score_column] >= threshold]
        total = len(subset)
        errors = int(subset["failure_type"].isin(FAILURE_TYPES).sum())
        model_errors = int((subset["failure_type"] == "model_mistake").sum())
        missing_errors = int((subset["failure_type"] == "missing_candidate").sum())
        rows.append(
            {
                "threshold": threshold,
                "kept_rows": total,
                "error_rate": _safe_rate(errors, total),
                "model_error_rate": _safe_rate(model_errors, total),
                "missing_candidate_rate": _safe_rate(missing_errors, total),
            }
        )
    return pd.DataFrame(rows)


def _plot_threshold_curve(
    curve: pd.DataFrame,
    *,
    x_column: str,
    title: str,
    xlabel: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        curve[x_column],
        curve["error_rate"],
        marker="o",
        linewidth=2.5,
        label="All errors",
        color="#222222",
    )
    ax.plot(
        curve[x_column],
        curve["model_error_rate"],
        marker="o",
        linewidth=2,
        label="Model mistakes",
        color="#4c78a8",
    )
    ax.plot(
        curve[x_column],
        curve["missing_candidate_rate"],
        marker="o",
        linewidth=2,
        label="Missing candidates",
        color="#f58518",
    )
    ax2 = ax.twinx()
    ax2.plot(
        curve[x_column],
        curve["kept_rows"],
        linestyle="--",
        linewidth=1.8,
        color="#777777",
        label="Rows kept",
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Error rate among kept rows")
    ax2.set_ylabel("Rows kept")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", alpha=0.25)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _prepare_annotated_rows(annotated_rows: pd.DataFrame) -> pd.DataFrame:
    prepared = annotated_rows.copy()
    for column in ["confidence", "margin", "candidate_count"]:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def _thresholds(values: pd.Series) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return [0.0]
    quantiles = clean.quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).tolist()
    return sorted({round(float(value), 6) for value in quantiles})


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _save_missing_metric_plot(path: Path, metric_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        f"{metric_name} is not available in this report",
        ha="center",
        va="center",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _validate_report_dir(report_dir: Path) -> None:
    missing = [name for name in REQUIRED_REPORT_FILES if not (report_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required report files in {report_dir}: {missing}. "
            "Run candidate_failure_analysis.py with --output-dir first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diagnostic visuals from candidate failure-analysis outputs."
    )
    parser.add_argument(
        "report_dir",
        type=Path,
        help="Directory containing field_breakdown.csv and annotated_rows.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for chart PNGs. Defaults to report_dir.",
    )
    args = parser.parse_args()

    paths = plot_failure_analysis(args.report_dir, output_dir=args.output_dir)
    for path in paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
