from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image



def _get_value(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def summarize_image_sizes(records: Iterable[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for record in records:
        width = None
        height = None

        image_size = _get_value(record, "image_size", {}) or {}
        if isinstance(image_size, dict):
            width = image_size.get("width")
            height = image_size.get("height")

        if width is None or height is None:
            image = _get_value(record, "image")
            if image is not None:
                try:
                    width, height = image.size
                except Exception:
                    width = height = None

        if width is None or height is None:
            image_path = _get_value(record, "image_path")
            if image_path:
                path = Path(image_path)
                if path.exists():
                    try:
                        with Image.open(path) as image:
                            width, height = image.size
                    except Exception:
                        width = height = None

        if width is None or height is None:
            continue

        rows.append(
            {
                "doc_id": _get_value(record, "doc_id"),
                "dataset": _get_value(record, "dataset"),
                "split": _get_value(record, "split"),
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height else None,
            }
        )

    return pd.DataFrame(rows)


def summarize_token_box_counts(records: Iterable[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        tokens = _get_value(record, "ocr_tokens") or []
        boxes = _get_value(record, "bboxes") or []
        rows.append(
            {
                "doc_id": _get_value(record, "doc_id"),
                "dataset": _get_value(record, "dataset"),
                "split": _get_value(record, "split"),
                "n_tokens": len(tokens),
                "n_boxes": len(boxes),
            }
        )
    return pd.DataFrame(rows)


def summarize_field_presence(records: Iterable[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        fields = _get_value(record, "fields") or {}
        row = {
            "doc_id": _get_value(record, "doc_id"),
            "dataset": _get_value(record, "dataset"),
            "split": _get_value(record, "split"),
            "n_fields": len(fields),
        }
        for key, value in fields.items():
            is_present = int(value not in (None, "", [], {}))
            row[f"has_{key}"] = is_present
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).fillna(0)


def field_frequency_table(records: Iterable[Any]) -> pd.DataFrame:
    counter: Counter[str] = Counter()
    n_docs = 0

    for record in records:
        n_docs += 1
        fields = _get_value(record, "fields") or {}
        for key, value in fields.items():
            if value not in (None, "", [], {}):
                counter[key] += 1

    rows = []
    for field_name, count in sorted(counter.items()):
        rows.append(
            {
                "field": field_name,
                "doc_count": count,
                "doc_fraction": count / n_docs if n_docs else None,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("doc_count", ascending=False).reset_index(drop=True)


def missingness_table(field_presence_df: pd.DataFrame) -> pd.DataFrame:
    if field_presence_df.empty:
        return pd.DataFrame()

    presence_cols = [col for col in field_presence_df.columns if col.startswith("has_")]
    rows = []
    n_docs = len(field_presence_df)

    for col in presence_cols:
        present = int(field_presence_df[col].sum())
        missing = n_docs - present
        rows.append(
            {
                "field": col.replace("has_", "", 1),
                "present_docs": present,
                "missing_docs": missing,
                "missing_fraction": missing / n_docs if n_docs else None,
            }
        )

    return pd.DataFrame(rows).sort_values("missing_fraction", ascending=False).reset_index(drop=True)


def print_basic_dataset_summary(records: Iterable[Any]) -> None:
    records = list(records)
    n_docs = len(records)
    datasets = Counter(_get_value(record, "dataset") for record in records)
    splits = Counter(_get_value(record, "split") for record in records)

    print(f"Documents: {n_docs}")
    print(f"Datasets: {dict(datasets)}")
    print(f"Splits: {dict(splits)}")

    token_df = summarize_token_box_counts(records)
    if not token_df.empty:
        print("\nToken/box summary:")
        print(token_df[["n_tokens", "n_boxes"]].describe())


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: str | None = None,
    figsize: tuple[int, int] = (7, 4),
) -> None:
    if df.empty or column not in df.columns:
        return

    plt.figure(figsize=figsize)
    df[column].dropna().hist(bins=bins)
    plt.title(title or column)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_bar_counts(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 4),
    rotate_x: bool = True,
) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        return

    plt.figure(figsize=figsize)
    plt.bar(df[x].astype(str), df[y])
    plt.title(title or f"{y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    if rotate_x:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
