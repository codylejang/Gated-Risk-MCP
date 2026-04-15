from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from datasets import DatasetDict, load_from_disk
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_KEYS = {"text", "word", "words", "value", "label", "menu", "item", "name", "nm", "cnt", "price"}
DATASET_COLORS = {"CORD": "#2A6F97", "SROIE": "#C96C3A"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small set of EDA plots for receipt datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("Outputs") / "eda")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def read_json(path: Path) -> Any | None:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception:
            continue
    return None


def read_image_size(path: Path) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None, None


def extract_image_size(image: Any) -> tuple[int, int] | None:
    size = getattr(image, "size", None)
    if not isinstance(size, tuple) or len(size) != 2:
        return None
    width, height = size
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    return None


def collect_strings(obj: Any, parent_key: str = "") -> list[tuple[str, str]]:
    if isinstance(obj, dict):
        out: list[tuple[str, str]] = []
        for key, value in obj.items():
            out.extend(collect_strings(value, key.lower()))
        return out
    if isinstance(obj, list):
        out: list[tuple[str, str]] = []
        for item in obj:
            out.extend(collect_strings(item, parent_key))
        return out
    if isinstance(obj, str):
        text = normalize_text(obj)
        return [(parent_key, text)] if text else []
    return []


def text_count_from_payload(payload: Any) -> int:
    return sum(1 for key, _ in collect_strings(payload) if not key or key in TEXT_KEYS)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def clean_counts(values: list[int]) -> list[int]:
    cleaned = sorted(value for value in values if isinstance(value, int) and value > 0)
    if len(cleaned) < 5:
        return cleaned
    upper = percentile([float(value) for value in cleaned], 0.99)
    return [value for value in cleaned if value <= upper]


def clean_image_sizes(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = [(width, height) for width, height in values if width > 0 and height > 0]
    if len(cleaned) < 5:
        return cleaned

    areas = sorted(float(width * height) for width, height in cleaned)
    upper_area = percentile(areas, 0.99)
    return [(width, height) for width, height in cleaned if (width * height) <= upper_area]


def quantiles(values: list[float], count: int = 100) -> list[float]:
    sorted_values = sorted(values)
    if not sorted_values:
        return []
    if len(sorted_values) == 1:
        return [sorted_values[0]] * count
    return [percentile(sorted_values, index / (count - 1)) for index in range(count)]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def binned_median_trend(
    x_values: list[float],
    y_values: list[float],
    bins: int = 8,
) -> tuple[list[float], list[float]]:
    paired = sorted((x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0)
    if len(paired) < bins:
        return [], []

    chunk_size = max(1, len(paired) // bins)
    trend_x: list[float] = []
    trend_y: list[float] = []

    for start in range(0, len(paired), chunk_size):
        chunk = paired[start:start + chunk_size]
        if len(chunk) < 3:
            continue
        chunk_x = [x for x, _ in chunk]
        chunk_y = [y for _, y in chunk]
        trend_x.append(median(chunk_x))
        trend_y.append(median(chunk_y))

    return trend_x, trend_y


def ranked_median_trend(
    x_values: list[float],
    y_values: list[float],
    bins: int = 8,
) -> tuple[list[float], list[float]]:
    paired = sorted((x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0)
    if len(paired) < bins:
        return [], []

    chunk_size = max(1, len(paired) // bins)
    trend_x: list[float] = []
    trend_y: list[float] = []

    for bucket_index, start in enumerate(range(0, len(paired), chunk_size), start=1):
        chunk = paired[start:start + chunk_size]
        if len(chunk) < 3:
            continue
        chunk_y = [y for _, y in chunk]
        trend_x.append(float(bucket_index))
        trend_y.append(median(chunk_y))

    return trend_x, trend_y


def ranked_quantile_band(
    x_values: list[float],
    y_values: list[float],
    bins: int = 8,
) -> tuple[list[float], list[float], list[float], list[float]]:
    paired = sorted((x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0)
    if len(paired) < bins:
        return [], [], [], []

    chunk_size = max(1, len(paired) // bins)
    band_x: list[float] = []
    q25_values: list[float] = []
    q50_values: list[float] = []
    q75_values: list[float] = []

    for bucket_index, start in enumerate(range(0, len(paired), chunk_size), start=1):
        chunk = paired[start:start + chunk_size]
        if len(chunk) < 3:
            continue
        chunk_y = sorted(y for _, y in chunk)
        band_x.append(float(bucket_index))
        q25_values.append(percentile(chunk_y, 0.25))
        q50_values.append(percentile(chunk_y, 0.50))
        q75_values.append(percentile(chunk_y, 0.75))

    return band_x, q25_values, q50_values, q75_values


def clean_joint_records(records: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    cleaned = [(count, width, height) for count, width, height in records if count > 0 and width > 0 and height > 0]
    if len(cleaned) < 5:
        return cleaned

    count_values = sorted(float(count) for count, _, _ in cleaned)
    area_values = sorted(float(width * height) for _, width, height in cleaned)
    count_upper = percentile(count_values, 0.99)
    area_upper = percentile(area_values, 0.99)
    return [
        (count, width, height)
        for count, width, height in cleaned
        if count <= count_upper and (width * height) <= area_upper
    ]


def load_cord(data_dir: Path) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    cord_root = data_dir / "CORD"
    if not cord_root.exists():
        return [], [], []

    text_counts: list[int] = []
    image_sizes: list[tuple[int, int]] = []
    joint_records: list[tuple[int, int, int]] = []

    if (cord_root / "dataset_dict.json").exists():
        loaded_dataset = load_from_disk(str(cord_root))
        split_datasets = (
            loaded_dataset.values()
            if isinstance(loaded_dataset, DatasetDict)
            else [cast(Any, loaded_dataset)]
        )

        for split_dataset in split_datasets:
            for row in split_dataset:
                row_data = cast(dict[str, Any], row)
                payload = json.loads(str(row_data.get("ground_truth", "{}")))
                text_count = text_count_from_payload(payload)
                text_counts.append(text_count)
                image_size = extract_image_size(row_data.get("image"))
                if image_size:
                    image_sizes.append(image_size)
                    joint_records.append((text_count, image_size[0], image_size[1]))
        return clean_counts(text_counts), clean_image_sizes(image_sizes), clean_joint_records(joint_records)

    for json_path in cord_root.rglob("*.json"):
        payload = read_json(json_path)
        if payload is None:
            continue
        text_count = text_count_from_payload(payload)
        text_counts.append(text_count)

        for image_path in json_path.parent.glob(f"{json_path.stem}.*"):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            width, height = read_image_size(image_path)
            if width and height:
                image_sizes.append((width, height))
                joint_records.append((text_count, width, height))
                break

    return clean_counts(text_counts), clean_image_sizes(image_sizes), clean_joint_records(joint_records)


def load_sroie(data_dir: Path) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int, int]]]:
    sroie_root = data_dir / "SROIE"
    text_counts: list[int] = []
    image_sizes: list[tuple[int, int]] = []
    joint_records: list[tuple[int, int, int]] = []

    if sroie_root.exists():
        for text_path in sroie_root.rglob("*.txt"):
            if text_path.parent == sroie_root:
                continue
            raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
            count = sum(1 for line in raw_text.splitlines() if normalize_text(line))
            if count > 0:
                text_counts.append(count)

        for image_path in sroie_root.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            width, height = read_image_size(image_path)
            if width and height:
                image_sizes.append((width, height))

        if text_counts and image_sizes:
            paired_limit = min(len(text_counts), len(image_sizes))
            for index in range(paired_limit):
                width, height = image_sizes[index]
                joint_records.append((text_counts[index], width, height))

    if text_counts or image_sizes:
        return clean_counts(text_counts), clean_image_sizes(image_sizes), clean_joint_records(joint_records)

    fallback_csv = Path("Outputs") / "eda" / "sroie_cleaned_eda.csv"
    if fallback_csv.exists():
        with fallback_csv.open(encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    line_count = int(float(row.get("num_ocr_lines", "") or 0))
                except Exception:
                    line_count = 0
                if line_count > 0:
                    text_counts.append(line_count)

                try:
                    width = int(float(row.get("img_width", "") or 0))
                    height = int(float(row.get("img_height", "") or 0))
                except Exception:
                    width, height = 0, 0
                if width > 0 and height > 0:
                    image_sizes.append((width, height))
                    if line_count > 0:
                        joint_records.append((line_count, width, height))

    return clean_counts(text_counts), clean_image_sizes(image_sizes), clean_joint_records(joint_records)


def plot_text_density(dataset_counts: dict[str, list[int]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for dataset, counts in dataset_counts.items():
        if not counts:
            continue
        color = DATASET_COLORS.get(dataset, None)
        ax.hist(
            counts,
            bins=28,
            alpha=0.35,
            label=f"{dataset} count",
            color=color,
            density=True,
            edgecolor="white",
            linewidth=0.6,
        )
        median_value = sorted(counts)[len(counts) // 2]
        ax.axvline(median_value, color=color, linestyle="--", linewidth=2, label=f"{dataset} median")

    if not ax.patches and not ax.lines:
        plt.close()
        return

    ax.set_title("Cleaned Text Density per Sample")
    ax.set_xlabel("Text entries / lines per sample")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.2, linestyle=":")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "text_density.png", dpi=200)
    plt.close()


def plot_image_dimensions(dataset_images: dict[str, list[tuple[int, int]]], output_dir: Path) -> None:
    if not any(dataset_images.values()):
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for dataset, image_sizes in dataset_images.items():
        if not image_sizes:
            continue
        widths = [width for width, _ in image_sizes]
        heights = [height for _, height in image_sizes]
        aspect_ratios = [width / height for width, height in image_sizes if height]
        color = DATASET_COLORS.get(dataset, None)

        axes[0].scatter(widths, heights, alpha=0.45, s=16, label=dataset, color=color)
        axes[1].hist(aspect_ratios, bins=28, alpha=0.45, color=color, label=dataset, edgecolor="white", linewidth=0.6)

    axes[0].set_title("Cleaned Image Width vs Height")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    axes[0].grid(alpha=0.2, linestyle=":")
    axes[0].legend()

    axes[1].set_title("Aspect Ratio Distribution")
    axes[1].set_xlabel("Width / Height")
    axes[1].set_ylabel("Images")
    axes[1].grid(alpha=0.2, linestyle=":")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "image_dimensions.png", dpi=200)
    plt.close()


def plot_quantile_comparison(
    dataset_counts: dict[str, list[int]],
    dataset_images: dict[str, list[tuple[int, int]]],
    dataset_joint: dict[str, list[tuple[int, int, int]]],
    output_dir: Path,
) -> None:
    cord_counts = [float(value) for value in dataset_counts.get("CORD", [])]
    sroie_counts = [float(value) for value in dataset_counts.get("SROIE", [])]
    if not (cord_counts and sroie_counts):
        return

    count_q = [index / 99 for index in range(100)]
    text_cord_q = quantiles(cord_counts)
    text_sroie_q = quantiles(sroie_counts)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"width_ratios": [1.15, 1, 1]}, constrained_layout=True)

    axes[0].plot(count_q, text_cord_q, color=DATASET_COLORS["CORD"], linewidth=2.2, label="CORD")
    axes[0].plot(count_q, text_sroie_q, color=DATASET_COLORS["SROIE"], linewidth=2.2, label="SROIE")
    axes[0].fill_between(count_q, text_cord_q, text_sroie_q, color="#7A7A7A", alpha=0.12)
    axes[0].set_title("Text Count Quantile Profiles")
    axes[0].set_xlabel("Quantile")
    axes[0].set_ylabel("Text entries / lines")
    axes[0].grid(alpha=0.2, linestyle=":")
    axes[0].legend()

    for dataset in ("CORD", "SROIE"):
        records = dataset_joint.get(dataset, [])
        if not records:
            continue
        aspect_values = [float(width) / float(height) for _, width, height in records if height > 0]
        area_values = [float(width * height) / 1_000_000.0 for _, width, height in records if width > 0 and height > 0]
        text_values = [float(count) for count, _, _ in records]
        color = DATASET_COLORS[dataset]

        aspect_x, aspect_q25, aspect_q50, aspect_q75 = ranked_quantile_band(aspect_values, text_values, bins=8)
        if aspect_x and aspect_q50:
            axes[1].fill_between(aspect_x, aspect_q25, aspect_q75, color=color, alpha=0.16)
            axes[1].plot(aspect_x, aspect_q50, color=color, linewidth=2.6, marker="o", markersize=4, label=dataset)
            axes[1].annotate(
                f"{dataset} {aspect_q50[-1]:.0f}",
                xy=(aspect_x[-1], aspect_q50[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )

        area_x, area_q25, area_q50, area_q75 = ranked_quantile_band(area_values, text_values, bins=8)
        if area_x and area_q50:
            axes[2].fill_between(area_x, area_q25, area_q75, color=color, alpha=0.16)
            axes[2].plot(area_x, area_q50, color=color, linewidth=2.6, marker="o", markersize=4, label=dataset)
            axes[2].annotate(
                f"{dataset} {area_q50[-1]:.0f}",
                xy=(area_x[-1], area_q50[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )

    axes[1].set_title("Aspect Ratio Rank vs Median Text Count")
    axes[1].set_xlabel("Aspect ratio bin (low to high)")
    axes[1].set_ylabel("Median text entries / lines")
    axes[1].set_xticks(range(1, 9))
    axes[1].grid(alpha=0.2, linestyle=":")
    axes[1].legend()
    axes[1].text(0.03, 0.04, "Band = middle 50% of receipts", transform=axes[1].transAxes, fontsize=8, color="#555555")

    axes[2].set_title("Image Size Rank vs Median Text Count")
    axes[2].set_xlabel("Image area bin (small to large)")
    axes[2].set_ylabel("Median text entries / lines")
    axes[2].set_xticks(range(1, 9))
    axes[2].grid(alpha=0.2, linestyle=":")
    axes[2].legend()
    axes[2].text(0.03, 0.04, "Band = middle 50% of receipts", transform=axes[2].transAxes, fontsize=8, color="#555555")

    fig.suptitle("Cross-Dataset Quantile Analysis", fontsize=13)
    plt.savefig(output_dir / "cross_dataset_quantile_analysis.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cord_text, cord_images, cord_joint = load_cord(args.data_dir)
    sroie_text, sroie_images, sroie_joint = load_sroie(args.data_dir)

    if not (cord_text or cord_images or sroie_text or sroie_images):
        raise FileNotFoundError("No dataset files were found. Expected folders like Data/CORD and/or Data/SROIE.")

    plot_text_density({"CORD": cord_text, "SROIE": sroie_text}, args.output_dir)
    plot_image_dimensions({"CORD": cord_images, "SROIE": sroie_images}, args.output_dir)
    plot_quantile_comparison(
        {"CORD": cord_text, "SROIE": sroie_text},
        {"CORD": cord_images, "SROIE": sroie_images},
        {"CORD": cord_joint, "SROIE": sroie_joint},
        args.output_dir,
    )
    print(f"EDA complete. Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
