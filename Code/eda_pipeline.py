from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
from datasets import DatasetDict, load_from_disk
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_KEYS = {"text", "word", "words", "value", "label", "menu", "item", "name", "nm", "cnt", "price"}


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
    if isinstance(width, int) and isinstance(height, int):
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


def load_cord(data_dir: Path) -> tuple[list[int], list[tuple[int, int]]]:
    cord_root = data_dir / "CORD"
    if not cord_root.exists():
        return [], []

    text_counts: list[int] = []
    image_sizes: list[tuple[int, int]] = []

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
                text_counts.append(text_count_from_payload(payload))
                image_size = extract_image_size(row_data.get("image"))
                if image_size:
                    image_sizes.append(image_size)
        return text_counts, image_sizes

    for json_path in cord_root.rglob("*.json"):
        payload = read_json(json_path)
        if payload is None:
            continue
        text_counts.append(text_count_from_payload(payload))

        for image_path in json_path.parent.glob(f"{json_path.stem}.*"):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            width, height = read_image_size(image_path)
            if width and height:
                image_sizes.append((width, height))

    return text_counts, image_sizes


def load_sroie(data_dir: Path) -> tuple[list[int], list[tuple[int, int]]]:
    sroie_root = data_dir / "SROIE"
    if not sroie_root.exists():
        return [], []

    text_counts: list[int] = []
    image_sizes: list[tuple[int, int]] = []

    for text_path in sroie_root.rglob("*.txt"):
        if text_path.parent == sroie_root:
            continue
        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
        text_counts.append(sum(1 for line in raw_text.splitlines() if normalize_text(line)))

    for image_path in sroie_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        width, height = read_image_size(image_path)
        if width and height:
            image_sizes.append((width, height))

    return text_counts, image_sizes


def plot_text_density(dataset_counts: dict[str, list[int]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for dataset, counts in dataset_counts.items():
        if counts:
            ax.hist(counts, bins=30, alpha=0.6, label=dataset)
    if not ax.patches:
        plt.close()
        return
    ax.set_title("Text Density per Sample")
    ax.set_xlabel("Text entries / lines per sample")
    ax.set_ylabel("Samples")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "text_density.png", dpi=200)
    plt.close()


def plot_image_dimensions(image_sizes: list[tuple[int, int]], output_dir: Path) -> None:
    if not image_sizes:
        return

    widths = [width for width, _ in image_sizes]
    heights = [height for _, height in image_sizes]
    aspect_ratios = [width / height for width, height in image_sizes if height]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(widths, heights, alpha=0.5)
    axes[0].set_title("Image Width vs Height")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")

    axes[1].hist(aspect_ratios, bins=30, color="#4C72B0", edgecolor="black")
    axes[1].set_title("Aspect Ratio Distribution")
    axes[1].set_xlabel("Width / Height")
    axes[1].set_ylabel("Images")

    plt.tight_layout()
    plt.savefig(output_dir / "image_dimensions.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cord_text, cord_images = load_cord(args.data_dir)
    sroie_text, sroie_images = load_sroie(args.data_dir)

    if not (cord_text or cord_images or sroie_text or sroie_images):
        raise FileNotFoundError("No dataset files were found. Expected folders like Data/CORD and/or Data/SROIE.")

    plot_text_density({"CORD": cord_text, "SROIE": sroie_text}, args.output_dir)
    plot_image_dimensions(cord_images + sroie_images, args.output_dir)
    print(f"EDA complete. Outputs saved to: {args.output_dir.resolve()}")


main()
