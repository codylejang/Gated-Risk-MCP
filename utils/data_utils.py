from __future__ import annotations

import json
from datasets import load_from_disk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "Data"


@dataclass
class DocumentRecord:
    doc_id: str
    dataset: str
    split: str

    image_path: Optional[Path] = None
    image: Any = None
    image_size: Optional[Dict[str, int]] = None

    annotation_path: Optional[Path] = None
    ocr_path: Optional[Path] = None

    ocr_tokens: List[str] = field(default_factory=list)
    bboxes: List[List[int]] = field(default_factory=list)

    fields: Dict[str, Any] = field(default_factory=dict)

    raw_annotation: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _flatten_cord_fields(gt_parse: Dict[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}

    menu_items = gt_parse.get("menu", [])
    sub_total = gt_parse.get("sub_total", {})
    total = gt_parse.get("total", {})

    fields["menu"] = menu_items
    fields["menu_count"] = len(menu_items)

    if isinstance(sub_total, dict):
        for k, v in sub_total.items():
            fields[f"sub_total.{k}"] = v

    if isinstance(total, dict):
        for k, v in total.items():
            fields[f"total.{k}"] = v

    return fields


def _quad_to_bbox(quad: Dict[str, Any]) -> List[int]:
    xs = [quad.get("x1", 0), quad.get("x2", 0), quad.get("x3", 0), quad.get("x4", 0)]
    ys = [quad.get("y1", 0), quad.get("y2", 0), quad.get("y3", 0), quad.get("y4", 0)]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def _extract_tokens_and_boxes_from_valid_line(valid_line: List[Dict[str, Any]]) -> tuple[List[str], List[List[int]]]:
    tokens: List[str] = []
    bboxes: List[List[int]] = []

    for line in valid_line:
        words = line.get("words", [])
        for word in words:
            text = word.get("text", "")
            quad = word.get("quad", {})
            if text:
                tokens.append(text)
                bboxes.append(_quad_to_bbox(quad))

    return tokens, bboxes


def _parse_cord_example(example: Dict[str, Any], split_name: str) -> Dict[str, Any]:
    gt = json.loads(example["ground_truth"])
    gt_parse = gt.get("gt_parse", {})
    meta = gt.get("meta", {})
    valid_line = gt.get("valid_line", [])

    ocr_tokens, bboxes = _extract_tokens_and_boxes_from_valid_line(valid_line)
    fields = _flatten_cord_fields(gt_parse)

    image_size = meta.get("image_size", {})
    split = meta.get("split", split_name)
    image_id = meta.get("image_id", "")

    return {
        "image": example["image"],
        "gt_parse": gt_parse,
        "meta": meta,
        "valid_line": valid_line,
        "ocr_tokens": ocr_tokens,
        "bboxes": bboxes,
        "fields": fields,
        "image_size": image_size,
        "split": split,
        "doc_id": f"cord_{split}_{image_id}",
    }


def get_dataset_root(dataset_name: str, data_root: Optional[Path] = None) -> Path:
    root = Path(data_root) if data_root is not None else DATA_ROOT
    return root / dataset_name


def _list_files(directory: Path, suffixes: Iterable[str]) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffix_set
    )


def _find_first_existing(base_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for relative in candidates:
        candidate = base_dir / relative
        if candidate.exists():
            return candidate
    return None


def load_cord_split(split: str, data_root: Optional[Path] = None) -> List[DocumentRecord]:
    if data_root is None:
        data_root = DATA_ROOT

    split_dir = Path(data_root) / "CORD" / split
    ds = load_from_disk(str(split_dir))

    records: List[DocumentRecord] = []

    for example in ds:
        parsed = _parse_cord_example(example, split_name=split)

        record = DocumentRecord(
            doc_id=parsed["doc_id"],
            dataset="CORD",
            split=parsed["split"],
            image=parsed["image"],
            image_size=parsed["image_size"],
            ocr_tokens=parsed["ocr_tokens"],
            bboxes=parsed["bboxes"],
            fields=parsed["fields"],
            raw_annotation={
                "gt_parse": parsed["gt_parse"],
                "valid_line": parsed["valid_line"],
            },
            metadata=parsed["meta"],
        )
        records.append(record)

    return records


def _parse_sroie_ocr_file(path: Path) -> tuple[List[str], List[List[int]]]:
    tokens: List[str] = []
    bboxes: List[List[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = [part.strip() for part in line.rstrip("\n").split(",")]
            if len(parts) < 9:
                continue
            coords = parts[:8]
            text = ",".join(parts[8:]).strip()
            if not text:
                continue
            try:
                numbers = [int(float(value)) for value in coords]
            except ValueError:
                continue
            xs = numbers[0::2]
            ys = numbers[1::2]
            tokens.append(text)
            bboxes.append([min(xs), min(ys), max(xs), max(ys)])

    return tokens, bboxes


def _parse_sroie_label_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return {str(k).strip().lower(): v for k, v in data.items()}
        return {}
    except Exception:
        return {}


def load_sroie_split(split: str, data_root: Optional[Path] = None) -> List[DocumentRecord]:
    dataset_root = get_dataset_root("SROIE", data_root=data_root)

    split_lower = split.lower()
    if split_lower == "train":
        image_dir = _find_first_existing(
            dataset_root,
            [
                "0325updated.task1train(626p)",
                "task1train",
                "train_images",
                "images/train",
            ],
        )
        ocr_dir = image_dir
        label_dir = _find_first_existing(
            dataset_root,
            [
                "0325updated.task2train(626p)",
                "task2train",
                "train_labels",
                "labels/train",
            ],
        )

    elif split_lower in {"test", "public_test"}:
        image_dir = _find_first_existing(
            dataset_root,
            [
                "task1&2_test(361p)",
                "test_images",
                "images/test",
            ],
        )
        ocr_dir = _find_first_existing(
            dataset_root,
            [
                "text.task1&2-test(361p)",
                "text.task1&2-test（361p)",
                "test_ocr",
                "ocr/test",
            ],
        )
        label_dir = _find_first_existing(
            dataset_root,
            [
                "task3-test(347p)",
                "task3-test 347p) -",
                "test_labels",
                "labels/test",
            ],
        )

    else:
        raise ValueError("SROIE split must be 'train' or 'test'")

    if image_dir is None:
        raise FileNotFoundError(f"Could not find SROIE image directory for split='{split}' under {dataset_root}")

    records: List[DocumentRecord] = []
    for image_path in _list_files(image_dir, [".jpg", ".jpeg", ".png"]):
        doc_id = image_path.stem
        ocr_path = ocr_dir / f"{doc_id}.txt" if ocr_dir is not None else None
        label_path = label_dir / f"{doc_id}.txt" if label_dir is not None else None

        tokens: List[str] = []
        bboxes: List[List[int]] = []
        if ocr_path is not None and ocr_path.exists():
            tokens, bboxes = _parse_sroie_ocr_file(ocr_path)

        fields = _parse_sroie_label_file(label_path) if label_path is not None else {}

        records.append(
            DocumentRecord(
                doc_id=doc_id,
                dataset="SROIE",
                split=split,
                image_path=image_path,
                annotation_path=label_path,
                ocr_path=ocr_path,
                fields=fields,
                ocr_tokens=tokens,
                bboxes=bboxes,
                raw_annotation={"fields": fields},
            )
        )

    return records


def preview_record(record: DocumentRecord) -> None:
    print("=" * 80)
    print(f"doc_id: {record.doc_id}")
    print(f"dataset: {record.dataset}")
    print(f"split: {record.split}")
    print(f"image_path: {record.image_path}")
    print(f"image object present: {record.image is not None}")
    print(f"image_size: {record.image_size}")
    print(f"num_tokens: {len(record.ocr_tokens)}")
    print(f"num_boxes: {len(record.bboxes)}")
    print(f"num_fields: {len(record.fields)}")
    print(f"field keys: {list(record.fields.keys())[:20]}")
    print(f"metadata keys: {list(record.metadata.keys())[:20]}")
    print("=" * 80)
