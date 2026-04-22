"""
Build a held-out set of DocumentRecords whose ocr_tokens / bboxes come from
a real OCR model rather than the SROIE-provided gold OCR text files.

Field labels (company / date / address / total) are taken from the
existing SROIE label files so the risk gate can score them in the same
way as the pre-parsed pipeline. Only the OCR layer is swapped out, which
is what we want to compare.

The official SROIE test directories shipped with the repo are empty, so
we deterministically hold out a subset of the train images by sorted
doc_id. The same held-out list is produced for both pipelines, which
makes the comparison apples-to-apples.
"""
from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import DocumentRecord, load_sroie_split
from OCR.ocr_extractor import extract_tokens_and_boxes


def select_holdout(
    records: List[DocumentRecord],
    n: int,
    offset: int = 0,
) -> List[DocumentRecord]:
    """Deterministic holdout: sort by doc_id, take a slice of size `n`."""
    ordered = sorted(records, key=lambda r: r.doc_id)
    if n <= 0 or n > len(ordered):
        return ordered[offset:]
    return ordered[offset : offset + n]


def load_preparsed_holdout(
    n: int = 30,
    offset: int = 0,
    data_root: Path | None = None,
) -> List[DocumentRecord]:
    """Holdout records using the SROIE-provided gold OCR (pre-parsed)."""
    records = load_sroie_split("train", data_root=data_root)
    return select_holdout(records, n=n, offset=offset)


def rebuild_with_ocr(
    records: List[DocumentRecord],
    min_confidence: float = 0.0,
    verbose: bool = True,
) -> List[DocumentRecord]:
    """Return a parallel list of records where tokens/bboxes are OCR-extracted."""
    rebuilt: List[DocumentRecord] = []
    for i, rec in enumerate(records):
        if rec.image_path is None or not Path(rec.image_path).exists():
            if verbose:
                print(f"  [{i+1}/{len(records)}] {rec.doc_id}: no image, skipping")
            continue
        if verbose:
            print(f"  [{i+1}/{len(records)}] running OCR on {rec.doc_id}")
        tokens, bboxes = extract_tokens_and_boxes(
            Path(rec.image_path), min_confidence=min_confidence
        )
        new_rec = replace(
            rec,
            ocr_tokens=tokens,
            bboxes=bboxes,
            fields=copy.deepcopy(rec.fields),
            metadata={**rec.metadata, "ocr_source": "easyocr"},
        )
        rebuilt.append(new_rec)
    return rebuilt
