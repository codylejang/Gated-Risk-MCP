"""
Lightweight OCR wrapper used to produce (tokens, bboxes) from receipt images.

Uses easyocr by default. The reader is created lazily and cached on the
module so repeated calls in the same process do not re-pay the model
load cost.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

_READER = None


def _get_reader(languages: List[str] | None = None, gpu: bool = False):
    global _READER
    if _READER is not None:
        return _READER
    import easyocr
    _READER = easyocr.Reader(languages or ["en"], gpu=gpu, verbose=False)
    return _READER


def extract_tokens_and_boxes(
    image_path: Path,
    languages: List[str] | None = None,
    gpu: bool = False,
    min_confidence: float = 0.0,
) -> Tuple[List[str], List[List[int]]]:
    """Run OCR on an image. Returns (tokens, bboxes) in [xmin,ymin,xmax,ymax] form."""
    reader = _get_reader(languages=languages, gpu=gpu)
    result = reader.readtext(str(image_path), detail=1, paragraph=False)

    tokens: List[str] = []
    bboxes: List[List[int]] = []
    for entry in result:
        # easyocr returns (quad, text, confidence)
        quad, text, conf = entry
        text = (text or "").strip()
        if not text or conf < min_confidence:
            continue
        xs = [int(p[0]) for p in quad]
        ys = [int(p[1]) for p in quad]
        tokens.append(text)
        bboxes.append([min(xs), min(ys), max(xs), max(ys)])

    return tokens, bboxes
