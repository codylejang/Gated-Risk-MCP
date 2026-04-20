from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


TARGET_FIELDS = ("company", "date", "address", "total")

AMOUNT_PATTERN = re.compile(r"(?<!\d)(?:RM\s*)?\d+(?:[.,]\d+)+(?!\d)")
DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{1,2}-\d{1,2}-\d{2,4}\b"),
    re.compile(r"\b\d{4}/\d{1,2}/\d{1,2}\b"),
    re.compile(r"\b\d{1,2}\s+[A-Z]{3,9}\s+\d{2,4}\b", re.IGNORECASE),
]

TOTAL_HINTS = ("total", "amount", "amt", "grand total", "nett", "nett", "balance due")
DATE_HINTS = ("date", "dated", "time", "invoice date")
COMPANY_HINTS = (
    "sdn",
    "bhd",
    "enterprise",
    "trading",
    "store",
    "mart",
    "shop",
    "restaurant",
    "restoran",
    "cafe",
    "hotel",
    "supermarket",
)
ADDRESS_HINTS = (
    "jalan",
    "jln",
    "road",
    "rd",
    "street",
    "st",
    "taman",
    "no.",
    "no ",
    "lot",
    "avenue",
    "blok",
    "floor",
    "plaza",
)

DATE_FORMATS = (
    "%d/%m/%Y",
    "%d/%m/%y",
    "%d-%m-%Y",
    "%d-%m-%y",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %b %y",
    "%d %B %Y",
    "%d %B %y",
)

FIELD_THRESHOLDS = {
    "company": 0.78,
    "date": 0.99,
    "address": 0.55,
    "total": 0.99,
}

FEATURE_COLUMNS = [
    "n_lines",
    "char_len",
    "token_count",
    "digit_ratio",
    "alpha_ratio",
    "upper_ratio",
    "contains_amount_pattern",
    "amount_match_count",
    "contains_date_pattern",
    "contains_total_hint",
    "contains_date_hint",
    "contains_company_hint",
    "contains_address_hint",
    "contains_currency_marker",
    "contains_address_number",
    "contains_entity_suffix",
    "contains_street_token",
    "x_left_norm",
    "x_center_norm",
    "x_right_norm",
    "y_top_norm",
    "y_center_norm",
    "y_bottom_norm",
    "width_norm",
    "height_norm",
    "area_norm",
    "rank_from_top_norm",
    "rank_from_bottom_norm",
    "nearest_total_hint_norm",
    "nearest_date_hint_norm",
    "nearest_company_hint_norm",
    "nearest_address_hint_norm",
    "source_is_line",
    "source_is_span",
    "source_is_regex",
    "amount_value_norm",
    "amount_is_doc_max",
]


@dataclass(frozen=True)
class OCRLine:
    text: str
    bbox: tuple[int, int, int, int]
    line_index: int
    image_width: int
    image_height: int

    @property
    def x0(self) -> int:
        return self.bbox[0]

    @property
    def y0(self) -> int:
        return self.bbox[1]

    @property
    def x1(self) -> int:
        return self.bbox[2]

    @property
    def y1(self) -> int:
        return self.bbox[3]


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    field_name: str
    text: str
    line_indices: tuple[int, ...]
    source_kind: str
    features: dict[str, float]


def normalize_text(text: str) -> str:
    lowered = str(text).lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def token_set(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token}


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def contains_any(text: str, phrases: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalize_text(phrase) in normalized for phrase in phrases)


def digit_ratio(text: str) -> float:
    digits = sum(character.isdigit() for character in text)
    return safe_ratio(digits, len(text))


def alpha_ratio(text: str) -> float:
    letters = sum(character.isalpha() for character in text)
    return safe_ratio(letters, len(text))


def upper_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    upper_count = sum(character.isupper() for character in letters)
    return safe_ratio(upper_count, len(letters))


def extract_amounts(text: str) -> list[str]:
    return [match.group(0).strip() for match in AMOUNT_PATTERN.finditer(text)]


def looks_like_total_amount(text: str) -> bool:
    cleaned = str(text).upper().replace("RM", "").strip()
    if not cleaned:
        return False
    if "." in cleaned or "," in cleaned:
        return True
    digits_only = re.sub(r"\D", "", cleaned)
    return len(digits_only) >= 4


def canonical_amount(text: str) -> str:
    cleaned = re.sub(r"[^0-9,\.]", "", str(text).upper().replace("RM", " ").strip())
    if not cleaned:
        return ""

    if "." in cleaned and "," in cleaned:
        last_dot = cleaned.rfind(".")
        last_comma = cleaned.rfind(",")
        separator_index = max(last_dot, last_comma)
        integer_part = re.sub(r"\D", "", cleaned[:separator_index]) or "0"
        fractional_part = re.sub(r"\D", "", cleaned[separator_index + 1 :])
        fractional_part = (fractional_part + "00")[:2]
        return f"{float(f'{integer_part}.{fractional_part}'):.2f}"

    if "." in cleaned:
        integer_part, fractional_part = cleaned.rsplit(".", maxsplit=1)
        integer_part = re.sub(r"\D", "", integer_part) or "0"
        fractional_part = re.sub(r"\D", "", fractional_part)
        if len(fractional_part) in {2, 3}:
            fractional_part = (fractional_part + "00")[:2]
            return f"{float(f'{integer_part}.{fractional_part}'):.2f}"
        digits = re.sub(r"\D", "", cleaned)
        return f"{float(digits):.2f}" if digits else ""

    if "," in cleaned:
        integer_part, fractional_part = cleaned.rsplit(",", maxsplit=1)
        integer_part = re.sub(r"\D", "", integer_part) or "0"
        fractional_part = re.sub(r"\D", "", fractional_part)
        if len(fractional_part) == 2:
            return f"{float(f'{integer_part}.{fractional_part}'):.2f}"
        digits = re.sub(r"\D", "", cleaned)
        return f"{float(digits):.2f}" if digits else ""

    digits = re.sub(r"\D", "", cleaned)
    return f"{float(digits):.2f}" if digits else ""


def parse_date_string(text: str) -> str:
    raw = str(text).strip().strip("()")
    if not raw:
        return ""
    for pattern in DATE_PATTERNS:
        match = pattern.search(raw)
        candidate = match.group(0) if match else raw
        for date_format in DATE_FORMATS:
            try:
                parsed = datetime.strptime(candidate, date_format)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
    normalized = normalize_text(raw)
    return normalized


def similarity_score(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    left_tokens = token_set(left_norm)
    right_tokens = token_set(right_norm)
    overlap = len(left_tokens & right_tokens)
    precision = safe_ratio(overlap, len(left_tokens))
    recall = safe_ratio(overlap, len(right_tokens))
    token_f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    sequence_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    containment = 1.0 if left_norm in right_norm or right_norm in left_norm else 0.0
    return float(max(token_f1, sequence_ratio, containment))


def candidate_match_score(field_name: str, candidate_text: str, gold_text: str) -> float:
    if not str(gold_text).strip():
        return 0.0
    if field_name == "total":
        return 1.0 if canonical_amount(candidate_text) == canonical_amount(gold_text) and canonical_amount(gold_text) else 0.0
    if field_name == "date":
        candidate_date = parse_date_string(candidate_text)
        gold_date = parse_date_string(gold_text)
        return 1.0 if candidate_date and candidate_date == gold_date else 0.0
    return similarity_score(candidate_text, gold_text)


@lru_cache(maxsize=4096)
def read_image_size(path_text: str) -> tuple[int, int]:
    with Image.open(path_text) as image:
        return image.size


def load_record_image_size(record: Any) -> tuple[int, int]:
    if record.image_size:
        width = int(record.image_size.get("width", 0))
        height = int(record.image_size.get("height", 0))
        if width > 0 and height > 0:
            return width, height
    if record.image is not None:
        width, height = record.image.size
        return int(width), int(height)
    if record.image_path is not None:
        return read_image_size(str(record.image_path.resolve()))
    return 1, 1


def build_lines(record: Any) -> list[OCRLine]:
    image_width, image_height = load_record_image_size(record)
    raw_lines: list[OCRLine] = []

    for index, (text, bbox) in enumerate(zip(record.ocr_tokens, record.bboxes)):
        cleaned_text = str(text).strip()
        if not cleaned_text or not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [int(value) for value in bbox]
        if x1 <= x0 or y1 <= y0:
            continue
        raw_lines.append(
            OCRLine(
                text=cleaned_text,
                bbox=(x0, y0, x1, y1),
                line_index=index,
                image_width=image_width,
                image_height=image_height,
            )
        )

    ordered_lines = sorted(raw_lines, key=lambda line: (line.y0, line.x0, line.y1, line.x1))
    return [
        OCRLine(
            text=line.text,
            bbox=line.bbox,
            line_index=ordered_index,
            image_width=line.image_width,
            image_height=line.image_height,
        )
        for ordered_index, line in enumerate(ordered_lines)
    ]


def nearest_anchor_distance(line_indices: Sequence[int], anchor_indices: Sequence[int], total_lines: int) -> float:
    if not anchor_indices:
        return 1.0
    best_distance = min(abs(candidate_index - anchor_index) for candidate_index in line_indices for anchor_index in anchor_indices)
    return safe_ratio(best_distance, max(total_lines - 1, 1))


def union_bbox(lines: Sequence[OCRLine]) -> tuple[int, int, int, int]:
    x0 = min(line.x0 for line in lines)
    y0 = min(line.y0 for line in lines)
    x1 = max(line.x1 for line in lines)
    y1 = max(line.y1 for line in lines)
    return x0, y0, x1, y1


def build_doc_context(lines: Sequence[OCRLine]) -> dict[str, Any]:
    normalized_texts = [normalize_text(line.text) for line in lines]
    amount_values = [canonical_amount(line.text) for line in lines]
    valid_amounts = [float(value) for value in amount_values if value]
    max_amount = max(valid_amounts) if valid_amounts else 0.0
    return {
        "total_hint_lines": [index for index, text in enumerate(normalized_texts) if contains_any(text, TOTAL_HINTS)],
        "date_hint_lines": [index for index, text in enumerate(normalized_texts) if contains_any(text, DATE_HINTS)],
        "company_hint_lines": [index for index, text in enumerate(normalized_texts) if contains_any(text, COMPANY_HINTS)],
        "address_hint_lines": [index for index, text in enumerate(normalized_texts) if contains_any(text, ADDRESS_HINTS)],
        "max_amount": max_amount,
        "n_lines": len(lines),
    }


def make_candidate(
    record: Any,
    field_name: str,
    text: str,
    line_indices: Sequence[int],
    source_kind: str,
    lines: Sequence[OCRLine],
    doc_context: dict[str, Any],
) -> Candidate:
    selected_lines = [lines[index] for index in line_indices]
    x0, y0, x1, y1 = union_bbox(selected_lines)
    image_width = max(selected_lines[0].image_width, 1)
    image_height = max(selected_lines[0].image_height, 1)
    normalized_candidate = normalize_text(text)
    amount_matches = extract_amounts(text)
    parsed_amounts = [float(canonical_amount(match)) for match in amount_matches if canonical_amount(match)]
    amount_value_float = max(parsed_amounts) if parsed_amounts else 0.0
    max_amount = doc_context["max_amount"]

    features = {
        "n_lines": float(len(selected_lines)),
        "char_len": float(len(text)),
        "token_count": float(len(normalized_candidate.split())),
        "digit_ratio": digit_ratio(text),
        "alpha_ratio": alpha_ratio(text),
        "upper_ratio": upper_ratio(text),
        "contains_amount_pattern": float(bool(amount_matches)),
        "amount_match_count": float(len(amount_matches)),
        "contains_date_pattern": float(any(pattern.search(text) for pattern in DATE_PATTERNS)),
        "contains_total_hint": float(contains_any(text, TOTAL_HINTS)),
        "contains_date_hint": float(contains_any(text, DATE_HINTS)),
        "contains_company_hint": float(contains_any(text, COMPANY_HINTS)),
        "contains_address_hint": float(contains_any(text, ADDRESS_HINTS)),
        "contains_currency_marker": float("rm" in normalized_candidate or "$" in text.lower()),
        "contains_address_number": float(bool(re.search(r"\bno\.?\s*\d", text.lower()))),
        "contains_entity_suffix": float(bool(re.search(r"\b(sdn|bhd|enterprise|trading|restaurant|restoran|mart|shop)\b", normalized_candidate))),
        "contains_street_token": float(bool(re.search(r"\b(jalan|jln|road|rd|street|st|taman|plaza|lot)\b", normalized_candidate))),
        "x_left_norm": safe_ratio(x0, image_width),
        "x_center_norm": safe_ratio((x0 + x1) / 2.0, image_width),
        "x_right_norm": safe_ratio(x1, image_width),
        "y_top_norm": safe_ratio(y0, image_height),
        "y_center_norm": safe_ratio((y0 + y1) / 2.0, image_height),
        "y_bottom_norm": safe_ratio(y1, image_height),
        "width_norm": safe_ratio(x1 - x0, image_width),
        "height_norm": safe_ratio(y1 - y0, image_height),
        "area_norm": safe_ratio((x1 - x0) * (y1 - y0), image_width * image_height),
        "rank_from_top_norm": safe_ratio(min(line_indices), max(doc_context["n_lines"] - 1, 1)),
        "rank_from_bottom_norm": safe_ratio(doc_context["n_lines"] - 1 - max(line_indices), max(doc_context["n_lines"] - 1, 1)),
        "nearest_total_hint_norm": nearest_anchor_distance(line_indices, doc_context["total_hint_lines"], doc_context["n_lines"]),
        "nearest_date_hint_norm": nearest_anchor_distance(line_indices, doc_context["date_hint_lines"], doc_context["n_lines"]),
        "nearest_company_hint_norm": nearest_anchor_distance(line_indices, doc_context["company_hint_lines"], doc_context["n_lines"]),
        "nearest_address_hint_norm": nearest_anchor_distance(line_indices, doc_context["address_hint_lines"], doc_context["n_lines"]),
        "source_is_line": float(source_kind == "line"),
        "source_is_span": float(source_kind == "span"),
        "source_is_regex": float(source_kind == "regex"),
        "amount_value_norm": safe_ratio(amount_value_float, max_amount) if max_amount > 0 else 0.0,
        "amount_is_doc_max": float(max_amount > 0 and math.isclose(amount_value_float, max_amount, rel_tol=0.0, abs_tol=0.01)),
    }
    return Candidate(
        doc_id=record.doc_id,
        field_name=field_name,
        text=text.strip(),
        line_indices=tuple(line_indices),
        source_kind=source_kind,
        features=features,
    )


def dedupe_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    unique: dict[tuple[str, tuple[int, ...], str], Candidate] = {}
    for candidate in candidates:
        key = (normalize_text(candidate.text), candidate.line_indices, candidate.source_kind)
        if not key[0]:
            continue
        unique.setdefault(key, candidate)
    return list(unique.values())


def build_company_candidates(record: Any, lines: Sequence[OCRLine], doc_context: dict[str, Any], max_span_lines: int) -> list[Candidate]:
    top_lines = [line for line in lines if line.line_index < min(8, len(lines)) or safe_ratio(line.y0, line.image_height) <= 0.35]
    candidates: list[Candidate] = []
    for line in top_lines:
        text = line.text.strip(" :-")
        if len(normalize_text(text)) < 3:
            continue
        candidates.append(make_candidate(record, "company", text, [line.line_index], "line", lines, doc_context))

    for start in range(min(len(top_lines), 6)):
        for span_length in range(2, max_span_lines + 1):
            end = start + span_length
            if end > len(top_lines):
                continue
            span_lines = top_lines[start:end]
            text = " ".join(line.text.strip() for line in span_lines).strip(" :-")
            if len(normalize_text(text)) < 5:
                continue
            candidates.append(
                make_candidate(
                    record,
                    "company",
                    text,
                    [line.line_index for line in span_lines],
                    "span",
                    lines,
                    doc_context,
                )
            )
    return dedupe_candidates(candidates)


def build_address_candidates(record: Any, lines: Sequence[OCRLine], doc_context: dict[str, Any], max_span_lines: int) -> list[Candidate]:
    candidate_lines = [line for line in lines if line.line_index < min(14, len(lines)) or safe_ratio(line.y0, line.image_height) <= 0.60]
    candidates: list[Candidate] = []

    for start in range(len(candidate_lines)):
        for span_length in range(1, max_span_lines + 1):
            end = start + span_length
            if end > len(candidate_lines):
                continue
            span_lines = candidate_lines[start:end]
            text = " ".join(line.text.strip() for line in span_lines).strip(" :-")
            if len(normalize_text(text)) < 8:
                continue
            candidates.append(
                make_candidate(
                    record,
                    "address",
                    text,
                    [line.line_index for line in span_lines],
                    "span" if span_length > 1 else "line",
                    lines,
                    doc_context,
                )
            )
    return dedupe_candidates(candidates)


def build_date_candidates(record: Any, lines: Sequence[OCRLine], doc_context: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for line in lines:
        matches = [match.group(0).strip() for pattern in DATE_PATTERNS for match in pattern.finditer(line.text)]
        for matched_text in matches:
            candidates.append(make_candidate(record, "date", matched_text, [line.line_index], "regex", lines, doc_context))
        if contains_any(line.text, DATE_HINTS):
            tail = re.split(r":", line.text, maxsplit=1)
            if len(tail) == 2 and tail[1].strip():
                candidates.append(make_candidate(record, "date", tail[1].strip(), [line.line_index], "line", lines, doc_context))
            candidates.append(make_candidate(record, "date", line.text.strip(), [line.line_index], "line", lines, doc_context))
    return dedupe_candidates(candidates)


def build_total_candidates(record: Any, lines: Sequence[OCRLine], doc_context: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for line in lines:
        line_has_total_hint = contains_any(line.text, TOTAL_HINTS)
        line_near_bottom = safe_ratio(line.y0, line.image_height) >= 0.55
        amount_matches = extract_amounts(line.text)
        for matched_text in amount_matches:
            if "." not in matched_text and not (line_has_total_hint or line_near_bottom):
                continue
            if not looks_like_total_amount(matched_text) and not (line_has_total_hint or line_near_bottom):
                continue
            candidates.append(make_candidate(record, "total", matched_text, [line.line_index], "regex", lines, doc_context))

        if line_has_total_hint:
            tail = re.split(r":", line.text, maxsplit=1)
            if len(tail) == 2 and tail[1].strip():
                for matched_text in extract_amounts(tail[1]) or [tail[1].strip()]:
                    candidates.append(make_candidate(record, "total", matched_text, [line.line_index], "line", lines, doc_context))
    return dedupe_candidates(candidates)


def build_field_candidates(record: Any, field_name: str, max_span_lines: int = 3) -> list[Candidate]:
    lines = build_lines(record)
    if not lines:
        return []
    doc_context = build_doc_context(lines)

    if field_name == "company":
        return build_company_candidates(record, lines, doc_context, max_span_lines=max_span_lines)
    if field_name == "address":
        return build_address_candidates(record, lines, doc_context, max_span_lines=max_span_lines)
    if field_name == "date":
        return build_date_candidates(record, lines, doc_context)
    if field_name == "total":
        return build_total_candidates(record, lines, doc_context)
    raise ValueError(f"Unsupported field: {field_name}")


def feature_frame(candidates: Sequence[Candidate]) -> pd.DataFrame:
    frame = pd.DataFrame([candidate.features for candidate in candidates], columns=FEATURE_COLUMNS)
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return frame.fillna(0.0).astype(float)


def split_records(records: Sequence[Any], eval_ratio: float = 0.2, random_state: int = 42) -> tuple[list[Any], list[Any]]:
    rng = np.random.default_rng(seed=random_state)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    eval_size = max(1, int(round(len(records) * eval_ratio)))
    eval_indices = set(indices[:eval_size].tolist())
    train_records = [record for index, record in enumerate(records) if index not in eval_indices]
    eval_records = [record for index, record in enumerate(records) if index in eval_indices]
    return train_records, eval_records


class CandidateFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class SROIEVLMBaseline:
    def __init__(self, max_span_lines: int = 3, random_state: int = 42) -> None:
        self.max_span_lines = max_span_lines
        self.random_state = random_state
        self.models: dict[str, Pipeline] = {}
        self.training_stats: dict[str, dict[str, float]] = {}

    def _build_training_rows(self, records: Sequence[Any], field_name: str) -> tuple[pd.DataFrame, pd.Series]:
        rows: list[dict[str, float]] = []
        labels: list[int] = []
        fitted_docs = 0
        skipped_docs = 0

        for record in records:
            gold_text = str(record.fields.get(field_name, "")).strip()
            if not gold_text:
                skipped_docs += 1
                continue
            candidates = build_field_candidates(record, field_name=field_name, max_span_lines=self.max_span_lines)
            if not candidates:
                skipped_docs += 1
                continue

            scores = [candidate_match_score(field_name, candidate.text, gold_text) for candidate in candidates]
            best_score = max(scores, default=0.0)
            if best_score < FIELD_THRESHOLDS[field_name]:
                skipped_docs += 1
                continue

            positive_indices = {index for index, score in enumerate(scores) if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-9)}
            for index, candidate in enumerate(candidates):
                row = dict(candidate.features)
                rows.append(row)
                labels.append(1 if index in positive_indices else 0)
            fitted_docs += 1

        self.training_stats[field_name] = {
            "fitted_docs": float(fitted_docs),
            "skipped_docs": float(skipped_docs),
            "candidate_rows": float(len(rows)),
            "positive_rows": float(sum(labels)),
        }

        return pd.DataFrame(rows, columns=FEATURE_COLUMNS).fillna(0.0), pd.Series(labels, dtype=int)

    def fit(self, records: Sequence[Any]) -> "SROIEVLMBaseline":
        for field_name in TARGET_FIELDS:
            X, y = self._build_training_rows(records, field_name)
            if X.empty or y.nunique() < 2:
                continue
            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            model.fit(X, y)
            self.models[field_name] = model
        return self

    def predict_field(self, record: Any, field_name: str) -> dict[str, Any]:
        gold_text = str(record.fields.get(field_name, "")).strip()
        candidates = build_field_candidates(record, field_name=field_name, max_span_lines=self.max_span_lines)
        if field_name not in self.models or not candidates:
            return {
                "doc_id": record.doc_id,
                "dataset": record.dataset,
                "split": record.split,
                "field_name": field_name,
                "gold_text": gold_text,
                "predicted_text": "",
                "confidence": 0.0,
                "margin": 0.0,
                "candidate_count": len(candidates),
                "predicted_source": "",
                "recoverable_by_candidates": np.nan if not gold_text else False,
                "match_score": 0.0,
                "correct": np.nan if not gold_text else False,
            }

        X = feature_frame(candidates)
        probabilities = self.models[field_name].predict_proba(X)[:, 1]
        ranked_indices = np.argsort(probabilities)[::-1]
        best_index = int(ranked_indices[0])
        best_candidate = candidates[best_index]
        best_probability = float(probabilities[best_index])
        second_probability = float(probabilities[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0
        margin = best_probability - second_probability
        recoverable_score = max((candidate_match_score(field_name, candidate.text, gold_text) for candidate in candidates), default=0.0)
        match_score = candidate_match_score(field_name, best_candidate.text, gold_text)

        return {
            "doc_id": record.doc_id,
            "dataset": record.dataset,
            "split": record.split,
            "field_name": field_name,
            "gold_text": gold_text,
            "predicted_text": best_candidate.text,
            "confidence": best_probability,
            "margin": margin,
            "candidate_count": len(candidates),
            "predicted_source": best_candidate.source_kind,
            "recoverable_by_candidates": recoverable_score >= FIELD_THRESHOLDS[field_name] if gold_text else np.nan,
            "match_score": match_score,
            "correct": match_score >= FIELD_THRESHOLDS[field_name] if gold_text else np.nan,
        }

    def predict_records(self, records: Sequence[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        field_rows: list[dict[str, Any]] = []
        for record in records:
            for field_name in TARGET_FIELDS:
                field_rows.append(self.predict_field(record, field_name))

        field_frame = pd.DataFrame(field_rows)
        if field_frame.empty:
            return field_frame, pd.DataFrame()

        receipt_frame = (
            field_frame.groupby(["doc_id", "dataset", "split"], as_index=False)
            .agg(
                min_confidence=("confidence", "min"),
                mean_confidence=("confidence", "mean"),
                min_margin=("margin", "min"),
                mean_margin=("margin", "mean"),
                n_fields=("field_name", "count"),
                n_correct=("correct", lambda values: float(values.dropna().sum()) if not values.dropna().empty else np.nan),
                n_recoverable=("recoverable_by_candidates", lambda values: float(values.dropna().sum()) if not values.dropna().empty else np.nan),
                any_error=("correct", lambda values: float((~values.dropna().astype(bool)).any()) if not values.dropna().empty else np.nan),
                any_unrecoverable=("recoverable_by_candidates", lambda values: float((~values.dropna().astype(bool)).any()) if not values.dropna().empty else np.nan),
            )
        )

        prediction_pivot = (
            field_frame.pivot_table(
                index=["doc_id", "dataset", "split"],
                columns="field_name",
                values="predicted_text",
                aggfunc="first",
            )
            .rename(
                columns={
                    "company": "company_prediction",
                    "date": "date_prediction",
                    "address": "address_prediction",
                    "total": "total_prediction",
                }
            )
            .reset_index()
        )
        receipt_frame = receipt_frame.merge(prediction_pivot, on=["doc_id", "dataset", "split"], how="left")

        return field_frame, receipt_frame

    def validation_summary(self, field_frame: pd.DataFrame) -> dict[str, Any]:
        summary: dict[str, Any] = {"overall": {}, "by_field": {}}
        if field_frame.empty:
            return summary

        labeled_mask = field_frame["gold_text"].fillna("").astype(str).str.len() > 0
        labeled_frame = field_frame.loc[labeled_mask].copy()
        if labeled_frame.empty:
            return summary

        summary["overall"] = {
            "field_accuracy": float(labeled_frame["correct"].mean()),
            "recoverable_field_accuracy": float(labeled_frame.loc[labeled_frame["recoverable_by_candidates"], "correct"].mean()) if labeled_frame["recoverable_by_candidates"].any() else 0.0,
            "mean_confidence": float(labeled_frame["confidence"].mean()),
            "mean_margin": float(labeled_frame["margin"].mean()),
        }

        for field_name in TARGET_FIELDS:
            field_subset = labeled_frame.loc[labeled_frame["field_name"] == field_name]
            if field_subset.empty:
                continue
            recoverable_subset = field_subset.loc[field_subset["recoverable_by_candidates"]]
            summary["by_field"][field_name] = {
                "field_accuracy": float(field_subset["correct"].mean()),
                "recoverable_accuracy": float(recoverable_subset["correct"].mean()) if not recoverable_subset.empty else 0.0,
                "recoverable_rate": float(field_subset["recoverable_by_candidates"].mean()),
                "mean_confidence": float(field_subset["confidence"].mean()),
                "mean_margin": float(field_subset["margin"].mean()),
            }

        return summary

    def model_weights(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for field_name, model in self.models.items():
            classifier: LogisticRegression = model.named_steps["classifier"]
            for feature_name, weight in zip(FEATURE_COLUMNS, classifier.coef_[0]):
                rows.append(
                    {
                        "field_name": field_name,
                        "feature_name": feature_name,
                        "weight": float(weight),
                        "abs_weight": float(abs(weight)),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["field_name", "feature_name", "weight", "abs_weight"])
        return pd.DataFrame(rows).sort_values(["field_name", "abs_weight"], ascending=[True, False]).reset_index(drop=True)


class SROIENeuralVLM:
    def __init__(
        self,
        max_span_lines: int = 3,
        random_state: int = 42,
        hidden_dims: Sequence[int] = (64, 32),
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 25,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
    ) -> None:
        self.max_span_lines = max_span_lines
        self.random_state = random_state
        self.hidden_dims = tuple(hidden_dims)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, FeatureMLP] = {}
        self.feature_scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.training_stats: dict[str, dict[str, float]] = {}
        self.training_history: dict[str, list[dict[str, float]]] = {}

    def _build_training_rows(self, records: Sequence[Any], field_name: str) -> tuple[pd.DataFrame, pd.Series]:
        rows: list[dict[str, float]] = []
        labels: list[int] = []
        fitted_docs = 0
        skipped_docs = 0

        for record in records:
            gold_text = str(record.fields.get(field_name, "")).strip()
            if not gold_text:
                skipped_docs += 1
                continue
            candidates = build_field_candidates(record, field_name=field_name, max_span_lines=self.max_span_lines)
            if not candidates:
                skipped_docs += 1
                continue

            scores = [candidate_match_score(field_name, candidate.text, gold_text) for candidate in candidates]
            best_score = max(scores, default=0.0)
            if best_score < FIELD_THRESHOLDS[field_name]:
                skipped_docs += 1
                continue

            positive_indices = {index for index, score in enumerate(scores) if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-9)}
            for index, candidate in enumerate(candidates):
                rows.append(dict(candidate.features))
                labels.append(1 if index in positive_indices else 0)
            fitted_docs += 1

        self.training_stats[field_name] = {
            "fitted_docs": float(fitted_docs),
            "skipped_docs": float(skipped_docs),
            "candidate_rows": float(len(rows)),
            "positive_rows": float(sum(labels)),
        }
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS).fillna(0.0), pd.Series(labels, dtype=int)

    def _fit_field_model(self, X_frame: pd.DataFrame, y_series: pd.Series, field_name: str) -> None:
        X = X_frame.to_numpy(dtype=np.float32)
        y = y_series.to_numpy(dtype=np.float32)

        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        X = (X - mean) / std

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)

        if len(indices) < 10:
            train_indices = indices
            val_indices = indices
        else:
            val_size = max(1, int(round(0.1 * len(indices))))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            if len(train_indices) == 0:
                train_indices = val_indices

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]

        train_dataset = CandidateFeatureDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True)

        model = FeatureMLP(
            input_dim=len(FEATURE_COLUMNS),
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        positive_count = float(y_train.sum())
        negative_count = float(len(y_train) - positive_count)
        pos_weight_value = negative_count / max(positive_count, 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_state = None
        best_val_loss = float("inf")
        history: list[dict[str, float]] = []

        X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)

        for epoch in range(1, self.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_examples = 0

            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                batch_size_actual = int(batch_features.shape[0])
                train_loss_sum += float(loss.item()) * batch_size_actual
                train_examples += batch_size_actual

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_loss = float(criterion(val_logits, y_val_tensor).item())

            mean_train_loss = train_loss_sum / max(train_examples, 1)
            history.append(
                {
                    "field_name": field_name,
                    "epoch": float(epoch),
                    "train_loss": mean_train_loss,
                    "val_loss": val_loss,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        self.models[field_name] = model
        self.feature_scalers[field_name] = (mean.astype(np.float32), std.astype(np.float32))
        self.training_history[field_name] = history
        self.training_stats[field_name]["best_val_loss"] = float(best_val_loss)

    def fit(self, records: Sequence[Any]) -> "SROIENeuralVLM":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        for field_name in TARGET_FIELDS:
            X, y = self._build_training_rows(records, field_name)
            if X.empty or y.nunique() < 2:
                continue
            self._fit_field_model(X, y, field_name)
        return self

    def _predict_candidate_probabilities(self, field_name: str, candidates: Sequence[Candidate]) -> np.ndarray:
        model = self.models[field_name]
        model.eval()
        features = feature_frame(candidates).to_numpy(dtype=np.float32)
        mean, std = self.feature_scalers[field_name]
        normalized = (features - mean) / std
        with torch.no_grad():
            tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            logits = model(tensor)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        return probabilities

    def predict_field(self, record: Any, field_name: str) -> dict[str, Any]:
        gold_text = str(record.fields.get(field_name, "")).strip()
        candidates = build_field_candidates(record, field_name=field_name, max_span_lines=self.max_span_lines)
        if field_name not in self.models or not candidates:
            return {
                "doc_id": record.doc_id,
                "dataset": record.dataset,
                "split": record.split,
                "field_name": field_name,
                "gold_text": gold_text,
                "predicted_text": "",
                "confidence": 0.0,
                "margin": 0.0,
                "candidate_count": len(candidates),
                "predicted_source": "",
                "recoverable_by_candidates": np.nan if not gold_text else False,
                "match_score": 0.0,
                "correct": np.nan if not gold_text else False,
            }

        probabilities = self._predict_candidate_probabilities(field_name, candidates)
        ranked_indices = np.argsort(probabilities)[::-1]
        best_index = int(ranked_indices[0])
        best_candidate = candidates[best_index]
        best_probability = float(probabilities[best_index])
        second_probability = float(probabilities[ranked_indices[1]]) if len(ranked_indices) > 1 else 0.0
        margin = best_probability - second_probability
        recoverable_score = max((candidate_match_score(field_name, candidate.text, gold_text) for candidate in candidates), default=0.0)
        match_score = candidate_match_score(field_name, best_candidate.text, gold_text)

        return {
            "doc_id": record.doc_id,
            "dataset": record.dataset,
            "split": record.split,
            "field_name": field_name,
            "gold_text": gold_text,
            "predicted_text": best_candidate.text,
            "confidence": best_probability,
            "margin": margin,
            "candidate_count": len(candidates),
            "predicted_source": best_candidate.source_kind,
            "recoverable_by_candidates": recoverable_score >= FIELD_THRESHOLDS[field_name] if gold_text else np.nan,
            "match_score": match_score,
            "correct": match_score >= FIELD_THRESHOLDS[field_name] if gold_text else np.nan,
        }

    def predict_records(self, records: Sequence[Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        field_rows: list[dict[str, Any]] = []
        for record in records:
            for field_name in TARGET_FIELDS:
                field_rows.append(self.predict_field(record, field_name))

        field_frame = pd.DataFrame(field_rows)
        if field_frame.empty:
            return field_frame, pd.DataFrame()

        receipt_frame = (
            field_frame.groupby(["doc_id", "dataset", "split"], as_index=False)
            .agg(
                min_confidence=("confidence", "min"),
                mean_confidence=("confidence", "mean"),
                min_margin=("margin", "min"),
                mean_margin=("margin", "mean"),
                n_fields=("field_name", "count"),
                n_correct=("correct", lambda values: float(values.dropna().sum()) if not values.dropna().empty else np.nan),
                n_recoverable=("recoverable_by_candidates", lambda values: float(values.dropna().sum()) if not values.dropna().empty else np.nan),
                any_error=("correct", lambda values: float((~values.dropna().astype(bool)).any()) if not values.dropna().empty else np.nan),
                any_unrecoverable=("recoverable_by_candidates", lambda values: float((~values.dropna().astype(bool)).any()) if not values.dropna().empty else np.nan),
            )
        )

        prediction_pivot = (
            field_frame.pivot_table(
                index=["doc_id", "dataset", "split"],
                columns="field_name",
                values="predicted_text",
                aggfunc="first",
            )
            .rename(
                columns={
                    "company": "company_prediction",
                    "date": "date_prediction",
                    "address": "address_prediction",
                    "total": "total_prediction",
                }
            )
            .reset_index()
        )
        receipt_frame = receipt_frame.merge(prediction_pivot, on=["doc_id", "dataset", "split"], how="left")
        return field_frame, receipt_frame

    def validation_summary(self, field_frame: pd.DataFrame) -> dict[str, Any]:
        summary: dict[str, Any] = {"overall": {}, "by_field": {}}
        if field_frame.empty:
            return summary

        labeled_mask = field_frame["gold_text"].fillna("").astype(str).str.len() > 0
        labeled_frame = field_frame.loc[labeled_mask].copy()
        if labeled_frame.empty:
            return summary

        summary["overall"] = {
            "field_accuracy": float(labeled_frame["correct"].mean()),
            "recoverable_field_accuracy": float(labeled_frame.loc[labeled_frame["recoverable_by_candidates"], "correct"].mean()) if labeled_frame["recoverable_by_candidates"].any() else 0.0,
            "mean_confidence": float(labeled_frame["confidence"].mean()),
            "mean_margin": float(labeled_frame["margin"].mean()),
        }

        for field_name in TARGET_FIELDS:
            field_subset = labeled_frame.loc[labeled_frame["field_name"] == field_name]
            if field_subset.empty:
                continue
            recoverable_subset = field_subset.loc[field_subset["recoverable_by_candidates"]]
            summary["by_field"][field_name] = {
                "field_accuracy": float(field_subset["correct"].mean()),
                "recoverable_accuracy": float(recoverable_subset["correct"].mean()) if not recoverable_subset.empty else 0.0,
                "recoverable_rate": float(field_subset["recoverable_by_candidates"].mean()),
                "mean_confidence": float(field_subset["confidence"].mean()),
                "mean_margin": float(field_subset["margin"].mean()),
            }

        return summary

    def model_weights(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for field_name, model in self.models.items():
            first_linear = next((layer for layer in model.network if isinstance(layer, nn.Linear)), None)
            if first_linear is None:
                continue
            weight_matrix = first_linear.weight.detach().cpu().numpy()
            feature_norms = np.linalg.norm(weight_matrix, axis=0)
            for feature_name, norm in zip(FEATURE_COLUMNS, feature_norms):
                rows.append(
                    {
                        "field_name": field_name,
                        "feature_name": feature_name,
                        "weight": float(norm),
                        "abs_weight": float(norm),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["field_name", "feature_name", "weight", "abs_weight"])
        return pd.DataFrame(rows).sort_values(["field_name", "abs_weight"], ascending=[True, False]).reset_index(drop=True)

    def training_history_frame(self) -> pd.DataFrame:
        rows: list[dict[str, float]] = []
        for history_rows in self.training_history.values():
            rows.extend(history_rows)
        if not rows:
            return pd.DataFrame(columns=["field_name", "epoch", "train_loss", "val_loss"])
        return pd.DataFrame(rows)
