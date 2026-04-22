from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


AMOUNT_PATTERN = re.compile(r"^\d+[.,]\d{2}$|^\d{1,3}(?:[,.]\d{3})*(?:[.,]\d{2})?$|^\d+$")


def parse_amount(value: Any) -> float | None:
    """Parse receipt-style amounts such as 1,591,600 or 144.69."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    cleaned = re.sub(r"[^0-9,.\-]", "", text)
    if not cleaned or cleaned in {"-", ".", ","}:
        return None

    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts[-1]) == 2 and len(parts) == 2:
            cleaned = ".".join(parts)
        else:
            cleaned = "".join(parts)

    try:
        return float(cleaned)
    except ValueError:
        return None


def _field_text(fields: dict[str, Any], key: str) -> str:
    value = fields.get(key, "")
    return "" if value is None else str(value).strip()


def _count_exact_matches(tokens: list[str], value: str) -> int:
    if not value:
        return 0
    needle = value.lower()
    compact_needle = re.sub(r"[^0-9a-z]+", "", needle)
    count = 0
    for token in tokens:
        text = str(token).strip().lower()
        compact_text = re.sub(r"[^0-9a-z]+", "", text)
        if text == needle or (compact_needle and compact_text == compact_needle):
            count += 1
    return count


def add_receipt_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio, transformed arithmetic, and proxy-rule helper columns."""
    df = df.copy()

    df["token_box_ratio"] = df["n_tokens"] / df["n_boxes"].clip(lower=1)
    df["amount_token_ratio"] = df["n_amount_like_tokens"] / df["n_tokens"].clip(lower=1)
    df["anchor_count"] = (
        df["has_total_anchor"].astype(int)
        + df["has_subtotal_anchor"].astype(int)
        + df["has_tax_anchor"].astype(int)
        + df["has_cash_anchor"].astype(int)
        + df["has_change_anchor"].astype(int)
    )
    df["menu_token_ratio"] = df["menu_count"] / df["n_tokens"].clip(lower=1)

    # stabilize the arithmetic inconsistency feature so extreme outliers do not dominate
    df["total_math_gap_ratio_clipped"] = df["total_math_gap_ratio"].clip(upper=10.0)
    df["log_total_math_gap_ratio"] = np.log1p(df["total_math_gap_ratio_clipped"])

    # revised proxy-rule helper columns from notebook validation
    df["extreme_token_count_v2"] = (df["n_tokens"] < 15) | (df["n_tokens"] > 100)
    df["too_many_total_matches_v2"] = df["exact_total_matches"] > 2
    df["missing_tax_anchor_v2"] = df["tax_present"] & (~df["has_tax_anchor"])
    df["missing_total_anchor_v2"] = df["total_present"] & (~df["has_total_anchor"])
    df["large_total_math_gap_v2"] = (
        df["total_math_check_available"] & (df["total_math_gap_ratio_clipped"] > 0.10)
    )

    return df


def build_receipt_signal_frame(records: list[Any]) -> pd.DataFrame:
    """Build CORD document-level features without touching the SROIE feature code."""
    rows: list[dict[str, Any]] = []

    for record in records:
        fields = record.fields
        tokens = [str(token).strip() for token in record.ocr_tokens]
        joined_ocr = " ".join(tokens).lower()

        total = _field_text(fields, "total.total_price")
        subtotal = _field_text(fields, "sub_total.subtotal_price")
        tax = _field_text(fields, "sub_total.tax_price")
        service = _field_text(fields, "sub_total.service_price")
        cash = _field_text(fields, "total.cashprice")
        change = _field_text(fields, "total.changeprice")

        menu = fields.get("menu", [])
        menu_count = fields.get("menu_count", len(menu) if isinstance(menu, list) else 0)

        parsed_total = parse_amount(total)
        parsed_subtotal = parse_amount(subtotal)
        parsed_tax = parse_amount(tax)
        parsed_service = parse_amount(service)

        expected_total = None
        total_math_gap = None
        total_math_gap_ratio = 0.0
        if parsed_total is not None and parsed_subtotal is not None:
            expected_total = parsed_subtotal
            if parsed_tax is not None:
                expected_total += parsed_tax
            if parsed_service is not None:
                expected_total += parsed_service
            total_math_gap = abs(parsed_total - expected_total)
            total_math_gap_ratio = total_math_gap / max(abs(parsed_total), 1.0)

        rows.append(
            {
                "doc_id": record.doc_id,
                "dataset": record.dataset,
                "split": record.split,
                "n_tokens": len(tokens),
                "n_boxes": len(record.bboxes),
                "menu_count": int(menu_count or 0),
                "total_present": bool(total),
                "subtotal_present": bool(subtotal),
                "tax_present": bool(tax),
                "service_present": bool(service),
                "cash_present": bool(cash),
                "change_present": bool(change),
                "total_len": len(total),
                "subtotal_len": len(subtotal),
                "tax_len": len(tax),
                "service_len": len(service),
                "total_in_ocr": total.lower() in joined_ocr if total else False,
                "subtotal_in_ocr": subtotal.lower() in joined_ocr if subtotal else False,
                "tax_in_ocr": tax.lower() in joined_ocr if tax else False,
                "service_in_ocr": service.lower() in joined_ocr if service else False,
                "cash_in_ocr": cash.lower() in joined_ocr if cash else False,
                "change_in_ocr": change.lower() in joined_ocr if change else False,
                "exact_total_matches": _count_exact_matches(tokens, total),
                "exact_subtotal_matches": _count_exact_matches(tokens, subtotal),
                "exact_tax_matches": _count_exact_matches(tokens, tax),
                "n_amount_like_tokens": sum(AMOUNT_PATTERN.match(token) is not None for token in tokens),
                "has_total_anchor": any(anchor in joined_ocr for anchor in ["total", "tot", "amount"]),
                "has_subtotal_anchor": any(anchor in joined_ocr for anchor in ["subtotal", "sub total", "sub-total"]),
                "has_tax_anchor": any(anchor in joined_ocr for anchor in ["tax", "ppn", "vat", "gst"]),
                "has_cash_anchor": "cash" in joined_ocr,
                "has_change_anchor": any(anchor in joined_ocr for anchor in ["change", "chg"]),
                "total_math_gap": 0.0 if total_math_gap is None else float(total_math_gap),
                "total_math_gap_ratio": float(total_math_gap_ratio),
                "total_math_check_available": total_math_gap is not None,
            }
        )

    return pd.DataFrame(rows)


def cord_review_label(row: pd.Series) -> int:
    """Revised CORD review target: 1 means this receipt is review-worthy."""
    risk_signals = 0

    if row.get("extreme_token_count_v2", False):
        risk_signals += 1
    if row.get("too_many_total_matches_v2", False):
        risk_signals += 1
    if row.get("missing_tax_anchor_v2", False):
        risk_signals += 1
    if row.get("missing_total_anchor_v2", False):
        risk_signals += 1
    if row.get("large_total_math_gap_v2", False):
        risk_signals += 1

    return 1 if risk_signals >= 2 else 0
