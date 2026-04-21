from pathlib import Path
from typing import Any
import re

import pandas as pd
from PIL import Image


AMOUNT_PATTERN = re.compile(
    r"^(?:rm)?\d+[.,]\d{2}$|^(?:rm)?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$",
    re.IGNORECASE,
)

DATE_PATTERNS = [
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),
    re.compile(r"^\d{2}-\d{2}-\d{2}$"),
    re.compile(r"^\d{2}-\d{2}-\d{4}$"),
    re.compile(r"^\d{4}/\d{2}/\d{2}$"),
    re.compile(r"^\d{8}$"),
]


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def _normalize_text(x: Any) -> str:
    s = _safe_str(x).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _normalize_for_match(x: Any) -> str:
    s = _normalize_text(x)
    return re.sub(r"[^a-z0-9]+", "", s)


def _safe_image_size(record: Any) -> tuple[float | None, float | None]:
    image_path = getattr(record, "image_path", None)
    if image_path is None:
        return None, None

    path = Path(image_path)
    if not path.exists():
        return None, None

    try:
        with Image.open(path) as img:
            width, height = img.size
        return float(width), float(height)
    except Exception:
        return None, None


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio, aggregate, and convenience features."""
    df = df.copy()

    df["token_box_ratio"] = df["n_tokens"] / df["n_boxes"].clip(lower=1)
    df["amount_token_ratio"] = df["n_amount_like_tokens"] / df["n_tokens"].clip(lower=1)
    df["date_token_ratio"] = df["n_date_like_tokens"] / df["n_tokens"].clip(lower=1)
    df["avg_token_len"] = df["ocr_char_count"] / df["n_tokens"].clip(lower=1)
    df["avg_words_per_token"] = df["ocr_word_count"] / df["n_tokens"].clip(lower=1)

    df["anchors_present_count"] = (
        df["has_total_anchor"].astype(int)
        + df["has_date_anchor"].astype(int)
        + df["has_cash_anchor"].astype(int)
    )

    df["fields_present_count"] = (
        df["company_present"].astype(int)
        + df["date_present"].astype(int)
        + df["address_present"].astype(int)
        + df["total_present"].astype(int)
    )

    df["all_fields_present"] = df["fields_present_count"] == 4
    df["any_field_missing"] = df["fields_present_count"] < 4
    df["ocr_is_empty"] = df["n_tokens"] == 0

    df["aspect_ratio"] = df["img_width"] / df["img_height"].clip(lower=1)

    return df


def sroie_feature_dataframe(records: list[Any]) -> pd.DataFrame:
    rows = []

    for record in records:
        ocr_tokens = list(getattr(record, "ocr_tokens", []))
        joined_ocr_raw = " ".join(_safe_str(tok) for tok in ocr_tokens)
        joined_ocr_norm = _normalize_for_match(joined_ocr_raw)

        company = _safe_str(record.fields.get("company", ""))
        date = _safe_str(record.fields.get("date", ""))
        address = _safe_str(record.fields.get("address", ""))
        total = _safe_str(record.fields.get("total", ""))

        company_norm = _normalize_for_match(company)
        date_norm = _normalize_for_match(date)
        address_norm = _normalize_for_match(address)
        total_norm = _normalize_for_match(total)

        token_norms = [_normalize_for_match(tok) for tok in ocr_tokens]
        stripped_tokens = [_safe_str(tok) for tok in ocr_tokens]

        exact_total_matches = 0
        if total_norm:
            exact_total_matches = sum(tok_norm == total_norm for tok_norm in token_norms)

        n_amount_like_tokens = sum(
            AMOUNT_PATTERN.match(tok) is not None
            for tok in stripped_tokens
        )

        n_date_like_tokens = sum(
            any(pattern.match(tok) for pattern in DATE_PATTERNS)
            for tok in stripped_tokens
        )

        joined_ocr_lower = joined_ocr_raw.lower()
        has_total_anchor = any(
            anchor in joined_ocr_lower
            for anchor in ["total", "amt", "amount", "grand total", "nett"]
        )
        has_date_anchor = any(
            anchor in joined_ocr_lower
            for anchor in ["date", "dated"]
        )
        has_cash_anchor = any(
            anchor in joined_ocr_lower
            for anchor in ["cash", "change"]
        )

        img_width, img_height = _safe_image_size(record)

        rows.append(
            {
                "doc_id": record.doc_id,
                "dataset": record.dataset,
                "split": record.split,
                "img_width": img_width,
                "img_height": img_height,
                "n_tokens": len(ocr_tokens),
                "n_boxes": len(getattr(record, "bboxes", [])),
                "ocr_char_count": sum(len(tok) for tok in stripped_tokens),
                "ocr_word_count": sum(len(tok.split()) for tok in stripped_tokens),
                "company_present": bool(company),
                "date_present": bool(date),
                "address_present": bool(address),
                "total_present": bool(total),
                "company_len": len(company),
                "date_len": len(date),
                "address_len": len(address),
                "total_len": len(total),
                "company_in_ocr": company_norm in joined_ocr_norm if company_norm else False,
                "date_in_ocr": date_norm in joined_ocr_norm if date_norm else False,
                "address_in_ocr": address_norm in joined_ocr_norm if address_norm else False,
                "total_in_ocr": total_norm in joined_ocr_norm if total_norm else False,
                "exact_total_matches": exact_total_matches,
                "n_amount_like_tokens": n_amount_like_tokens,
                "n_date_like_tokens": n_date_like_tokens,
                "has_total_anchor": has_total_anchor,
                "has_date_anchor": has_date_anchor,
                "has_cash_anchor": has_cash_anchor,
            }
        )

    df = pd.DataFrame(rows)
    return add_derived_features(df)


def sroie_proxy_label_dataframe(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Build more selective proxy verification labels from the feature table."""
    df = feature_df.copy()

    df["company_hard"] = (
        (~df["company_present"])
        | ((~df["company_in_ocr"]) & (~df["ocr_is_empty"]))
    )

    df["date_hard"] = (
        (~df["date_present"])
        | (~df["date_in_ocr"])
        | ((df["n_date_like_tokens"] > 1) & (~df["date_in_ocr"]))
    )

    df["address_hard"] = (
        (~df["address_present"])
        | ((~df["address_in_ocr"]) & (~df["ocr_is_empty"]))
    )

    df["total_hard"] = (
        (~df["total_present"])
        | (~df["total_in_ocr"])
        | ((df["exact_total_matches"] == 0) & (~df["ocr_is_empty"]))
        | (
            (df["n_amount_like_tokens"] >= 25)
            & (df["exact_total_matches"] >= 2)
            & (~df["has_total_anchor"])
        )
    )

    df["low_ocr_support"] = (
        df["ocr_is_empty"]
        | (df["n_tokens"] <= 20)
        | (df["ocr_word_count"] <= 35)
    )

    df["proxy_risk_score"] = (
        df["company_hard"].astype(int)
        + df["date_hard"].astype(int)
        + df["address_hard"].astype(int)
        + df["total_hard"].astype(int)
        + df["low_ocr_support"].astype(int)
    )

    df["proxy_verify"] = df["proxy_risk_score"] >= 2
    df["proxy_high_risk"] = df["proxy_risk_score"] >= 3

    keep_cols = [
        "doc_id",
        "company_hard",
        "date_hard",
        "address_hard",
        "total_hard",
        "low_ocr_support",
        "proxy_risk_score",
        "proxy_verify",
        "proxy_high_risk",
    ]
    return df[keep_cols].copy()
