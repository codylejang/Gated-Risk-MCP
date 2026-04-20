import pandas as pd
from typing import Any
import re


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """add ratio and aggregate features based on eda insights."""
    df = df.copy()

    df["token_box_ratio"] = df["n_tokens"] / df["n_boxes"].clip(lower=1)
    df["amount_token_ratio"] = df["n_amount_like_tokens"] / df["n_tokens"].clip(lower=1)
    df["date_token_ratio"] = df["n_date_like_tokens"] / df["n_tokens"].clip(lower=1)

    df["anchors_present_count"] = (
        df["has_total_anchor"].astype(int)
        + df["has_date_anchor"].astype(int)
        + df["has_cash_anchor"].astype(int)
    )

    return df


def sroie_feature_dataframe(records: list[Any]) -> pd.DataFrame:
    rows = []

    amount_pattern = re.compile(r"^\d+[.,]\d{2}$|^\d{1,3}(?:,\d{3})*(?:\.\d{2})?$")
    date_patterns = [
        re.compile(r"^\d{2}/\d{2}/\d{4}$"),
        re.compile(r"^\d{2}-\d{2}-\d{2}$"),
        re.compile(r"^\d{2}-\d{2}-\d{4}$"),
        re.compile(r"^\d{4}/\d{2}/\d{2}$"),
    ]

    for record in records:
        joined_ocr = " ".join(str(tok).strip() for tok in record.ocr_tokens).lower()

        company = str(record.fields.get("company", "")).strip()
        date = str(record.fields.get("date", "")).strip()
        address = str(record.fields.get("address", "")).strip()
        total = str(record.fields.get("total", "")).strip()

        exact_total_matches = 0
        if total:
            exact_total_matches = sum(
                str(tok).strip().lower() == total.lower()
                for tok in record.ocr_tokens
            )


        n_amount_like_tokens = sum(
            amount_pattern.match(str(tok).strip()) is not None
            for tok in record.ocr_tokens
        )

        n_date_like_tokens = sum(
            any(pattern.match(str(tok).strip()) for pattern in date_patterns)
            for tok in record.ocr_tokens
        )

        has_total_anchor = any(
            anchor in joined_ocr
            for anchor in ["total", "amt", "amount", "grand total", "nett"]
        )

        has_date_anchor = any(
            anchor in joined_ocr
            for anchor in ["date", "dated"]
        )

        has_cash_anchor = any(
            anchor in joined_ocr
            for anchor in ["cash", "change"]
        )

        rows.append({
            "doc_id": record.doc_id,
            "dataset": record.dataset,
            "split": record.split,
            "n_tokens": len(record.ocr_tokens),
            "n_boxes": len(record.bboxes),
            "company_present": bool(company),
            "date_present": bool(date),
            "address_present": bool(address),
            "total_present": bool(total),
            "company_len": len(company),
            "date_len": len(date),
            "address_len": len(address),
            "total_len": len(total),
            "company_in_ocr": company.lower() in joined_ocr if company else False,
            "date_in_ocr": date.lower() in joined_ocr if date else False,
            "address_in_ocr": address.lower() in joined_ocr if address else False,
            "total_in_ocr": total.lower() in joined_ocr if total else False,
            "exact_total_matches": exact_total_matches,
            "n_amount_like_tokens": n_amount_like_tokens,
            "n_date_like_tokens": n_date_like_tokens,
            "has_total_anchor": has_total_anchor,
            "has_date_anchor": has_date_anchor,
            "has_cash_anchor": has_cash_anchor,
        })
    return pd.DataFrame(rows)

