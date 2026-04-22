from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import load_sroie_split, DocumentRecord
from src.sroie.sroie_features import sroie_feature_dataframe, sroie_proxy_label_dataframe


def build_sroie_rule_table(records: list[DocumentRecord]) -> pd.DataFrame:
    feature_df = sroie_feature_dataframe(records)
    label_df = sroie_proxy_label_dataframe(feature_df)
    return feature_df.merge(label_df, on="doc_id", how="left")


def action_from_row(row: pd.Series) -> str:
    if bool(row["strict_high_risk"]):
        return "human_required"
    if bool(row["review_worthy"]):
        return "review"
    return "auto_accept"


class SROIERuleGate:
    """Rule-based SROIE verification gate."""

    def score_record(self, record: DocumentRecord) -> dict[str, Any]:
        df = build_sroie_rule_table([record])
        row = df.iloc[0]
        return {
            "doc_id": row["doc_id"],
            "strict_high_risk": bool(row["strict_high_risk"]),
            "review_worthy": bool(row["review_worthy"]),
            "action": action_from_row(row),
        }

    def score_dataframe(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        results = []
        for _, row in df.iterrows():
            results.append({
                "doc_id": row["doc_id"],
                "strict_high_risk": bool(row["strict_high_risk"]),
                "review_worthy": bool(row["review_worthy"]),
                "action": action_from_row(row),
            })
        return results

    def score(self, ocr_tokens: list[str], bboxes: list[list[int]], fields: dict[str, Any]) -> dict[str, Any]:
        record = DocumentRecord(
            doc_id="inference",
            dataset="inference",
            split="inference",
            ocr_tokens=ocr_tokens,
            bboxes=bboxes,
            fields=fields,
        )
        return self.score_record(record)


def run_rule_inference(
    data_root: Optional[Path] = None,
    split: str = "train",
) -> pd.DataFrame:
    if data_root is None:
        data_root = PROJECT_ROOT / "Data"

    records = load_sroie_split(split, data_root=data_root)
    rule_df = build_sroie_rule_table(records)
    rule_df["action"] = rule_df.apply(action_from_row, axis=1)

    return rule_df[
        [
            "doc_id",
            "strict_high_risk",
            "review_worthy",
            "action",
        ]
    ].copy()
