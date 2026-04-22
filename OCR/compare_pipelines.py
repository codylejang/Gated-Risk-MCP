"""
Compare the trained risk gate's outputs on two parallel views of the same
held-out receipts:

  pre-parsed pipeline : SROIE-provided gold OCR tokens + boxes
  ocr pipeline        : tokens + boxes produced by an OCR model (easyocr)

Both pipelines reuse the same trained risk gate (models/risk_gate.pkl)
and the same field labels, so any difference in score / action is
attributable to the OCR layer.

Usage:
    python OCR/compare_pipelines.py --n 30 --output Outputs/ocr_vs_preparsed.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sroie.risk_gate import RiskGate
from src.sroie.sroie_features import sroie_feature_dataframe
from OCR.build_ocr_records import load_preparsed_holdout, rebuild_with_ocr


def _score_records(gate: RiskGate, records, label: str) -> pd.DataFrame:
    df = sroie_feature_dataframe(records)
    results = gate.score_dataframe(df)
    out = pd.DataFrame(results)
    out = out.rename(
        columns={"risk_score": f"risk_score_{label}", "action": f"action_{label}"}
    )
    return out


def compare(
    n: int = 30,
    offset: int = 0,
    model_path: Path | None = None,
    data_root: Path | None = None,
    min_confidence: float = 0.0,
    output: Path | None = None,
) -> pd.DataFrame:
    print(f"loading {n} held-out SROIE train records (offset={offset})")
    pre_records = load_preparsed_holdout(n=n, offset=offset, data_root=data_root)
    print(f"  selected {len(pre_records)} records")

    print("running OCR on the same images")
    ocr_records = rebuild_with_ocr(pre_records, min_confidence=min_confidence)

    if not ocr_records:
        raise RuntimeError("no records survived OCR; check image paths")

    pre_by_id = {r.doc_id: r for r in pre_records}
    ocr_ids = {r.doc_id for r in ocr_records}
    pre_records = [pre_by_id[i] for i in pre_by_id if i in ocr_ids]

    print("loading risk gate")
    gate = RiskGate(model_path=model_path)

    print("scoring pre-parsed pipeline")
    pre_df = _score_records(gate, pre_records, "preparsed")

    print("scoring ocr pipeline")
    ocr_df = _score_records(gate, ocr_records, "ocr")

    merged = pre_df.merge(ocr_df, on="doc_id", how="inner")
    merged["risk_delta"] = merged["risk_score_ocr"] - merged["risk_score_preparsed"]
    merged["action_changed"] = merged["action_ocr"] != merged["action_preparsed"]

    pre_token_counts = {r.doc_id: len(r.ocr_tokens) for r in pre_records}
    ocr_token_counts = {r.doc_id: len(r.ocr_tokens) for r in ocr_records}
    merged["n_tokens_preparsed"] = merged["doc_id"].map(pre_token_counts)
    merged["n_tokens_ocr"] = merged["doc_id"].map(ocr_token_counts)

    _print_summary(merged)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output, index=False)
        print(f"wrote per-document comparison to {output}")

    return merged


def _print_summary(df: pd.DataFrame) -> None:
    print("\n=== summary ===")
    print(f"docs compared: {len(df)}")
    print(
        f"mean |risk_delta|: {df['risk_delta'].abs().mean():.4f}  "
        f"(max {df['risk_delta'].abs().max():.4f})"
    )
    agree = (~df["action_changed"]).sum()
    print(f"action agreement: {agree}/{len(df)} ({agree/len(df):.1%})")

    print("\npre-parsed action distribution:")
    print(df["action_preparsed"].value_counts().to_string())
    print("\nocr action distribution:")
    print(df["action_ocr"].value_counts().to_string())

    print("\naction transitions (pre-parsed -> ocr):")
    transitions = (
        df.groupby(["action_preparsed", "action_ocr"]).size().rename("count").reset_index()
    )
    print(transitions.to_string(index=False))

    print("\ntop 5 docs by |risk_delta|:")
    cols = [
        "doc_id",
        "risk_score_preparsed",
        "risk_score_ocr",
        "risk_delta",
        "action_preparsed",
        "action_ocr",
        "n_tokens_preparsed",
        "n_tokens_ocr",
    ]
    print(df.reindex(df["risk_delta"].abs().sort_values(ascending=False).index)[cols].head(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR vs pre-parsed risk gate outputs")
    parser.add_argument("--n", type=int, default=30, help="held-out size")
    parser.add_argument("--offset", type=int, default=0, help="held-out start offset (sorted by doc_id)")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "Outputs" / "ocr_vs_preparsed.csv")
    args = parser.parse_args()

    compare(
        n=args.n,
        offset=args.offset,
        model_path=args.model_path,
        data_root=args.data_root,
        min_confidence=args.min_confidence,
        output=args.output,
    )


if __name__ == "__main__":
    main()
