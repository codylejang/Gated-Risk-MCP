from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import load_cord_split
from src.cord.receipt_signals import (
    add_receipt_ratios,
    build_receipt_signal_frame,
    cord_review_label,
)


FEATURE_COLUMNS = [
    "n_tokens",
    "menu_count",
    "total_len",
    "subtotal_len",
    "tax_len",
    "service_len",
    "exact_total_matches",
    "exact_subtotal_matches",
    "exact_tax_matches",
    "n_amount_like_tokens",
    "has_total_anchor",
    "has_subtotal_anchor",
    "has_tax_anchor",
    "has_cash_anchor",
    "has_change_anchor",
    "amount_token_ratio",
    "anchor_count",
    "menu_token_ratio",
    "total_math_check_available",
    "total_math_gap_ratio_clipped",
    "log_total_math_gap_ratio",
]


@dataclass
class CordGateConfig:
    model_path: Path = Path("models/cord_review_gate.pkl")
    report_path: Path = Path("Outputs/cord_review_gate/metrics.json")
    threshold_low: float = 0.3
    threshold_high: float = 0.6
    random_state: int = 60
    test_size: float = 0.2


def _action(prob: float, low: float, high: float) -> str:
    if prob < low:
        return "auto_accept"
    if prob < high:
        return "review"
    return "human_required"


def make_cord_training_table(data_root: Path | None = None, split: str = "train") -> pd.DataFrame:
    records = load_cord_split(split, data_root=data_root)
    if not records:
        raise ValueError(f"No CORD records found for split={split!r}.")

    df = build_receipt_signal_frame(records)
    df = add_receipt_ratios(df)
    df["risk_label"] = df.apply(cord_review_label, axis=1)
    return df


def train_cord_review_gate(config: CordGateConfig, data_root: Path | None = None) -> dict[str, Any]:
    df = make_cord_training_table(data_root=data_root)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    target = df["risk_label"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    base = GradientBoostingClassifier(random_state=config.random_state)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary")

    importances: dict[str, float] = {}
    for calibrated_model in model.calibrated_classifiers_:
        base_model = calibrated_model.estimator
        for i, col in enumerate(FEATURE_COLUMNS):
            importances[col] = importances.get(col, 0.0) + float(base_model.feature_importances_[i])

    for col in importances:
        importances[col] /= len(model.calibrated_classifiers_)

    metrics = {
        "auc": float(roc_auc_score(y_val, probs)),
        "brier_score": float(brier_score_loss(y_val, probs)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_val": float(y_val.mean()),
    }

    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    with config.model_path.open("wb") as file:
        pickle.dump(
            {
                "model": model,
                "feature_columns": FEATURE_COLUMNS,
                "threshold_low": config.threshold_low,
                "threshold_high": config.threshold_high,
                "metrics": metrics,
                "feature_importance": importances,
            },
            file,
        )

    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "metrics": metrics,
        "feature_importance": importances,
        "model_path": str(config.model_path),
        "report_path": str(config.report_path),
    }


def run_inference(model_path: Path, data_root: Path | None = None, split: str = "validation") -> pd.DataFrame:
    with model_path.open("rb") as file:
        bundle = pickle.load(file)

    records = load_cord_split(split, data_root=data_root)
    df = build_receipt_signal_frame(records)
    df = add_receipt_ratios(df)

    probs = bundle["model"].predict_proba(df[bundle["feature_columns"]].values.astype(np.float32))[:, 1]
    return pd.DataFrame(
        {
            "doc_id": df["doc_id"],
            "risk_score": probs.astype(float),
            "action": [
                _action(float(prob), bundle["threshold_low"], bundle["threshold_high"])
                for prob in probs
            ],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the isolated CORD review gate.")
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path("models/cord_review_gate.pkl"))
    parser.add_argument("--report-path", type=Path, default=Path("Outputs/cord_review_gate/metrics.json"))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        result = train_cord_review_gate(
            CordGateConfig(model_path=args.model_path, report_path=args.report_path),
            data_root=args.data_root,
        )
        print(f"model saved to {result['model_path']}")
        print(f"metrics saved to {result['report_path']}")
        for key, value in result["metrics"].items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

        print("feature importance:")
        sorted_importance = sorted(
            result["feature_importance"].items(),
            key=lambda x: -x[1],
        )
        for feature, importance in sorted_importance[:10]:
            print(f"{feature}: {importance:.4f}")

        return

    result_df = run_inference(args.model_path, data_root=args.data_root, split=args.split)
    print(result_df["action"].value_counts().to_string())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(args.output, index=False)
        print(f"results saved to {args.output}")
    else:
        print(result_df.head(10).to_json(orient="records"))


if __name__ == "__main__":
    main()
