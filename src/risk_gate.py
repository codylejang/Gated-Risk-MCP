"""
risk gate: train and inference pipeline for extraction uncertainty scoring.
outputs a calibrated probability that an extraction is risky/uncertain.
can be used standalone or as an mcp tool.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import load_sroie_split, DocumentRecord
from src.sroie_features import (
    sroie_feature_dataframe,
    add_derived_features,
    sroie_proxy_label_dataframe,
)


# feature columns used for model input
FEATURE_COLS = [
    "n_tokens",
    "n_boxes",
    "ocr_char_count",
    "ocr_word_count",
    "company_len",
    "date_len",
    "address_len",
    "total_len",
    "n_amount_like_tokens",
    "n_date_like_tokens",
    "token_box_ratio",
    "amount_token_ratio",
    "date_token_ratio",
    "avg_token_len",
    "avg_words_per_token",
    "anchors_present_count",
    "fields_present_count",
    "aspect_ratio",
    "has_total_anchor",
    "has_date_anchor",
    "has_cash_anchor",
]


@dataclass
class RiskGateConfig:
    """config for risk gate training."""
    model_path: Path = Path("models/risk_gate.pkl")
    threshold_low: float = 0.6
    threshold_high: float = 0.95
    random_state: int = 60
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.1


def _ensure_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure derived features exist and required model columns are present."""
    df = df.copy()

    missing_feature_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_feature_cols:
        df = add_derived_features(df)

    missing_feature_cols = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_feature_cols:
        raise ValueError(f"missing required feature columns: {missing_feature_cols}")

    return df


def train_risk_gate(
    records: list[DocumentRecord],
    config: Optional[RiskGateConfig] = None,
) -> dict[str, Any]:
    """
    train risk gate model on sroie records.
    returns dict with model, metrics, and feature importance.
    """
    if config is None:
        config = RiskGateConfig()

    # build finalized feature dataframe
    df = sroie_feature_dataframe(records)
    df = _ensure_feature_dataframe(df)

    # build canonical proxy labels from the finalized feature table
    label_df = sroie_proxy_label_dataframe(df)
    df = df.merge(label_df, on="doc_id", how="left")

    # prepare features and labels
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["proxy_verify"].astype(int).values

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    # train base classifier
    base_clf = GradientBoostingClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
    )

    # calibrate for better probability estimates
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    clf.fit(X_train, y_train)

    # evaluate
    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= config.threshold_low).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    brier = brier_score_loss(y_val, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary"
    )

    # feature importance from base estimator
    importances = {}
    for cal_clf in clf.calibrated_classifiers_:
        base = cal_clf.estimator
        for i, col in enumerate(FEATURE_COLS):
            importances[col] = importances.get(col, 0) + base.feature_importances_[i]
    for col in importances:
        importances[col] /= len(clf.calibrated_classifiers_)

    metrics = {
        "auc": float(auc),
        "brier_score": float(brier),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "pos_rate_train": float(y_train.mean()),
        "pos_rate_val": float(y_val.mean()),
    }

    # save model (store config as dict for portability)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "feature_cols": FEATURE_COLS,
            "config": {
                "threshold_low": config.threshold_low,
                "threshold_high": config.threshold_high,
            },
            "metrics": metrics,
        }, f)

    return {
        "model": clf,
        "metrics": metrics,
        "feature_importance": importances,
        "model_path": str(config.model_path),
    }


class RiskGate:
    """
    risk gate inference class.
    loads trained model and scores new documents.
    """

    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = PROJECT_ROOT / "models" / "risk_gate.pkl"
        self.model_path = Path(model_path)
        self._model = None
        self._feature_cols = None
        self._config = None
        self._load()

    def _load(self) -> None:
        """load model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"model not found: {self.model_path}")
        with open(self.model_path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._feature_cols = data["feature_cols"]

        # handle both dict and dataclass config formats
        cfg = data.get("config", {})
        if isinstance(cfg, dict):
            self._config = RiskGateConfig(
                threshold_low=cfg.get("threshold_low", 0.3),
                threshold_high=cfg.get("threshold_high", 0.6),
            )
        else:
            self._config = cfg

    def score_record(self, record: DocumentRecord) -> dict[str, Any]:
        """
        score a single document record.
        returns risk score and recommended action.
        """
        df = sroie_feature_dataframe([record])
        return self.score_dataframe(df)[0]

    def score_dataframe(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        score a dataframe of documents.
        returns list of dicts with risk_score and action.
        """
        df = _ensure_feature_dataframe(df)
        X = df[self._feature_cols].values.astype(np.float32)
        probs = self._model.predict_proba(X)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            action = self._get_action(prob)
            results.append({
                "doc_id": df.iloc[i].get("doc_id", f"doc_{i}"),
                "risk_score": float(prob),
                "action": action,
            })
        return results

    def _get_action(self, prob: float) -> str:
        """map probability to recommended action."""
        if prob < self._config.threshold_low:
            return "auto_accept"
        elif prob < self._config.threshold_high:
            return "review"
        else:
            return "human_required"

    def score(self, ocr_tokens: list[str], bboxes: list[list[int]], fields: dict[str, Any]) -> dict[str, Any]:
        """MCP-compatible scoring interface; fields should be candidate extracted values."""
        record = DocumentRecord(
            doc_id="inference",
            dataset="inference",
            split="inference",
            ocr_tokens=ocr_tokens,
            bboxes=bboxes,
            fields=fields,
        )
        return self.score_record(record)


def run_training(data_root: Optional[Path] = None) -> dict[str, Any]:
    """run full training pipeline."""
    if data_root is None:
        data_root = PROJECT_ROOT / "Data"

    print("loading sroie train split")
    records = load_sroie_split("train", data_root=data_root)
    print(f"loaded {len(records)} records")

    print("training risk gate model")
    result = train_risk_gate(records)

    print(f"model saved to {result['model_path']}")
    print("metrics:")
    for k, v in result["metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("feature importance:")
    sorted_imp = sorted(result["feature_importance"].items(), key=lambda x: -x[1])
    for feat, imp in sorted_imp[:10]:
        print(f"  {feat}: {imp:.4f}")

    return result


def run_inference(
    model_path: Optional[Path] = None,
    data_root: Optional[Path] = None,
    split: str = "test",
) -> pd.DataFrame:
    """run inference on a dataset split."""
    if data_root is None:
        data_root = PROJECT_ROOT / "Data"

    print(f"loading sroie {split} split")
    try:
        records = load_sroie_split(split, data_root=data_root)
    except FileNotFoundError:
        print(f"split '{split}' not found, falling back to train")
        records = load_sroie_split("train", data_root=data_root)
    print(f"loaded {len(records)} records")

    n_with_fields = sum(bool(record.fields) for record in records)
    if n_with_fields == 0:
        raise ValueError(
            "run_inference requires records that already contain candidate extracted fields. "
            "Raw SROIE test receipts without fields cannot be meaningfully scored by the risk gate."
        )

    print("loading risk gate model")
    gate = RiskGate(model_path)

    print("scoring documents")
    df = sroie_feature_dataframe(records)
    results = gate.score_dataframe(df)

    results_df = pd.DataFrame(results)
    print(f"scored {len(results_df)} documents")
    print("action distribution:")
    print(results_df["action"].value_counts().to_string())

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="risk gate train/inference")
    parser.add_argument("--mode", choices=["train", "inference"], default="train")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        run_training(data_root=args.data_root)
    else:
        results_df = run_inference(
            model_path=args.model_path,
            data_root=args.data_root,
            split=args.split,
        )
        if args.output:
            results_df.to_csv(args.output, index=False)
            print(f"results saved to {args.output}")
