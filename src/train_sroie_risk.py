from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.sroie_risk import (
    RISK_LABEL,
    ReceiptRiskLogisticModel,
    ReceiptRiskMLPModel,
    build_receipt_risk_features,
    choose_threshold,
    gate_decision_table,
    heuristic_risk_score,
    load_public_test_outputs,
    load_receipt_outputs,
    save_json,
    threshold_metrics,
    threshold_sweep_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a receipt-level risk model and verification gate for SROIE.")
    parser.add_argument(
        "--extractor-dir",
        type=Path,
        default=Path("Outputs") / "sroie_extraction" / "mlp_neural",
        help="Directory containing full_train/validation/public_test extraction outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Outputs") / "sroie_risk" / "mlp_extractor_mlp_gate",
    )
    parser.add_argument("--model-type", choices=["logistic", "mlp"], default="mlp")
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    return parser.parse_args()


def build_risk_model(args: argparse.Namespace):
    if args.model_type == "logistic":
        return ReceiptRiskLogisticModel(random_state=args.random_state)
    return ReceiptRiskMLPModel(
        random_state=args.random_state,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_receipt_df, train_field_df, validation_receipt_df, validation_field_df = load_receipt_outputs(args.extractor_dir)
    public_receipt_df, public_field_df = load_public_test_outputs(args.extractor_dir)

    train_features = build_receipt_risk_features(train_receipt_df, train_field_df)
    validation_features = build_receipt_risk_features(validation_receipt_df, validation_field_df)
    public_features = build_receipt_risk_features(public_receipt_df, public_field_df)

    risk_model = build_risk_model(args).fit(train_features, label_column=RISK_LABEL)

    train_learned_scores = risk_model.predict_proba(train_features)
    validation_learned_scores = risk_model.predict_proba(validation_features)
    public_learned_scores = risk_model.predict_proba(public_features)

    train_heuristic_scores = heuristic_risk_score(train_features)
    validation_heuristic_scores = heuristic_risk_score(validation_features)

    learned_threshold = choose_threshold(train_features[RISK_LABEL], train_learned_scores, target_recall=args.target_recall)
    heuristic_threshold = choose_threshold(train_features[RISK_LABEL], train_heuristic_scores, target_recall=args.target_recall)

    train_learned_metrics = threshold_metrics(train_features[RISK_LABEL], train_learned_scores, learned_threshold.threshold)
    validation_learned_metrics = threshold_metrics(validation_features[RISK_LABEL], validation_learned_scores, learned_threshold.threshold)
    train_heuristic_metrics = threshold_metrics(train_features[RISK_LABEL], train_heuristic_scores, heuristic_threshold.threshold)
    validation_heuristic_metrics = threshold_metrics(validation_features[RISK_LABEL], validation_heuristic_scores, heuristic_threshold.threshold)

    train_learned_table = gate_decision_table(train_features, train_learned_scores, learned_threshold.threshold, "learned_risk_score")
    validation_learned_table = gate_decision_table(validation_features, validation_learned_scores, learned_threshold.threshold, "learned_risk_score")
    public_learned_table = gate_decision_table(public_features, public_learned_scores, learned_threshold.threshold, "learned_risk_score")

    validation_heuristic_table = gate_decision_table(validation_features, validation_heuristic_scores, heuristic_threshold.threshold, "heuristic_risk_score")

    threshold_sweep_table(train_features[RISK_LABEL], train_learned_scores).to_csv(
        args.output_dir / "learned_gate_train_threshold_sweep.csv", index=False
    )
    threshold_sweep_table(validation_features[RISK_LABEL], validation_learned_scores).to_csv(
        args.output_dir / "learned_gate_validation_threshold_sweep.csv", index=False
    )
    threshold_sweep_table(train_features[RISK_LABEL], train_heuristic_scores).to_csv(
        args.output_dir / "heuristic_gate_train_threshold_sweep.csv", index=False
    )
    threshold_sweep_table(validation_features[RISK_LABEL], validation_heuristic_scores).to_csv(
        args.output_dir / "heuristic_gate_validation_threshold_sweep.csv", index=False
    )

    train_learned_table.to_csv(args.output_dir / "train_receipt_risk_predictions.csv", index=False)
    validation_learned_table.to_csv(args.output_dir / "validation_receipt_risk_predictions.csv", index=False)
    public_learned_table.to_csv(args.output_dir / "public_test_gate_decisions.csv", index=False)
    validation_heuristic_table.to_csv(args.output_dir / "validation_heuristic_gate_predictions.csv", index=False)
    risk_model.weight_table().to_csv(args.output_dir / "risk_model_weights.csv", index=False)
    if hasattr(risk_model, "training_history_frame"):
        history_frame = risk_model.training_history_frame()
        if not history_frame.empty:
            history_frame.to_csv(args.output_dir / "risk_training_history.csv", index=False)

    feature_columns = pd.DataFrame({"feature_name": risk_model.feature_names})
    feature_columns.to_csv(args.output_dir / "risk_feature_columns.csv", index=False)

    save_json(
        args.output_dir / "risk_summary.json",
        {
            "extractor_dir": str(args.extractor_dir),
            "model_type": args.model_type,
            "target_recall": args.target_recall,
            "learned_gate": {
                "selected_threshold": {
                    "threshold": learned_threshold.threshold,
                    "recall": learned_threshold.recall,
                    "precision": learned_threshold.precision,
                    "verify_rate": learned_threshold.verify_rate,
                },
                "train_metrics": train_learned_metrics,
                "validation_metrics": validation_learned_metrics,
            },
            "heuristic_gate": {
                "selected_threshold": {
                    "threshold": heuristic_threshold.threshold,
                    "recall": heuristic_threshold.recall,
                    "precision": heuristic_threshold.precision,
                    "verify_rate": heuristic_threshold.verify_rate,
                },
                "train_metrics": train_heuristic_metrics,
                "validation_metrics": validation_heuristic_metrics,
            },
        },
    )

    print("Saved risk outputs to:", args.output_dir)
    print("Risk model type:", args.model_type)
    print("Validation learned-gate recall:", validation_learned_metrics["recall"])
    print("Validation learned-gate precision:", validation_learned_metrics["precision"])
    print("Validation learned-gate verify rate:", validation_learned_metrics["verify_rate"])
    print("Validation heuristic recall:", validation_heuristic_metrics["recall"])
    print("Validation heuristic precision:", validation_heuristic_metrics["precision"])


if __name__ == "__main__":
    main()
