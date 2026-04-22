from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sroie.sroie_vlm import SROIENeuralVLM, SROIEVLMBaseline, split_records
from utils.data_utils import load_sroie_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the SROIE layout-aware extraction baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("Outputs") / "sroie_extraction")
    parser.add_argument("--model-type", choices=["logistic", "mlp"], default="mlp")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--max-span-lines", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_model(args: argparse.Namespace):
    if args.model_type == "logistic":
        return SROIEVLMBaseline(
            max_span_lines=args.max_span_lines,
            random_state=args.random_state,
        )
    return SROIENeuralVLM(
        max_span_lines=args.max_span_lines,
        random_state=args.random_state,
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
    )


def model_output_name(model_type: str) -> str:
    if model_type == "logistic":
        return "logistic_baseline"
    if model_type == "mlp":
        return "mlp_neural"
    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    args = parse_args()
    model_output_dir = args.output_dir / model_output_name(args.model_type)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_sroie_split("train", data_root=args.data_dir)
    labeled_train_records = [record for record in train_records if record.fields]
    eval_train_records, eval_holdout_records = split_records(
        labeled_train_records,
        eval_ratio=args.eval_ratio,
        random_state=args.random_state,
    )

    baseline = build_model(args).fit(eval_train_records)

    validation_field_frame, validation_receipt_frame = baseline.predict_records(eval_holdout_records)
    validation_summary = baseline.validation_summary(validation_field_frame)

    validation_field_frame.to_csv(model_output_dir / "validation_field_predictions.csv", index=False)
    validation_receipt_frame.to_csv(model_output_dir / "validation_receipt_summary.csv", index=False)
    baseline.model_weights().to_csv(model_output_dir / "validation_model_weights.csv", index=False)
    if hasattr(baseline, "training_history_frame"):
        history_frame = baseline.training_history_frame()
        if not history_frame.empty:
            history_frame.to_csv(model_output_dir / "validation_training_history.csv", index=False)
    save_json(
        model_output_dir / "validation_summary.json",
        {
            "model_type": args.model_type,
            "validation_summary": validation_summary,
            "training_stats": baseline.training_stats,
            "n_eval_train_records": len(eval_train_records),
            "n_eval_holdout_records": len(eval_holdout_records),
        },
    )

    final_model = build_model(args).fit(labeled_train_records)

    full_train_field_frame, full_train_receipt_frame = final_model.predict_records(labeled_train_records)
    public_test_records = load_sroie_split("test", data_root=args.data_dir)
    public_test_field_frame, public_test_receipt_frame = final_model.predict_records(public_test_records)

    full_train_field_frame.to_csv(model_output_dir / "full_train_field_predictions.csv", index=False)
    full_train_receipt_frame.to_csv(model_output_dir / "full_train_receipt_summary.csv", index=False)
    public_test_field_frame.to_csv(model_output_dir / "public_test_field_predictions.csv", index=False)
    public_test_receipt_frame.to_csv(model_output_dir / "public_test_receipt_summary.csv", index=False)
    final_model.model_weights().to_csv(model_output_dir / "final_model_weights.csv", index=False)
    if hasattr(final_model, "training_history_frame"):
        history_frame = final_model.training_history_frame()
        if not history_frame.empty:
            history_frame.to_csv(model_output_dir / "final_training_history.csv", index=False)
    save_json(
        model_output_dir / "run_metadata.json",
        {
            "model_type": args.model_type,
            "n_labeled_train_records": len(labeled_train_records),
            "n_public_test_records": len(public_test_records),
            "training_stats": final_model.training_stats,
            "max_span_lines": args.max_span_lines,
            "eval_ratio": args.eval_ratio,
            "random_state": args.random_state,
            "hidden_dims": args.hidden_dims,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
        },
    )

    print("Saved validation outputs to:", model_output_dir)
    print("Model type:", args.model_type)
    print("Validation field accuracy:", validation_summary.get("overall", {}).get("field_accuracy", 0.0))
    print("Validation recoverable field accuracy:", validation_summary.get("overall", {}).get("recoverable_field_accuracy", 0.0))
    print("Public test receipts scored:", len(public_test_records))


if __name__ == "__main__":
    main()
