from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


RISK_LABEL = "any_error"
UNRECOVERABLE_LABEL = "any_unrecoverable"
FIELD_NAMES = ("company", "date", "address", "total")
BASE_RISK_FEATURES = [
    "min_confidence",
    "mean_confidence",
    "min_margin",
    "mean_margin",
]


def load_receipt_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_receipt = pd.read_csv(output_dir / "full_train_receipt_summary.csv")
    train_field = pd.read_csv(output_dir / "full_train_field_predictions.csv")
    validation_receipt = pd.read_csv(output_dir / "validation_receipt_summary.csv")
    validation_field = pd.read_csv(output_dir / "validation_field_predictions.csv")
    return train_receipt, train_field, validation_receipt, validation_field


def load_public_test_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    public_receipt = pd.read_csv(output_dir / "public_test_receipt_summary.csv")
    public_field = pd.read_csv(output_dir / "public_test_field_predictions.csv")
    return public_receipt, public_field


def _first_value(series: pd.Series) -> Any:
    return series.iloc[0] if not series.empty else np.nan


def build_receipt_risk_features(receipt_df: pd.DataFrame, field_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = receipt_df.copy()

    for column in (RISK_LABEL, UNRECOVERABLE_LABEL):
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].astype(float)

    field_work = field_df.copy()
    field_work["predicted_text"] = field_work["predicted_text"].fillna("").astype(str)
    field_work["predicted_text_len"] = field_work["predicted_text"].str.len().astype(float)

    pivot_specs = {
        "confidence": "confidence",
        "margin": "margin",
        "candidate_count": "candidate_count",
        "predicted_text_len": "predicted_text_len",
    }
    for value_column, prefix in pivot_specs.items():
        pivot = (
            field_work.pivot_table(
                index=["doc_id", "dataset", "split"],
                columns="field_name",
                values=value_column,
                aggfunc="first",
            )
            .rename(columns={field_name: f"{prefix}_{field_name}" for field_name in FIELD_NAMES})
            .reset_index()
        )
        feature_df = feature_df.merge(pivot, on=["doc_id", "dataset", "split"], how="left")

    source_flags = (
        field_work.assign(source_key=lambda df: df["field_name"] + "_" + df["predicted_source"].fillna("unknown").astype(str))
        .pivot_table(
            index=["doc_id", "dataset", "split"],
            columns="source_key",
            values="candidate_count",
            aggfunc="size",
            fill_value=0,
        )
        .reset_index()
    )
    source_columns = [column for column in source_flags.columns if column not in {"doc_id", "dataset", "split"}]
    if source_columns:
        source_flags = source_flags.rename(columns={column: f"sourceflag_{column}" for column in source_columns})
        feature_df = feature_df.merge(source_flags, on=["doc_id", "dataset", "split"], how="left")

    confidence_columns = [f"confidence_{field_name}" for field_name in FIELD_NAMES]
    margin_columns = [f"margin_{field_name}" for field_name in FIELD_NAMES]

    feature_df["low_confidence_field_count"] = (feature_df[confidence_columns] < 0.90).sum(axis=1).astype(float)
    feature_df["very_low_confidence_field_count"] = (feature_df[confidence_columns] < 0.75).sum(axis=1).astype(float)
    feature_df["low_margin_field_count"] = (feature_df[margin_columns] < 0.05).sum(axis=1).astype(float)
    feature_df["very_low_margin_field_count"] = (feature_df[margin_columns] < 0.01).sum(axis=1).astype(float)
    feature_df["total_minus_mean_confidence"] = feature_df["confidence_total"] - feature_df["mean_confidence"]
    feature_df["total_minus_mean_margin"] = feature_df["margin_total"] - feature_df["mean_margin"]

    numeric_columns = [
        column
        for column in feature_df.columns
        if column not in {"doc_id", "dataset", "split", "address_prediction", "company_prediction", "date_prediction", "total_prediction"}
    ]
    feature_df[numeric_columns] = feature_df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    feature_df = feature_df.fillna(0.0)
    return feature_df


def risk_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    excluded = {
        "doc_id",
        "dataset",
        "split",
        "address_prediction",
        "company_prediction",
        "date_prediction",
        "total_prediction",
        RISK_LABEL,
        UNRECOVERABLE_LABEL,
        "n_fields",
        "n_correct",
        "n_recoverable",
    }
    return [column for column in feature_df.columns if column not in excluded]


def heuristic_risk_score(feature_df: pd.DataFrame) -> np.ndarray:
    min_confidence_risk = 1.0 - np.clip(feature_df["min_confidence"].to_numpy(dtype=float), 0.0, 1.0)
    min_margin_risk = 1.0 - np.clip(feature_df["min_margin"].to_numpy(dtype=float), 0.0, 1.0)
    low_conf_count = feature_df["low_confidence_field_count"].to_numpy(dtype=float) / max(len(FIELD_NAMES), 1)
    low_margin_count = feature_df["low_margin_field_count"].to_numpy(dtype=float) / max(len(FIELD_NAMES), 1)
    total_confidence_risk = 1.0 - np.clip(feature_df["confidence_total"].to_numpy(dtype=float), 0.0, 1.0)
    score = (
        0.35 * min_confidence_risk
        + 0.25 * min_margin_risk
        + 0.15 * low_conf_count
        + 0.10 * low_margin_count
        + 0.15 * total_confidence_risk
    )
    return np.clip(score, 0.0, 1.0)


@dataclass
class ThresholdSelection:
    threshold: float
    recall: float
    precision: float
    verify_rate: float


def threshold_metrics(y_true: Sequence[float], risk_scores: Sequence[float], threshold: float) -> dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=float)
    score_array = np.asarray(risk_scores, dtype=float)
    predicted_verify = score_array >= threshold
    positive_mask = y_true_array >= 0.5
    negative_mask = ~positive_mask

    tp = float(np.sum(predicted_verify & positive_mask))
    fp = float(np.sum(predicted_verify & negative_mask))
    tn = float(np.sum((~predicted_verify) & negative_mask))
    fn = float(np.sum((~predicted_verify) & positive_mask))

    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    verify_rate = float(np.mean(predicted_verify))
    false_accept_rate = fn / max(tp + fn, 1.0)
    safe_accept_rate = tn / max(tn + fp, 1.0)
    accuracy = (tp + tn) / max(len(y_true_array), 1)

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "verify_rate": verify_rate,
        "false_accept_rate": false_accept_rate,
        "safe_accept_rate": safe_accept_rate,
        "accuracy": accuracy,
    }


def choose_threshold(y_true: Sequence[float], risk_scores: Sequence[float], target_recall: float = 0.90) -> ThresholdSelection:
    y_true_array = np.asarray(y_true, dtype=float)
    score_array = np.asarray(risk_scores, dtype=float)

    unique_thresholds = sorted(set(float(score) for score in score_array), reverse=True)
    candidate_thresholds = [1.01] + unique_thresholds + [0.0]

    best_metrics: dict[str, float] | None = None
    for threshold in candidate_thresholds:
        metrics = threshold_metrics(y_true_array, score_array, threshold)
        if metrics["recall"] >= target_recall:
            if best_metrics is None:
                best_metrics = metrics
                continue
            if metrics["verify_rate"] < best_metrics["verify_rate"]:
                best_metrics = metrics
            elif np.isclose(metrics["verify_rate"], best_metrics["verify_rate"]) and metrics["precision"] > best_metrics["precision"]:
                best_metrics = metrics

    if best_metrics is None:
        best_metrics = max(
            (threshold_metrics(y_true_array, score_array, threshold) for threshold in candidate_thresholds),
            key=lambda metrics: (metrics["recall"], -metrics["verify_rate"]),
        )

    return ThresholdSelection(
        threshold=float(best_metrics["threshold"]),
        recall=float(best_metrics["recall"]),
        precision=float(best_metrics["precision"]),
        verify_rate=float(best_metrics["verify_rate"]),
    )


def threshold_sweep_table(y_true: Sequence[float], risk_scores: Sequence[float]) -> pd.DataFrame:
    score_array = np.asarray(risk_scores, dtype=float)
    thresholds = [1.01] + sorted(set(float(score) for score in score_array), reverse=True) + [0.0]
    rows = [threshold_metrics(y_true, risk_scores, threshold) for threshold in thresholds]
    return pd.DataFrame(rows).drop_duplicates(subset=["threshold"]).reset_index(drop=True)


class ReceiptRiskLogisticModel:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.feature_names: list[str] = []

    def fit(self, feature_df: pd.DataFrame, label_column: str = RISK_LABEL) -> "ReceiptRiskLogisticModel":
        self.feature_names = risk_feature_columns(feature_df)
        X = self._prepare_X(feature_df)
        y = feature_df[label_column].astype(int)
        self.pipeline.fit(X, y)
        return self

    def _prepare_X(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        X = feature_df.copy()
        for feature_name in self.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = 0.0
        return X[self.feature_names].fillna(0.0)

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_X(feature_df)
        return self.pipeline.predict_proba(X)[:, 1]

    def weight_table(self) -> pd.DataFrame:
        classifier: LogisticRegression = self.pipeline.named_steps["classifier"]
        rows = []
        for feature_name, weight in zip(self.feature_names, classifier.coef_[0]):
            rows.append(
                {
                    "feature_name": feature_name,
                    "weight": float(weight),
                    "abs_weight": float(abs(weight)),
                }
            )
        return pd.DataFrame(rows).sort_values("abs_weight", ascending=False).reset_index(drop=True)


class RiskFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class RiskMLP(nn.Module):
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


class ReceiptRiskMLPModel:
    def __init__(
        self,
        random_state: int = 42,
        hidden_dims: Sequence[int] = (32, 16),
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 100,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
    ) -> None:
        self.random_state = random_state
        self.hidden_dims = tuple(hidden_dims)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_names: list[str] = []
        self.model: RiskMLP | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.training_history: list[dict[str, float]] = []

    def _prepare_X(self, feature_df: pd.DataFrame) -> np.ndarray:
        X = feature_df.copy()
        for feature_name in self.feature_names:
            if feature_name not in X.columns:
                X[feature_name] = 0.0
        values = X[self.feature_names].fillna(0.0).to_numpy(dtype=np.float32)
        if self.feature_mean is not None and self.feature_std is not None:
            values = (values - self.feature_mean) / self.feature_std
        return values

    def fit(self, feature_df: pd.DataFrame, label_column: str = RISK_LABEL) -> "ReceiptRiskMLPModel":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.feature_names = risk_feature_columns(feature_df)
        X = feature_df[self.feature_names].fillna(0.0).to_numpy(dtype=np.float32)
        y = feature_df[label_column].astype(np.float32).to_numpy()

        feature_mean = X.mean(axis=0, keepdims=True)
        feature_std = X.std(axis=0, keepdims=True)
        feature_std[feature_std < 1e-6] = 1.0
        X = (X - feature_mean) / feature_std
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        if len(indices) < 10:
            train_indices = indices
            val_indices = indices
        else:
            val_size = max(1, int(round(0.15 * len(indices))))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            if len(train_indices) == 0:
                train_indices = val_indices

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]

        dataset = RiskFeatureDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        self.model = RiskMLP(len(self.feature_names), self.hidden_dims, self.dropout).to(self.device)
        positive_count = float(y_train.sum())
        negative_count = float(len(y_train) - positive_count)
        pos_weight_value = negative_count / max(positive_count, 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        best_state = None
        best_val_loss = float("inf")
        self.training_history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss_sum = 0.0
            train_examples = 0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                batch_size_actual = int(batch_features.shape[0])
                train_loss_sum += float(loss.item()) * batch_size_actual
                train_examples += batch_size_actual

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor)
                val_loss = float(criterion(val_logits, y_val_tensor).item())

            mean_train_loss = train_loss_sum / max(train_examples, 1)
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": mean_train_loss,
                    "val_loss": val_loss,
                }
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

        if best_state is not None and self.model is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Risk MLP model has not been fit.")
        X = self._prepare_X(feature_df)
        self.model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(tensor)
            return torch.sigmoid(logits).detach().cpu().numpy()

    def weight_table(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(columns=["feature_name", "weight", "abs_weight"])
        first_linear = next((layer for layer in self.model.network if isinstance(layer, nn.Linear)), None)
        if first_linear is None:
            return pd.DataFrame(columns=["feature_name", "weight", "abs_weight"])
        weight_matrix = first_linear.weight.detach().cpu().numpy()
        feature_norms = np.linalg.norm(weight_matrix, axis=0)
        rows = [
            {"feature_name": feature_name, "weight": float(norm), "abs_weight": float(norm)}
            for feature_name, norm in zip(self.feature_names, feature_norms)
        ]
        return pd.DataFrame(rows).sort_values("abs_weight", ascending=False).reset_index(drop=True)

    def training_history_frame(self) -> pd.DataFrame:
        if not self.training_history:
            return pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
        return pd.DataFrame(self.training_history)


def gate_decision_table(
    feature_df: pd.DataFrame,
    risk_scores: Sequence[float],
    threshold: float,
    score_column: str,
    label_column: str = RISK_LABEL,
) -> pd.DataFrame:
    decision_df = feature_df.copy()
    decision_df[score_column] = np.asarray(risk_scores, dtype=float)
    decision_df["verify_threshold"] = float(threshold)
    decision_df["gate_action"] = np.where(decision_df[score_column] >= threshold, "verify", "accept")
    if label_column in decision_df.columns:
        decision_df["risk_label"] = decision_df[label_column]
    return decision_df


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
