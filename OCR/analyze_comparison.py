"""
Analyze the per-document CSV produced by compare_pipelines.py and emit
- four PNG figures into Outputs/ocr_analysis/
- a markdown notes file for a slide deck

The four figures are picked to each carry a single point a slide can
hang on:

  1. risk_score_scatter.png      : pre-parsed vs ocr risk score per doc
  2. risk_delta_hist.png         : how often (and how far) OCR moves the score
  3. action_transition_heatmap.png: confusion of routing decisions
  4. token_count_vs_delta.png    : does OCR-token-count gap drive disagreement?
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_ORDER = ["auto_accept", "review", "human_required"]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in ("action_preparsed", "action_ocr"):
        df[col] = pd.Categorical(df[col], categories=ACTION_ORDER, ordered=True)
    return df


def plot_scatter(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    color = np.where(df["action_changed"], "#d62728", "#2ca02c")
    ax.scatter(df["risk_score_preparsed"], df["risk_score_ocr"], c=color, alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    for thr, label in [(0.3, "low"), (0.65, "high")]:
        ax.axvline(thr, color="#888", linestyle=":", linewidth=0.8)
        ax.axhline(thr, color="#888", linestyle=":", linewidth=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("risk score — pre-parsed OCR")
    ax.set_ylabel("risk score — easyocr")
    ax.set_title("Per-document risk scores: pre-parsed vs OCR")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor="black", label="action unchanged"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markeredgecolor="black", label="action changed"),
    ], loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_delta_hist(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["risk_delta"], bins=30, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    mean = df["risk_delta"].mean()
    ax.axvline(mean, color="red", linestyle="--", linewidth=1, label=f"mean Δ = {mean:+.3f}")
    ax.set_xlabel("risk_score(ocr) − risk_score(pre-parsed)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of OCR-induced risk-score shift")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_transition_heatmap(df: pd.DataFrame, out: Path) -> None:
    cm = pd.crosstab(df["action_preparsed"], df["action_ocr"], dropna=False).reindex(
        index=ACTION_ORDER, columns=ACTION_ORDER, fill_value=0
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(ACTION_ORDER))); ax.set_xticklabels(ACTION_ORDER, rotation=20)
    ax.set_yticks(range(len(ACTION_ORDER))); ax.set_yticklabels(ACTION_ORDER)
    ax.set_xlabel("OCR action"); ax.set_ylabel("Pre-parsed action")
    ax.set_title("Action transitions (pre-parsed → OCR)")
    for i in range(len(ACTION_ORDER)):
        for j in range(len(ACTION_ORDER)):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.values.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return cm


def plot_token_vs_delta(df: pd.DataFrame, out: Path) -> None:
    df = df.copy()
    df["token_gap"] = df["n_tokens_ocr"] - df["n_tokens_preparsed"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    color = np.where(df["action_changed"], "#d62728", "#2ca02c")
    ax.scatter(df["token_gap"], df["risk_delta"], c=color, alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("token-count gap (ocr − pre-parsed)")
    ax.set_ylabel("risk_delta (ocr − pre-parsed)")
    ax.set_title("Does OCR token-count drift predict score drift?")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def write_notes(df: pd.DataFrame, cm: pd.DataFrame, out: Path) -> None:
    n = len(df)
    agree = int((~df["action_changed"]).sum())
    agree_rate = agree / n if n else 0.0
    mean_abs = df["risk_delta"].abs().mean()
    mean_signed = df["risk_delta"].mean()
    max_abs = df["risk_delta"].abs().max()
    pearson = float(df[["risk_score_preparsed", "risk_score_ocr"]].corr().iloc[0, 1])

    pre_dist = df["action_preparsed"].value_counts().reindex(ACTION_ORDER, fill_value=0)
    ocr_dist = df["action_ocr"].value_counts().reindex(ACTION_ORDER, fill_value=0)

    pre_review_rate = (pre_dist["review"] + pre_dist["human_required"]) / n if n else 0.0
    ocr_review_rate = (ocr_dist["review"] + ocr_dist["human_required"]) / n if n else 0.0

    df_sorted = df.reindex(df["risk_delta"].abs().sort_values(ascending=False).index)
    movers = df_sorted.head(5)[
        ["doc_id", "risk_score_preparsed", "risk_score_ocr", "risk_delta",
         "action_preparsed", "action_ocr", "n_tokens_preparsed", "n_tokens_ocr"]
    ]

    token_gap = (df["n_tokens_ocr"] - df["n_tokens_preparsed"])
    median_token_gap = float(token_gap.median())
    token_corr = float(token_gap.corr(df["risk_delta"]))

    lines = [
        "# Risk gate: OCR vs pre-parsed pipeline — slide notes",
        "",
        "## Setup",
        f"- Held-out sample: {n} SROIE train receipts (sorted by doc_id, image folder for the official test split is empty).",
        "- Pre-parsed pipeline: SROIE-provided gold OCR tokens + boxes.",
        "- OCR pipeline: easyocr tokens + boxes from the same JPGs.",
        "- Both pipelines reuse the same trained risk gate (`models/risk_gate.pkl`) and identical field labels — only the OCR layer changes.",
        "- Action thresholds: `auto_accept` < 0.30 ≤ `review` < 0.65 ≤ `human_required`.",
        "",
        "## Headline numbers",
        f"- Action agreement: **{agree}/{n} ({agree_rate:.0%})**.",
        f"- Mean |risk Δ|: **{mean_abs:.3f}**, max |Δ|: **{max_abs:.3f}**, signed mean Δ: **{mean_signed:+.3f}**.",
        f"- Pearson correlation of risk scores across pipelines: **{pearson:.3f}**.",
        f"- Verify-rate (review + human_required): pre-parsed **{pre_review_rate:.0%}** vs OCR **{ocr_review_rate:.0%}**.",
        "",
        "## Action distribution",
        "| action | pre-parsed | ocr |",
        "|---|---|---|",
        *[f"| {a} | {int(pre_dist[a])} | {int(ocr_dist[a])} |" for a in ACTION_ORDER],
        "",
        "## Action transitions (rows = pre-parsed, cols = ocr)",
        "| | " + " | ".join(ACTION_ORDER) + " |",
        "|---|" + "|".join(["---"] * len(ACTION_ORDER)) + "|",
        *[f"| {row} | " + " | ".join(str(int(cm.loc[row, c])) for c in ACTION_ORDER) + " |" for row in ACTION_ORDER],
        "",
        "## Top movers (largest |Δ|)",
        "| doc | pre score | ocr score | Δ | pre action | ocr action | tok pre | tok ocr |",
        "|---|---|---|---|---|---|---|---|",
        *[
            f"| {row.doc_id} | {row.risk_score_preparsed:.3f} | {row.risk_score_ocr:.3f} | "
            f"{row.risk_delta:+.3f} | {row.action_preparsed} | {row.action_ocr} | "
            f"{int(row.n_tokens_preparsed)} | {int(row.n_tokens_ocr)} |"
            for row in movers.itertuples()
        ],
        "",
        "## Token-count drift",
        f"- Median token-count gap (ocr − pre-parsed): **{median_token_gap:+.0f}**.",
        f"- Correlation of token-count gap with risk Δ: **{token_corr:+.3f}**.",
        "",
        "## Talking points",
        f"- The two pipelines route the same way on {agree_rate:.0%} of receipts; disagreement is concentrated on a small tail of large-|Δ| docs.",
        f"- OCR pipeline {'raises' if mean_signed > 0 else 'lowers'} the average risk score by {abs(mean_signed):.3f}, "
        f"shifting the gate's verify-rate from {pre_review_rate:.0%} to {ocr_review_rate:.0%}.",
        "- The biggest movers are the right docs to eyeball next: they show where features derived from raw OCR (token counts, anchor presence) diverge most from the curated SROIE OCR.",
        f"- Token-count gap is {'a useful' if abs(token_corr) > 0.3 else 'a weak'} predictor of score drift (corr {token_corr:+.2f}); {'this is the lever to tighten OCR or feature normalization' if abs(token_corr) > 0.3 else 'so feature drift, not raw token count, is doing most of the work'}.",
        "",
        "## Figures",
        "- `risk_score_scatter.png` — per-doc score parity (red = action changed).",
        "- `risk_delta_hist.png` — distribution of Δ; bias and spread at a glance.",
        "- `action_transition_heatmap.png` — confusion of routing decisions.",
        "- `token_count_vs_delta.png` — OCR token-count gap vs score Δ.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "Outputs" / "ocr_vs_preparsed.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "Outputs" / "ocr_analysis")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.csv)

    plot_scatter(df, args.out_dir / "risk_score_scatter.png")
    plot_delta_hist(df, args.out_dir / "risk_delta_hist.png")
    cm = plot_transition_heatmap(df, args.out_dir / "action_transition_heatmap.png")
    plot_token_vs_delta(df, args.out_dir / "token_count_vs_delta.png")
    write_notes(df, cm, args.out_dir / "slide_notes.md")

    print(f"wrote figures + slide notes to {args.out_dir}")


if __name__ == "__main__":
    main()
