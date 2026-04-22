# Risk gate: OCR vs pre-parsed pipeline — slide notes

## Setup
- Held-out sample: 100 SROIE train receipts (sorted by doc_id, image folder for the official test split is empty).
- Pre-parsed pipeline: SROIE-provided gold OCR tokens + boxes.
- OCR pipeline: easyocr tokens + boxes from the same JPGs.
- Both pipelines reuse the same trained risk gate (`models/risk_gate.pkl`) and identical field labels — only the OCR layer changes.
- Action thresholds: `auto_accept` < 0.30 ≤ `review` < 0.65 ≤ `human_required`.

## Headline numbers
- Action agreement: **71/100 (71%)**.
- Mean |risk Δ|: **0.231**, max |Δ|: **0.976**, signed mean Δ: **-0.187**.
- Pearson correlation of risk scores across pipelines: **0.593**.
- Verify-rate (review + human_required): pre-parsed **52%** vs OCR **34%**.

## Action distribution
| action | pre-parsed | ocr |
|---|---|---|
| auto_accept | 48 | 66 |
| review | 4 | 7 |
| human_required | 48 | 27 |

## Action transitions (rows = pre-parsed, cols = ocr)
| | auto_accept | review | human_required |
|---|---|---|---|
| auto_accept | 44 | 2 | 2 |
| review | 2 | 2 | 0 |
| human_required | 20 | 3 | 25 |

## Top movers (largest |Δ|)
| doc | pre score | ocr score | Δ | pre action | ocr action | tok pre | tok ocr |
|---|---|---|---|---|---|---|---|
| 38 | 1.000 | 0.024 | -0.976 | human_required | auto_accept | 69 | 64 |
| 13 | 1.000 | 0.024 | -0.976 | human_required | auto_accept | 60 | 62 |
| 90 | 1.000 | 0.024 | -0.976 | human_required | auto_accept | 53 | 73 |
| 20 | 0.994 | 0.024 | -0.970 | human_required | auto_accept | 59 | 62 |
| 22 | 1.000 | 0.041 | -0.959 | human_required | auto_accept | 41 | 50 |

## Token-count drift
- Median token-count gap (ocr − pre-parsed): **+6**.
- Correlation of token-count gap with risk Δ: **+0.115**.

## Figures
- `risk_score_scatter.png` — per-doc score parity (red = action changed).
- `risk_delta_hist.png` — distribution of Δ; bias and spread at a glance.
- `action_transition_heatmap.png` — confusion of routing decisions.
- `token_count_vs_delta.png` — OCR token-count gap vs score Δ.
