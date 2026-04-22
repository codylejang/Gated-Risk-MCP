# Risk Gate Model - Notes

## Model Architecture Comparison

| Component | Teammate's Neural Extractor | Your Risk Gate |
|-----------|---------------------------|----------------|
| **Task** | Field extraction (company, date, address, total) | Extraction uncertainty scoring |
| **Model Type** | MLP (64→32→1) with Sigmoid | Gradient Boosting (100 trees, depth=4) |
| **Calibration** | N/A | Isotonic (cv=3) |
| **Input Features** | 37 candidate features | 21 engineered features |
| **Output** | Predicted text + confidence per field | Risk score + action (JSON) |

---

## Performance Metrics

| Metric | Risk Gate Value |
|--------|-----------------|
| **AUC-ROC** | 99.3% |
| **Precision** | 97.0% |
| **Recall** | 95.6% |
| **F1 Score** | 96.3% |
| **Brier Score** | 0.029 |
| **Train/Val** | 500/126 |

---

## Top Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `exact_total_matches` | 41.7% |
| 2 | `n_amount_like_tokens` | 35.2% |
| 3 | `address_len` | 6.1% |
| 4 | `company_len` | 4.4% |
| 5 | `date_len` | 3.8% |
| 6 | `amount_token_ratio` | 3.7% |

> The top 2 features account for **77% of predictive power**, both related to monetary amounts.

---

## Key Design Decisions

1. **Gradient Boosting over MLP**
   - Better for small data (626 receipts)
   - Inherently interpretable (feature importances)
   - Robust to overfitting with shallow trees

2. **Isotonic Calibration**
   - Ensures probabilities are well-calibrated
   - Critical for threshold-based decisions
   - Non-parametric: learns monotonic mapping

3. **Proxy Labels (Weak Supervision)**
   - No human verification labels available
   - Heuristic rules derive `proxy_verify` label
   - Risk signals: field not in OCR, too many amounts, missing anchors

4. **MCP-Compatible Output**
   - JSON format for direct agent tool calls
   - Three-tier action system: `auto_accept`, `review`, `human_required`

## Threshold Decision Logic

| Risk Score | Action | Use Case |
|------------|--------|----------|
| p̂ < 0.30 | `auto_accept` | High confidence, no review needed |
| 0.30 ≤ p̂ < 0.60 | `review` | Moderate confidence, quick check |
| p̂ ≥ 0.60 | `human_required` | Low confidence, manual verification |

---

## Code Reference

```python
# MCP tool call interface
from src.risk_gate import RiskGate

gate = RiskGate()
result = gate.score(
    ocr_tokens=["STORE", "$21.49", "TOTAL"],
    bboxes=[[10,10,50,20], [10,40,60,50], [70,40,110,50]],
    fields={"company": "STORE", "total": "21.49"}
)
# Returns: {"doc_id": "...", "risk_score": 0.23, "action": "auto_accept"}
```
