# CORDProj_38616

Risk-aware verification for financial document processing using CORD and SROIE datasets.

## Project Overview

This project builds a cost-sensitive decision system for automated receipt extraction pipelines. The goal is to use model uncertainty and consistency signals to decide when to accept outputs versus trigger verification.

Core idea:
image → extraction → risk score → (accept vs verify)

## Stage 1 Baseline

The repo now includes a simple layout-aware extraction pipeline for SROIE in
`src/sroie_vlm.py`.

Two model options are available:

- `logistic`: a lightweight interpretable baseline
- `mlp`: a small PyTorch neural network scorer suitable for a neural-networks course

Both versions:

- build candidate field values directly from OCR lines and short line spans
- score them using text, layout, and anchor features
- output field-level confidence, score margins, and receipt-level summary tables

Run it with:

```bash
python src/train_sroie_vlm.py --model-type logistic
python src/train_sroie_vlm.py --model-type mlp
```

By default, outputs are written to:

- `Outputs/sroie_extraction/logistic_baseline/`
- `Outputs/sroie_extraction/mlp_neural/`

Those tables can be reused directly by the later risk-model and verification stages.

## Datasets

- CORD: ~11k receipts with OCR text, layout, and structured labels  
- SROIE: ~1k receipts with labeled key fields (total, date, company, address)

## Setup

Clone the repository:

```bash
git clone https://github.com/quentinleny/CORDProj_38616
cd CORDProj_38616
```

Run the one-time setup script:

```bash
python setup_project.py
```

This will:

- install the Python dependencies from `requirements.txt`
- create the expected `Data/CORD` and `Data/SROIE` folder structure

If you also want to download CORD automatically during setup:

```bash
python setup_project.py --download-cord
```

## Dataset Download Instructions

### CORD v2

Dataset is from:
https://huggingface.co/datasets/naver-clova-ix/cord-v2

```bash
python setup_project.py --download-cord
```

Note: You may see a warning about unauthenticated Hugging Face requests. This is expected and can be ignored unless download speed/rate limits become an issue.

### SROIE

Download from:
https://rrc.cvc.uab.es/?ch=13&com=downloads

Steps:
1. Create an account on the ICDAR Robust Reading Competition site
2. Verify your email
3. Log in and navigate to the downloads page
4. Access the provided Google Drive link
5. Download the dataset files (train_img.zip, train_gt.zip, test_img.zip)
6. Unzip the download
7. Move extracted contents into the folders created by `python setup_project.py`

### Final Structure

```bash
CORDPROJ_38616/
├── Code/
│
├── Data/
│   ├── CORD/
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   │
│   ├── SROIE/
│   │   ├── 0325updated.task1train(626p)/
│   │   ├── 0325updated.task2train(626p)/
│   │   ├── task1&2_test(361p)/
│   │   ├── task3-test(347p)/
│   │   └── text.task1&2-test(361p)/
│   │
│   └── Misc/
│

```
