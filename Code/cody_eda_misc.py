# additional eda on the sroie cleaned dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("Outputs") / "eda"
CSV_PATH = OUTPUT_DIR / "sroie_cleaned_eda.csv"

df = pd.read_csv(CSV_PATH)


# missing values audit
# notebook never does a systematic nulls check across all columns
print("missing values per column")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if missing.empty:
    print("no missing values in any column")
else:
    for col, n in missing.items():
        print(f"  {col}: {n} / {len(df)} ({100*n/len(df):.1f}%)")
print()


# date temporal distribution
# notebook parses dates and extracts year/month but never plots the distribution
print("receipt count by year-month")
df["date_parsed_dt"] = pd.to_datetime(df["date_parsed"], errors="coerce")
ym = df["date_parsed_dt"].dt.to_period("M").value_counts().sort_index()
for period, count in ym.items():
    print(f"  {period}: {count}")

fig, ax = plt.subplots(figsize=(10, 4))
ym.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="black")
ax.set_title("receipts per year-month")
ax.set_xlabel("year-month")
ax.set_ylabel("count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "date_temporal_dist.png", dpi=150)
plt.close()
print("  saved date_temporal_dist.png")
print()


# ocr character level quality indicators
# notebook checks field matching but not character level noise
print("ocr character level quality indicators")
ocr = df["ocr_full_text"].fillna("")
df["ocr_word_count"] = ocr.apply(lambda x: len(x.split()))
df["ocr_digit_ratio"] = ocr.apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
df["ocr_alpha_ratio"] = ocr.apply(lambda x: sum(c.isalpha() for c in x) / max(len(x), 1))
df["ocr_special_ratio"] = 1.0 - df["ocr_digit_ratio"] - df["ocr_alpha_ratio"] - ocr.apply(lambda x: sum(c.isspace() for c in x) / max(len(x), 1))
df["ocr_upper_ratio"] = ocr.apply(lambda x: sum(c.isupper() for c in x) / max(sum(c.isalpha() for c in x), 1))

for col in ["ocr_word_count", "ocr_digit_ratio", "ocr_alpha_ratio", "ocr_special_ratio", "ocr_upper_ratio"]:
    print(f"  {col}:")
    desc = df[col].describe()
    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        print(f"    {stat}: {desc[stat]:.4f}")
print()


# total amount vs receipt complexity correlation
# notebook shows total distribution and ocr stats separately but not their relationship
print("total amount vs ocr complexity correlation")
cols_for_corr = ["total_num", "num_ocr_lines", "ocr_text_len", "img_height", "img_width",
                 "company_len", "address_len", "ocr_word_count"]
corr = df[cols_for_corr].corr()
print(corr.to_string())
print()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols_for_corr)))
ax.set_yticks(range(len(cols_for_corr)))
ax.set_xticklabels(cols_for_corr, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(cols_for_corr, fontsize=8)
for i in range(len(cols_for_corr)):
    for j in range(len(cols_for_corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im)
ax.set_title("correlation heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()
print("  saved correlation_heatmap.png")
print()


# singleton vs frequent merchants
# notebook shows top merchants but doesnt compare singleton merchant receipts to frequent ones
print("singleton vs frequent merchant comparison")
merchant_freq = df["company"].value_counts()
df["merchant_freq"] = df["company"].map(merchant_freq)
df["merchant_bucket"] = pd.cut(df["merchant_freq"], bins=[0, 1, 3, 10, 999], labels=["1", "2-3", "4-10", "11+"])

bucket_stats = df.groupby("merchant_bucket", observed=True).agg(
    n=("receipt_id", "count"),
    avg_total=("total_num", "mean"),
    avg_ocr_lines=("num_ocr_lines", "mean"),
    avg_ocr_len=("ocr_text_len", "mean"),
    hard_pct=("hard_receipt", "mean"),
).round(3)
print(bucket_stats.to_string())
print()


# field match failure breakdown
# notebook flags hard receipts but doesnt show which field fails most often
print("field level match failure rates (normalized matching)")
for col in ["company_in_ocr_norm", "date_in_ocr_norm", "address_in_ocr_norm", "total_in_ocr_norm"]:
    if col in df.columns:
        fail_rate = 1.0 - df[col].mean()
        print(f"  {col.replace('_in_ocr_norm', '')}: {fail_rate:.1%} fail rate ({int(fail_rate * len(df))} receipts)")
print()


# ocr word count distribution
# notebook has line count and char length but not word count
print("ocr word count distribution")
print(df["ocr_word_count"].describe().to_string())

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(df["ocr_word_count"], bins=30, color="#4C72B0", edgecolor="black")
ax.set_title("ocr word count distribution")
ax.set_xlabel("word count")
ax.set_ylabel("receipts")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ocr_word_count_dist.png", dpi=150)
plt.close()
print("  saved ocr_word_count_dist.png")
print()


# address geographic patterns
# notebook checks address matching but not whats in the addresses
print("top cities/regions in addresses")
addr = df["address"].fillna("").str.upper()
cities = ["JOHOR BAHRU", "KUALA LUMPUR", "PETALING JAYA", "SHAH ALAM", "SUBANG",
          "SELANGOR", "PENANG", "IPOH", "MELAKA", "KLANG", "AMPANG", "CHERAS",
          "PUCHONG", "SETIA ALAM", "TAMAN DAYA", "KEPONG", "BANGSAR"]
for city in sorted(cities):
    count = addr.str.contains(city, na=False).sum()
    if count > 0:
        print(f"  {city.lower()}: {count}")
print()

print("done")
print("new outputs: date_temporal_dist.png, correlation_heatmap.png, ocr_word_count_dist.png")
