# ============================================================
# CodeXGLUE Preprocessing + Tokenisation Pipeline
# COM748 MSc Project - Muhammad Sumair (20098428)
# Performs: UTF-8 normalisation, malformed-record removal,
# exact-duplicate removal, cross-split leakage check,
# CodeBERT tokenisation (max_length=512, truncation),
# and an 80/10/10 split with random_state=42.
# ============================================================

import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# ---- Configuration ----
DATA_DIR = "data/raw/codexglue"      # adjust if your folder differs
OUT_DIR  = "data/processed"
TOKENIZER_NAME = "microsoft/codebert-base"
MAX_LEN = 512
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. Load the three splits ----
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

print("Loading data...")
train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
valid = load_jsonl(os.path.join(DATA_DIR, "valid.jsonl"))
test  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

train["split"] = "train"
valid["split"] = "valid"
test["split"]  = "test"
full = pd.concat([train, valid, test], ignore_index=True)
print(f"  Loaded: {len(train):,} train / {len(valid):,} valid / {len(test):,} test  (total {len(full):,})")

# ---- 2. Normalise & remove malformed/empty records ----
before = len(full)
full["func"] = full["func"].astype(str)
full = full[full["func"].str.strip().str.len() > 0]          # drop empty
full = full[full["target"].isin([0, 1])]                      # valid labels only
print(f"Removed {before - len(full):,} malformed/empty records.")

# ---- 3. Exact-duplicate removal ----
before = len(full)
dups = full["func"].duplicated().sum()
full = full.drop_duplicates(subset="func", keep="first").reset_index(drop=True)
print(f"Removed {dups:,} exact-duplicate functions ({before - len(full):,} rows dropped).")

# ---- 4. Cross-split leakage report (informational) ----
# Count functions that appeared in more than one original split before dedup
leak = (pd.concat([train, valid, test])
        .groupby("func")["split"].nunique())
cross_split = (leak > 1).sum()
print(f"Cross-split duplicate functions detected (pre-dedup): {cross_split:,}")
print("  (deduplication above removes these, preventing train/test leakage.)")

# ---- 5. Re-split 80/10/10 with seed 42 (stratified on label) ----
train_df, temp_df = train_test_split(
    full, test_size=0.20, random_state=SEED, stratify=full["target"])
valid_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["target"])
print(f"Re-split -> train {len(train_df):,} / valid {len(valid_df):,} / test {len(test_df):,}")

# ---- 6. Tokenise with CodeBERT ----
print(f"Loading tokenizer: {TOKENIZER_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def tokenise(df):
    enc = tokenizer(
        df["func"].tolist(),
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )
    out = []
    targets = df["target"].tolist()
    for i in range(len(df)):
        out.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "label": int(targets[i]),
        })
    return out

print("Tokenising (this may take a few minutes)...")
splits = {"train": train_df, "validation": valid_df, "test": test_df}
for name, df in splits.items():
    toks = tokenise(df)
    out_path = os.path.join(OUT_DIR, f"{name}_tokenised.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(toks, f)
    print(f"  Saved {name}: {len(toks):,} samples -> {out_path}")

# ---- 7. Final report ----
trunc = (full["func"].apply(lambda s: len(tokenizer.encode(s)) > MAX_LEN)).mean() * 100
print("\n--- PREPROCESSING COMPLETE ---")
print(f"  Final dataset size: {len(full):,}")
print(f"  Splits: {len(train_df):,} / {len(valid_df):,} / {len(test_df):,} (80/10/10, seed 42)")
print(f"  Functions exceeding {MAX_LEN} tokens (truncated): {trunc:.1f}%")
print(f"  Output written to: {OUT_DIR}/")