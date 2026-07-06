# ============================================================
# CodeXGLUE Preprocessing + Tokenisation Pipeline (v2)
# COM748 MSc Project - Muhammad Sumair (20098428)
#
# CANONICAL-SPLIT-PRESERVING version.
# Keeps Devign's original train/valid/test partitions intact
# (for comparability with published results and to protect the
# held-out test set), and removes only cross-split leakage:
# any training function that also appears in validation or test
# is dropped from TRAIN, leaving valid/test untouched.
# Then tokenises with CodeBERT (max_length=512, truncation).
# ============================================================

import json
import os
import pandas as pd
from transformers import AutoTokenizer

# ---- Configuration ----
DATA_DIR = "data/raw/codexglue"      # adjust if your folder differs
OUT_DIR  = "data/processed"
TOKENIZER_NAME = "microsoft/codebert-base"
MAX_LEN = 512

os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load the three canonical splits ----
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

print("Loading canonical splits...")
train = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
valid = load_jsonl(os.path.join(DATA_DIR, "valid.jsonl"))
test  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
print(f"  Raw: {len(train):,} train / {len(valid):,} valid / {len(test):,} test")

# ---- Normalise & drop malformed/empty records ----
for name, df in [("train", train), ("valid", valid), ("test", test)]:
    df["func"] = df["func"].astype(str)

def clean(df):
    before = len(df)
    df = df[df["func"].str.strip().str.len() > 0]
    df = df[df["target"].isin([0, 1])]
    return df.reset_index(drop=True), before - len(df)

train, dt = clean(train)
valid, dv = clean(valid)
test,  dte = clean(test)
print(f"Removed malformed/empty: {dt} train, {dv} valid, {dte} test")

# ---- Within-split exact-duplicate removal (keep first) ----
def dedup_within(df):
    before = len(df)
    df = df.drop_duplicates(subset="func", keep="first").reset_index(drop=True)
    return df, before - len(df)

train, wt = dedup_within(train)
valid, wv = dedup_within(valid)
test,  wte = dedup_within(test)
print(f"Removed within-split duplicates: {wt} train, {wv} valid, {wte} test")

# ---- Cross-split leakage removal ----
# Protect valid/test: remove from TRAIN any function that also
# appears in validation or test. Valid/test remain untouched.
eval_funcs = set(valid["func"]) | set(test["func"])
before = len(train)
train = train[~train["func"].isin(eval_funcs)].reset_index(drop=True)
cross_removed = before - len(train)
print(f"Removed {cross_removed} cross-split leakage functions from TRAIN "
      f"(valid/test left intact).")

print(f"\nFinal canonical splits:")
print(f"  train {len(train):,} / valid {len(valid):,} / test {len(test):,}")

# ---- Tokenise with CodeBERT ----
print(f"\nLoading tokenizer: {TOKENIZER_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def tokenise(df):
    enc = tokenizer(
        df["func"].tolist(),
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )
    targets = df["target"].tolist()
    out = []
    for i in range(len(df)):
        out.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "label": int(targets[i]),
        })
    return out

print("Tokenising (a few minutes)...")
splits = {"train": train, "validation": valid, "test": test}
for name, df in splits.items():
    toks = tokenise(df)
    out_path = os.path.join(OUT_DIR, f"{name}_tokenised.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(toks, f)
    print(f"  Saved {name}: {len(toks):,} samples -> {out_path}")

# ---- Truncation report (real BPE token counts) ----
all_funcs = pd.concat([train, valid, test])["func"]
trunc = all_funcs.apply(lambda s: len(tokenizer.encode(s)) > MAX_LEN).mean() * 100

print("\n--- PREPROCESSING COMPLETE ---")
print(f"  Canonical splits preserved (Devign train/valid/test).")
print(f"  train {len(train):,} / valid {len(valid):,} / test {len(test):,}")
print(f"  Cross-split leakage removed from train: {cross_removed}")
print(f"  Functions exceeding {MAX_LEN} tokens (truncated): {trunc:.1f}%")
print(f"  Output written to: {OUT_DIR}/")