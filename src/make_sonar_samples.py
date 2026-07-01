# ============================================================
# Generate .c files from CodeXGLUE test functions for SonarQube
# COM748 MSc Project - Muhammad Sumair (20098428)
# Takes a balanced sample (100 buggy + 100 clean) and writes
# each function as an individual .c file, plus a ground-truth CSV.
# ============================================================

import json
import os
import csv
import pandas as pd

DATA_DIR = "data/raw/codexglue"      # adjust if your folder differs
OUT_DIR  = "sonar_samples"
N_PER_CLASS = 100
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load the test split ----
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

test = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
print(f"Loaded test split: {len(test):,} functions")

# ---- Balanced sample ----
buggy = test[test["target"] == 1].sample(n=N_PER_CLASS, random_state=SEED)
clean = test[test["target"] == 0].sample(n=N_PER_CLASS, random_state=SEED)
sample = pd.concat([buggy, clean]).reset_index(drop=True)
print(f"Sampled {len(buggy)} buggy + {len(clean)} clean = {len(sample)} functions")

# ---- Write each function as a .c file + build ground-truth CSV ----
ground_truth = []
for i, row in sample.iterrows():
    fname = f"func_{i:04d}.c"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(row["func"])
    ground_truth.append({"file": fname, "true_label": int(row["target"])})

# ---- Save ground-truth labels ----
with open("sonar_ground_truth.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "true_label"])
    writer.writeheader()
    writer.writerows(ground_truth)

print(f"\nDone. Wrote {len(sample)} .c files to {OUT_DIR}/")
print("Ground-truth labels saved to sonar_ground_truth.csv")
print(f"  Buggy (1): {N_PER_CLASS}   Clean (0): {N_PER_CLASS}")