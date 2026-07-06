# ============================================================
# Classical ML Baseline — Random Forest & Logistic Regression
# COM748 MSc Project - Muhammad Sumair (20098428)
#
# Trains classical models on hand-crafted lexical/structural
# features from the canonical CodeXGLUE (Devign) training split
# and evaluates on the protected canonical test split.
# Produces the middle tier of the three-tier results comparison
# (Cppcheck -> Classical ML -> CodeBERT).
# ============================================================

import json
import re
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# ---- Configuration ----
DATA_DIR = "data/raw/codexglue"   # canonical raw splits
SEED = 42

# ---- Load canonical splits (same partitions as CodeBERT will use) ----
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
test  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
print(f"  Train: {len(train):,}  Test: {len(test):,}")

# ---- Feature extraction ----
# Hand-crafted lexical/structural features from each C function.
def extract_features(code):
    code = str(code)
    lines = code.count("\n") + 1
    chars = len(code)
    tokens = len(re.findall(r"\w+", code))
    return {
        "n_lines": lines,
        "n_chars": chars,
        "n_tokens": tokens,
        "avg_line_len": chars / lines if lines else 0,
        "n_pointers": code.count("*") + code.count("->"),
        "n_derefs": code.count("->"),
        "n_malloc": len(re.findall(r"\b(malloc|calloc|realloc)\b", code)),
        "n_free": len(re.findall(r"\bfree\b", code)),
        "n_memops": len(re.findall(r"\b(memcpy|memset|memmove|strcpy|strncpy|strcat)\b", code)),
        "n_loops": len(re.findall(r"\b(for|while)\b", code)),
        "n_conds": len(re.findall(r"\b(if|else|switch|case)\b", code)),
        "n_returns": len(re.findall(r"\breturn\b", code)),
        "n_calls": len(re.findall(r"\w+\s*\(", code)),
        "n_assigns": code.count("="),
        "n_brackets": code.count("[") + code.count("]"),
        "n_semicolons": code.count(";"),
        "n_null": len(re.findall(r"\bNULL\b", code)),
        "n_sizeof": len(re.findall(r"\bsizeof\b", code)),
    }

print("Extracting features...")
X_train = pd.DataFrame([extract_features(c) for c in train["func"]])
X_test  = pd.DataFrame([extract_features(c) for c in test["func"]])
y_train = train["target"].astype(int).values
y_test  = test["target"].astype(int).values

# ---- Scale features (needed for Logistic Regression) ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ---- Helper to evaluate and report ----
def evaluate(name, y_true, y_pred):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    a = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\n--- {name} ---")
    print(f"  Precision: {p:.3f}   Recall: {r:.3f}   F1: {f:.3f}   Accuracy: {a:.3f}")
    print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    return {"model": name, "precision": round(p,3), "recall": round(r,3),
            "f1": round(f,3), "accuracy": round(a,3),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn}

results = []

# ---- Model 1: Random Forest ----
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced")
rf.fit(X_train, y_train)                      # RF doesn't need scaling
rf_pred = rf.predict(X_test)
results.append(evaluate("Random Forest", y_test, rf_pred))

# ---- Model 2: Logistic Regression ----
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
lr.fit(X_train_s, y_train)                     # LR uses scaled features
lr_pred = lr.predict(X_test_s)
results.append(evaluate("Logistic Regression", y_test, lr_pred))

# ---- Feature importance (Random Forest) ----
print("\n--- TOP 8 FEATURES (Random Forest) ---")
imp = sorted(zip(X_train.columns, rf.feature_importances_), key=lambda x: -x[1])
for feat, val in imp[:8]:
    print(f"  {feat:16s}: {val:.3f}")

# ---- Save results ----
out = pd.DataFrame(results)
out.to_csv("classical_baseline_results.csv", index=False)
print("\nSaved: classical_baseline_results.csv")
print("\n=== CLASSICAL ML BASELINE COMPLETE ===")
print(f"Evaluated on {len(y_test):,} canonical test functions "
      f"({int(y_test.sum())} buggy / {int((1-y_test).sum())} clean)")