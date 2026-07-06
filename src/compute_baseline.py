# ============================================================
# Cppcheck Static-Analysis Baseline — Precision/Recall/F1
# COM748 MSc Project - Muhammad Sumair (20098428)
#
# Reads Cppcheck XML output + CodeXGLUE ground-truth labels,
# applies the mapping rule (Rule A: error OR warning => buggy),
# and computes the static-analysis baseline metrics.
#
# The Cppcheck results file parsed by this script was produced
# with Cppcheck 2.21.0 using the following invocation:
#
#   cppcheck --enable=all --xml --output-file=cppcheck_results.xml sonar_samples
#
# Sample generation: src/make_sonar_samples.py (200 functions,
# 100 buggy / 100 clean, drawn with random seed 42).
# ============================================================

import xml.etree.ElementTree as ET
import csv
from collections import defaultdict, Counter

# ---- Configuration ----
XML_PATH   = "cppcheck_results.xml"       # Cppcheck output (project root)
TRUTH_PATH = "sonar_ground_truth.csv"     # ground-truth labels (project root)

# Mapping rule A: a file is PREDICTED BUGGY if it has any finding
# of these severities. Style/information findings are treated as
# non-defect signals (code quality, not vulnerability).
BUGGY_SEVERITIES = {"error", "warning"}

# ---- Load ground-truth labels ----
truth = {}
with open(TRUTH_PATH, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        truth[row["file"]] = int(row["true_label"])
print(f"Loaded ground truth: {len(truth)} files "
      f"({sum(truth.values())} buggy / {len(truth)-sum(truth.values())} clean)")

# ---- Parse Cppcheck XML ----
tree = ET.parse(XML_PATH)
root = tree.getroot()

file_severities = defaultdict(set)   # filename -> set of severities found
severity_totals = Counter()

for err in root.iter("error"):
    sev = err.get("severity")
    f0 = err.get("file0")
    severity_totals[sev] += 1
    if f0:
        fname = f0.split("/")[-1].split("\\")[-1]   # strip any path
        file_severities[fname].add(sev)

print("\nCppcheck findings by severity:")
for s, c in severity_totals.most_common():
    print(f"  {s:14s}: {c}")

# ---- Apply the mapping rule and score ----
tp = fp = tn = fn = 0
predictions = {}
for fname, true_label in truth.items():
    sevs = file_severities.get(fname, set())
    pred = 1 if (sevs & BUGGY_SEVERITIES) else 0
    predictions[fname] = pred
    if pred == 1 and true_label == 1:
        tp += 1
    elif pred == 1 and true_label == 0:
        fp += 1
    elif pred == 0 and true_label == 0:
        tn += 1
    else:
        fn += 1

# ---- Compute metrics ----
precision = tp / (tp + fp) if (tp + fp) else 0.0
recall    = tp / (tp + fn) if (tp + fn) else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
accuracy  = (tp + tn) / len(truth)

# ---- Report ----
print("\n" + "=" * 55)
print("CPPCHECK STATIC-ANALYSIS BASELINE")
print("Rule A: a function is predicted buggy if Cppcheck")
print("reports any 'error'- or 'warning'-severity finding.")
print("=" * 55)
print(f"Sample size:      {len(truth)} functions "
      f"({sum(truth.values())} buggy / {len(truth)-sum(truth.values())} clean)")
print(f"\nConfusion matrix:")
print(f"  True Positives  (buggy, flagged):    {tp}")
print(f"  False Positives (clean, flagged):    {fp}")
print(f"  True Negatives  (clean, not flagged):{tn}")
print(f"  False Negatives (buggy, missed):     {fn}")
print(f"\nMETRICS:")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-score:  {f1:.3f}")
print(f"  Accuracy:  {accuracy:.3f}")
print("=" * 55)

# ---- Save results to a small CSV for the dissertation ----
with open("baseline_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "precision", "recall", "f1", "accuracy", "sample_size", "rule"])
    w.writerow(["Cppcheck (static analysis)",
                f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", f"{accuracy:.3f}",
                len(truth), "error OR warning => buggy"])
print("\nSaved: baseline_results.csv")