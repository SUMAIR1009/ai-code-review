# ============================================================
# review.py — AI-Assisted Code Review CLI
# COM748 MSc Project - Muhammad Sumair (20098428)
#
# Usage:
#   python src\review.py path\to\file.c
#   python src\review.py path\to\file.c --explain   (adds SHAP, slow on CPU)
#   python src\review.py path\to\file.c --json      (machine-readable output)
# ============================================================

import argparse
import json
import os
import sys

import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "codebert_final")
MAX_LEN = 512

def load_model():
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    return model, tokenizer

def predict(model, tokenizer, code):
    enc = tokenizer(code, return_tensors="pt", truncation=True,
                    max_length=MAX_LEN, padding="max_length")
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    probs = torch.softmax(out.logits, dim=1)[0]
    p_buggy = float(probs[1])
    n_tokens = int(enc["attention_mask"].sum())
    truncated = len(tokenizer.encode(code)) > MAX_LEN
    return p_buggy, truncated, n_tokens

def priority(p):
    if p >= 0.85: return "HIGH — review before merge"
    if p >= 0.60: return "MEDIUM — review recommended"
    if p >= 0.40: return "LOW — borderline, reviewer discretion"
    return "MINIMAL — no strong defect signal"

def explain(model, tokenizer, code, top_k=10):
    import shap
    import numpy as np
    import scipy.special

    def f(texts):
        out = []
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True,
                            max_length=MAX_LEN, padding="max_length")
            with torch.no_grad():
                o = model(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"])
            out.append(scipy.special.softmax(o.logits.numpy(), axis=1)[0])
        return np.array(out)

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(f, masker)
    sv = explainer([code])
    vals = sv[0, :, 1].values
    toks = sv[0, :, 1].data
    pairs = sorted(zip(toks, vals), key=lambda x: -abs(x[1]))[:top_k]
    return [(t.strip(), round(float(v), 4)) for t, v in pairs if t.strip()]

def main():
    ap = argparse.ArgumentParser(description="AI-assisted code review (CodeBERT + SHAP)")
    ap.add_argument("file", help="C source file to review")
    ap.add_argument("--explain", action="store_true",
                    help="add SHAP token attribution (slow on CPU: several minutes)")
    ap.add_argument("--json", action="store_true", help="output JSON")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        sys.exit(f"File not found: {args.file}")

    with open(args.file, "r", encoding="utf-8", errors="replace") as fh:
        code = fh.read()

    model, tokenizer = load_model()
    p_buggy, truncated, n_tokens = predict(model, tokenizer, code)
    verdict = "LIKELY DEFECTIVE" if p_buggy >= 0.5 else "LIKELY CLEAN"

    result = {
        "file": args.file,
        "defect_probability": round(p_buggy, 3),
        "verdict": verdict,
        "review_priority": priority(p_buggy),
        "tokens_analysed": min(n_tokens, MAX_LEN),
        "truncated_at_512_tokens": truncated,
    }

    if args.explain:
        print("Generating SHAP explanation (this can take several minutes on CPU)...",
              file=sys.stderr)
        result["top_contributing_tokens"] = explain(model, tokenizer, code)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=" * 58)
    print("AI-ASSISTED CODE REVIEW REPORT")
    print("=" * 58)
    print(f"File:               {result['file']}")
    print(f"Defect probability: {result['defect_probability']}")
    print(f"Verdict:            {result['verdict']}")
    print(f"Review priority:    {result['review_priority']}")
    print(f"Tokens analysed:    {result['tokens_analysed']}"
          + ("  (input truncated at 512 tokens)" if truncated else ""))
    if args.explain and result.get("top_contributing_tokens"):
        print("-" * 58)
        print("Top contributing tokens (positive = pushes toward defective):")
        for tok, val in result["top_contributing_tokens"]:
            print(f"  {val:+.4f}  {tok}")
    print("=" * 58)

if __name__ == "__main__":
    main()