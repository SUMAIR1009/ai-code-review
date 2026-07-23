import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from review import load_model, predict

model, tokenizer = load_model()
print("Scanning first 15 known-buggy samples...")
for i in range(15):
    path = f"sonar_samples/func_{i:04d}.c"
    with open(path, encoding="utf-8", errors="replace") as f:
        code = f.read()
    p, trunc, _ = predict(model, tokenizer, code)
    flag = "  <-- good demo case" if p >= 0.75 else ""
    print(f"  {path}: {p:.3f}{'  (truncated)' if trunc else ''}{flag}")