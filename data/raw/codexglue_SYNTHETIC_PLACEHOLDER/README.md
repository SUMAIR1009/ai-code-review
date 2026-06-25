# SYNTHETIC PLACEHOLDER — NOT REAL DATA

The JSONL files in this folder (train.jsonl, valid.jsonl, test.jsonl) are
synthetically generated placeholders. They have the correct schema, split
sizes, and class balance as the real CodeXGLUE Defect Detection (Devign)
dataset, but the code functions are fabricated and must NOT be used for
training or evaluation.

## How to replace with real data

Run the following in Google Colab (or any machine with Hugging Face access):

```python
from datasets import load_dataset
import json, os

ds = load_dataset('google/code_x_glue_cc_defect_detection')

os.makedirs('data/raw/codexglue', exist_ok=True)
for split in ['train', 'validation', 'test']:
    fname = 'valid' if split == 'validation' else split
    with open(f'data/raw/codexglue/{fname}.jsonl', 'w') as f:
        for row in ds[split]:
            f.write(json.dumps(row) + '\n')
    print(f'{split}: {len(ds[split]):,} rows written')
```

Once downloaded, delete this folder and use data/raw/codexglue/ instead.

## Real dataset stats (for reference)
- train:      21,854 samples  (44.9% buggy, 55.1% clean)
- validation:  2,732 samples
- test:        2,732 samples
- Columns: func (str), target (int), project (str), commit_id (str)
- Source: Devign dataset via google/code_x_glue_cc_defect_detection on HuggingFace
