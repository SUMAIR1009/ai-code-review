The CodeXGLUE defect detection dataset (Devign) is not stored in this repository due to size and is excluded via .gitignore. To reproduce, run:

```python
from datasets import load_dataset
ds = load_dataset('google/code_x_glue_cc_defect_detection')
```

This yields 21,854 train / 2,732 validation / 2,732 test C functions labelled buggy (target=1) or clean (target=0), approximately 44.9% buggy. Verified locally June 2025.
