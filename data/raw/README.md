The CodeXGLUE defect detection dataset (Devign) is not stored in this repository due to size and is excluded via .gitignore. To reproduce, run:

```python
from datasets import load_dataset
ds = load_dataset('google/code_x_glue_cc_defect_detection')
```

This yields 21,854 train / 2,732 validation / 2,732 test C functions labelled buggy (target=1) or clean (target=0), approximately 44.9% buggy. Verified locally June 2025.

The code review replication package from Tufano et al. (Zenodo record 5897020, 16.5 MB) has been downloaded and extracted locally. It contains manual analysis spreadsheets with input/target/prediction columns across three tasks (code-to-code, code&comment-to-code, code-to-comment) for the CodeT5 comment-generation task. Excluded from Git via .gitignore.
