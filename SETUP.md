# Environment Setup — COM748 AI Code Review Project

This file records the setup commands used to reproduce the project environment.

## Python (preprocessing, tokenisation, modelling)

Python 3.11+ required. Install dependencies:

pip install pandas matplotlib seaborn scikit-learn transformers

Or install everything at once from the pinned list:
pip install -r requirements.txt

## R (exploratory data analysis)

R 4.5+ required. Install packages (run once in the R console):

```r
install.packages(c("jsonlite", "ggplot2"))
```

## Datasets

Raw datasets are excluded from version control (see .gitignore).

See `data/raw/README.md` for download/reproduction instructions.

## Scripts

- `notebooks/eda.R` — exploratory data analysis (class balance, code length, duplicates)
- `src/preprocess.py` — preprocessing + CodeBERT tokenisation (dedup, leakage check, 512-token truncation, 80/10/10 split, seed 42)