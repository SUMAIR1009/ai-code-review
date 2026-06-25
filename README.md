# COM748 MSc Project — AI-Assisted Automated Code Review Tool for Software Quality Assurance

## Project Information
- **Student:** Muhammad Sumair
- **Student Number:** 20098428
- **Supervisor:** Ajmal Gharib
- **Module:** COM748 Masters Research Project
- **University:** Ulster University / QAHE London

## Project Overview
This project develops an AI-assisted code review tool that uses fine-tuned
pre-trained language models (CodeBERT, CodeT5) and SHAP-based explainability
to detect software defects and generate interpretable review feedback. Outputs
are evaluated against ISO/IEC 25010 software quality characteristics.

## Project Structure
- `data/raw/` — downloaded datasets (CodeXGLUE, ManySStuBs4J, Gerrit, SARD, CodeSearchNet)
- `data/processed/` — tokenised and split data (80/10/10)
- `models/` — saved model checkpoints (CodeBERT, CodeT5)
- `notebooks/` — Jupyter EDA and analysis notebooks
- `src/` — all Python source scripts
- `results/` — evaluation outputs, charts, reports
- `dissertation/` — dissertation chapter drafts

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/review.py --file yourcode.py --output report.json
```

## Key Technologies
CodeBERT · CodeT5 · SHAP · PyTorch · Hugging Face Transformers · scikit-learn
