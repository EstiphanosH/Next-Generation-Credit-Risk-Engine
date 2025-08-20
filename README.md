# Next-Generation Credit Risk Engine

[![Python Version](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Overview
A production-ready ML pipeline for credit scoring using alternative data. Supports multiple models and SHAP explainability.

## Repository Structure
```
scripts/                  # Training scripts
data/processed/           # Preprocessed data
models/hybrid_ensemble/artifacts/  # Saved models
docs/model_cards/         # SHAP explainability reports
tests/                    # Unit tests
README.md                 # Project documentation
requirements.txt          # Python dependencies
```

## Installation
```bash
git clone https://github.com/yourusername/Next-Generation-Credit-Risk-Engine.git
cd Next-Generation-Credit-Risk-Engine
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Usage
```bash
python scripts/train.py --processed data/processed/users.parquet --artifacts models/hybrid_ensemble/artifacts
```

## Features
- Data preprocessing & imputation
- Multi-model training & evaluation (ROC-AUC)
- Automatic best model selection
- SHAP explainability reports for the best model

## License
MIT License
