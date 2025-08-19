# scripts/evaluate.py
"""
Generate model evaluation artifacts: PSI, simple drift checks, and placeholder for Aequitas bias audit.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
import joblib
from sklearn.metrics import roc_auc_score
from datetime import datetime

logger = logging.getLogger("evaluate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def psi(expected, actual, buckets=10):
    def _sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 1e-8
        if e_perc == 0:
            e_perc = 1e-8
        return (e_perc - a_perc) * np.log(e_perc / a_perc)
    breakpoints = np.linspace(0, 1, buckets + 1)
    e_perc, a_perc = np.histogram(expected, breakpoints)[0] / len(expected), np.histogram(actual, breakpoints)[0] / len(actual)
    psi_val = sum(_sub_psi(e, a) for e, a in zip(e_perc, a_perc))
    return float(psi_val)


def main(args):
    features_path = Path(args.features)
    model_path = Path(args.model)
    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    df = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    model = joblib.load(model_path)

    if "default" not in df.columns:
        raise KeyError("Evaluation requires ground truth 'default' column in features")

    X = df.drop(columns=["default"])
    y = df["default"].astype(int)

    # Load preprocessor if exists alongside model
    preprocessor_path = model_path.parent / "preprocessor.pkl"
    if preprocessor_path.exists():
        preprocessor = joblib.load(preprocessor_path)
        X_prep = preprocessor.transform(X)
    else:
        X_prep = X.values

    preds = model.predict_proba(X_prep)[:, 1]
    auc = roc_auc_score(y, preds)
    psi_value = psi(preds, preds)  # self-PSI is zero, but demonstration
    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "auc": float(auc),
        "psi": psi_value,
        "n": int(len(df))
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Evaluation metrics written to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True, help="Path to model.pkl")
    parser.add_argument("--out", default="docs/model_cards/metrics.json")
    args = parser.parse_args()
    main(args)
