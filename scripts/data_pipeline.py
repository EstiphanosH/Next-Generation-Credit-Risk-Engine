# scripts/data_pipeline.py
"""
Lightweight ETL pipeline for local development.
In production this would be a PySpark job writing Delta. Here we use pandas
with clear interfaces so the same logic can be migrated.
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger("data_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    if path.suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    elif path.suffix in [".parquet"]:
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported raw file format")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Example cleaning: standardize column names, drop duplicates, fill na
    df = df.rename(columns=lambda c: c.strip().lower().replace(" ", "_"))
    df = df.drop_duplicates()
    # Basic imputations
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df.fillna({"region": "unknown"})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example features
    df["age"] = df["age"].astype(int)
    df["employment_length_years"] = df.get("employment_length_years", 0).astype(int)
    df["credit_history_length_years"] = df.get("credit_history_length_years", 0).astype(int)
    # Cap numeric outliers
    for col in ["annual_income", "dti"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    # Generate simple risk proxy
    df["risk_score_proxy"] = (df["num_open_loans"] * 0.1 + df["dti"].astype(float) * 0.5) / (df["annual_income"] + 1)
    return df


def write_processed(df: pd.DataFrame, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    logger.info("Wrote processed features to %s", out)


def main(args):
    raw_path = Path(args.raw)
    processed_path = Path(args.processed)
    df = read_raw(raw_path)
    df = clean(df)
    df = engineer_features(df)
    write_processed(df, processed_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local data ETL")
    parser.add_argument("--raw", required=True, help="Path to raw CSV/parquet")
    parser.add_argument("--processed", required=True, help="Path to write processed features (parquet/csv)")
    args = parser.parse_args()
    main(args)
