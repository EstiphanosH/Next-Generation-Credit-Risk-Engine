#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import joblib
import logging
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -----------------------------
# Load data
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading processed data from {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    logger.info(f"Loaded data shape: {df.shape}")
    return df

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_features(df: pd.DataFrame, target_col: str):
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    logger.info(f"Dropped rows with missing target. Remaining rows: {df.shape[0]}")

    # Identify categorical and numeric columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    logger.info(f"Categorical columns: {categorical_cols}")
    logger.info(f"Numeric columns: {numeric_cols}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols)
        ]
    )

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_processed = preprocessor.fit_transform(X)
    logger.info(f"Feature matrix shape after preprocessing: {X_processed.shape}")

    return X_processed, y, preprocessor

# -----------------------------
# Train multiple models
# -----------------------------
def train_models(X_train, y_train):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, random_state=42),
        "lightgbm": lgb.LGBMClassifier(n_estimators=200, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training model: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

# -----------------------------
# Evaluate and select best model
# -----------------------------
def evaluate_models(models, X_test, y_test):
    best_model = None
    best_score = 0
    results = {}
    for name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:,1]
        score = roc_auc_score(y_test, y_pred_prob)
        results[name] = score
        logger.info(f"{name} ROC-AUC: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
    logger.info(f"Best model selected: {best_model.__class__.__name__} with ROC-AUC: {best_score:.4f}")
    return best_model, results

# -----------------------------
# SHAP explainability
# -----------------------------
def generate_shap_report(model, X_sample, feature_names, output_path):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap_html = shap.plots.force(shap_values, matplotlib=False, show=False)
    shap.save_html(output_path, shap_values)
    logger.info(f"SHAP report saved to {output_path}")

# -----------------------------
# Main
# -----------------------------
def main(processed_path: str, artifacts_dir: str):
    df = load_data(processed_path)

    target_col = "FraudResult"
    X, y, preprocessor = preprocess_features(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    trained_models = train_models(X_train, y_train)
    best_model, scores = evaluate_models(trained_models, X_test, y_test)

    # Save preprocessor and best model
    os.makedirs(artifacts_dir, exist_ok=True)
    model_artifact_path = os.path.join(artifacts_dir, "best_model.pkl")
    joblib.dump({"model": best_model, "preprocessor": preprocessor}, model_artifact_path)
    logger.info(f"Saved best model and preprocessor to {model_artifact_path}")

    # SHAP report
    shap_output_path = os.path.join("docs/model_cards/shap_report.html")
    os.makedirs(os.path.dirname(shap_output_path), exist_ok=True)

    # Sample 5000 rows if dataset is large
    sample_size = min(5000, X_train.shape[0])
    X_sample = X_train[:sample_size]
    generate_shap_report(best_model, X_sample, feature_names=None, output_path=shap_output_path)

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed/users.parquet")
    ap.add_argument("--artifacts", default="models/hybrid_ensemble/artifacts")
    args = ap.parse_args()
    main(args.processed, args.artifacts)
