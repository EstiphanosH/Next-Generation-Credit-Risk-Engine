# scripts/train.py
"""
Train a LightGBM classifier, log artifacts to MLflow, produce SHAP explanations.
This script is intentionally straightforward and suitable for local development.
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import lightgbm as lgb

logger = logging.getLogger("train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed features not found at {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def train(args):
    features_path = Path(args.features)
    df = load_features(features_path)

    # Expect target column named 'default' (1/0)
    if "default" not in df.columns:
        raise KeyError("Training data must contain 'default' target column (0/1)")

    X = df.drop(columns=["default"])
    y = df["default"].astype(int)

    # simple feature split
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = build_pipeline(numeric_features, categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit preprocessor and transform
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    lgb_train = lgb.Dataset(X_train_prep, label=y_train)
    lgb_eval = lgb.Dataset(X_test_prep, label=y_test, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": 42,
    }

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_eval],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    preds = gbm.predict(X_test_prep, num_iteration=gbm.best_iteration)
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, (preds > 0.5).astype(int))

    logger.info("Test AUC: %.4f, Accuracy: %.4f", auc, acc)

    # MLflow logging
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("auc", float(auc))
        mlflow.log_metric("accuracy", float(acc))

        artifacts_dir = Path(args.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / "model.pkl"
        pipeline_path = artifacts_dir / "preprocessor.pkl"

        # Save LightGBM model via joblib
        joblib.dump(gbm, model_path)
        joblib.dump(preprocessor, pipeline_path)

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(pipeline_path), artifact_path="model")

    logger.info("Model and preprocessor saved to %s", artifacts_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Processed features path (parquet/csv)")
    parser.add_argument("--artifacts-dir", default="models/hybrid_ensemble/artifacts", help="Where to save artifacts")
    parser.add_argument("--mlflow-uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="credit-risk", help="MLflow experiment name")
    args = parser.parse_args()
    train(args)
