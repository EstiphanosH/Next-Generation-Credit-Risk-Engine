# api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, condecimal
from typing import Optional
import joblib
import os
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = os.environ.get("MODEL_PATH", "models/hybrid_ensemble/artifacts/model.pkl")
ENCODER_PATH = os.environ.get("ENCODER_PATH", "models/hybrid_ensemble/artifacts/encoder.pkl")

app = FastAPI(title="Credit Risk Scoring API", version="0.1.0")


class Applicant(BaseModel):
    applicant_id: str = Field(..., description="Unique applicant identifier")
    age: int = Field(..., ge=18, le=120)
    annual_income: condecimal(gt=0)
    employment_length_years: int = Field(..., ge=0)
    num_open_loans: int = Field(..., ge=0)
    dti: condecimal(ge=0)  # debt-to-income ratio
    credit_history_length_years: int = Field(..., ge=0)
    region: Optional[str] = Field(None, description="Geographical region code")


class ScoreResponse(BaseModel):
    applicant_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    decision: str
    probabilities: dict


def _load_artifacts():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    encoder = None
    if Path(ENCODER_PATH).exists():
        encoder = joblib.load(ENCODER_PATH)
    return model, encoder


@app.on_event("startup")
def startup_event():
    try:
        app.state.model, app.state.encoder = _load_artifacts()
    except Exception as e:
        # Do not crash on startup — log and allow endpoints to return helpful error.
        app.state.model, app.state.encoder = None, None
        app.state.startup_error = str(e)


def _prepare_features(payload: Applicant, encoder) -> pd.DataFrame:
    df = pd.DataFrame([payload.dict()])
    # Basic feature engineering: create float columns and missing handling
    df["annual_income"] = df["annual_income"].astype(float)
    df["dti"] = df["dti"].astype(float)
    numeric_cols = ["age", "annual_income", "employment_length_years",
                    "num_open_loans", "dti", "credit_history_length_years"]
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = 0
    # One-hot encode region if encoder provided
    if encoder is not None:
        try:
            enc_df = encoder.transform(df[["region"]])
            enc_df = pd.DataFrame(enc_df.toarray() if hasattr(enc_df, "toarray") else enc_df,
                                  columns=encoder.get_feature_names_out(["region"]))
            df = pd.concat([df.drop(columns=["region"]), enc_df], axis=1)
        except Exception:
            # fallback: drop region
            df = df.drop(columns=["region"], errors="ignore")
    else:
        df = df.drop(columns=["region"], errors="ignore")
    return df[numeric_cols + [c for c in df.columns if c not in numeric_cols]]


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    if getattr(app.state, "startup_error", None):
        raise HTTPException(status_code=503, detail=f"Model not available: {app.state.startup_error}")
    model = app.state.model
    encoder = app.state.encoder
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        X = _prepare_features(applicant, encoder)
        proba = model.predict_proba(X)[:, 1]
        score = float(proba[0])
        decision = "accept" if score < 0.5 else "decline"  # example policy: lower probability of default -> accept
        return ScoreResponse(
            applicant_id=applicant.applicant_id,
            score=score,
            decision=decision,
            probabilities={"default": score, "non_default": 1 - score}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")


@app.get("/health")
def health():
    ok = app.state.model is not None and getattr(app.state, "startup_error", None) is None
    return {"status": "ok" if ok else "degraded", "startup_error": getattr(app.state, "startup_error", None)}
