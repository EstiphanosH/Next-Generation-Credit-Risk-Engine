
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib, os
import numpy as np
import pandas as pd

app = FastAPI(title="Credit Risk Scoring API", version="1.0")

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "models/hybrid_ensemble/artifacts")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lightgbm_model.txt")

_pre = None
_model = None

def _load():
    global _pre, _model
    if _pre is None or _model is None:
        if not (os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH)):
            raise RuntimeError("Artifacts not found. Train the model first.")
        _pre = joblib.load(PREPROCESSOR_PATH)
        import lightgbm as lgb
        _model = lgb.Booster(model_file=MODEL_PATH)
    return _pre, _model

class ScoreRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(...)

class ScoreResponse(BaseModel):
    scores: List[float]
    detail: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ScoreResponse)
def predict(req: ScoreRequest):
    pre, model = _load()
    try:
        X_raw = pd.DataFrame(req.records)
        X = pre.transform(X_raw)
        proba = model.predict(X)
        if hasattr(proba, "ndim") and proba.ndim > 1:
            scores = proba.max(axis=1).tolist()
        else:
            scores = proba.tolist()
        return ScoreResponse(scores=scores)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
