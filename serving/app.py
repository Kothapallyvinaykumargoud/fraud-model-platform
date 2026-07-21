"""FR-8: Prediction Endpoint. Loads the Model Version named by
production_pointer.json (FR-4) — plain file read, no MLflow process
required — and serves fraud verdicts over REST.

Also exposes /metrics (FR-10/FR-11/FR-12): request latency, predicted-class
distribution, and the current Drift Score (set by mlops/drift_check.py via
the shared drift_state.json file, since drift checks run on-demand and the
API process is what Prometheus actually scrapes).
"""
import json
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from model.data import FEATURE_COLUMNS

POINTER_PATH = os.environ.get("PRODUCTION_POINTER_PATH", "production_pointer.json")
DRIFT_STATE_PATH = os.environ.get("DRIFT_STATE_PATH", "models/drift_state.json")

app = FastAPI(title="Fraud Model Platform — Inference API")

PREDICTION_LATENCY = Histogram(
    "inference_latency_seconds", "Time to serve a single prediction"
)
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Predictions served, by verdict", ["verdict"]
)
DRIFT_SCORE = Gauge("drift_score", "Most recent Drift Score computed by mlops/drift_check.py")
DRIFT_THRESHOLD = Gauge("drift_threshold", "Drift Score threshold that triggers retraining")


class TransactionPayload(BaseModel):
    Time: float
    Amount: float
    V: list[float] = Field(..., min_length=28, max_length=28, description="V1..V28 PCA features, in order")


class FraudVerdict(BaseModel):
    model_config = {"protected_namespaces": ()}

    verdict: str
    confidence: float
    model_version: str


def _load_pointer() -> dict:
    if not os.path.exists(POINTER_PATH):
        raise HTTPException(
            status_code=503,
            detail=f"No production model registered — {POINTER_PATH} not found. Run the training pipeline first.",
        )
    with open(POINTER_PATH) as f:
        return json.load(f)


def _load_model():
    pointer = _load_pointer()
    model = joblib.load(pointer["artifact_path"])
    return model, pointer


# Loaded once at process start. A new deployment (new container, per FR-7)
# is how a new Model Version takes effect — matches the pipeline's
# build-and-deploy model rather than hot-reloading inside a running pod.
_model, _pointer = (None, None)


@app.on_event("startup")
def _startup():
    global _model, _pointer
    try:
        _model, _pointer = _load_model()
        print(f"[serving] loaded model version {_pointer['model_version']}")
    except HTTPException as e:
        print(f"[serving] WARNING: {e.detail}")
    if os.path.exists(DRIFT_STATE_PATH):
        with open(DRIFT_STATE_PATH) as f:
            state = json.load(f)
        DRIFT_SCORE.set(state.get("drift_score", 0.0))
        DRIFT_THRESHOLD.set(state.get("threshold", 0.0))


@app.get("/health")
def health():
    if _pointer is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return {"status": "ok", "model_version": _pointer["model_version"]}


@app.post("/predict", response_model=FraudVerdict)
def predict(payload: TransactionPayload):
    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded — see /health")

    row = {"Time": payload.Time, "Amount": payload.Amount}
    row.update({f"V{i+1}": v for i, v in enumerate(payload.V)})
    features = pd.DataFrame([row])[FEATURE_COLUMNS]

    start = time.perf_counter()
    proba = _model.predict_proba(features)[0][1]
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    verdict = "fraud" if proba >= 0.5 else "not_fraud"
    PREDICTIONS_TOTAL.labels(verdict=verdict).inc()

    return FraudVerdict(
        verdict=verdict,
        confidence=round(float(proba if verdict == "fraud" else 1 - proba), 4),
        model_version=str(_pointer["model_version"]),
    )


@app.get("/metrics")
def metrics():
    if os.path.exists(DRIFT_STATE_PATH):
        with open(DRIFT_STATE_PATH) as f:
            state = json.load(f)
        DRIFT_SCORE.set(state.get("drift_score", 0.0))
        DRIFT_THRESHOLD.set(state.get("threshold", 0.0))
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
