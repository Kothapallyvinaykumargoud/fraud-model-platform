"""FR-8: Prediction Endpoint. Loads the Model Version named by
production_pointer.json (FR-4) — plain file read, no MLflow process
required — and serves fraud verdicts over REST.

Also exposes /metrics (FR-10/FR-11/FR-12): request latency, predicted-class
distribution, and the current Drift Score (set by mlops/drift_check.py via
the shared drift_state.json file, since drift checks run on-demand and the
API process is what Prometheus actually scrapes).

Shadow deployment: if models/shadow/shadow_pointer.json exists, every real
request is ALSO scored by the shadow model, in the background, after the
real response has already been sent — shadow scoring never adds latency to
a real caller, and its verdict is never returned to them. Only
mlops/promote_shadow.py can make a shadow model start actually serving
real traffic (see that file's docstring for why this is manual).
"""
import json
import os
import time

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from model.data import FEATURE_COLUMNS

POINTER_PATH = os.environ.get("PRODUCTION_POINTER_PATH", "production_pointer.json")
SHADOW_POINTER_PATH = os.environ.get("SHADOW_POINTER_PATH", "models/shadow/shadow_pointer.json")
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

SHADOW_LOADED = Gauge("shadow_model_loaded", "1 if a shadow candidate is currently loaded, else 0")
SHADOW_PREDICTIONS_TOTAL = Counter(
    "shadow_predictions_total",
    "Shadow model predictions, scored against the same request as production",
    ["verdict", "agreement"],
)


class TransactionPayload(BaseModel):
    Time: float
    Amount: float
    V: list[float] = Field(..., min_length=28, max_length=28, description="V1..V28 PCA features, in order")


class FraudVerdict(BaseModel):
    model_config = {"protected_namespaces": ()}

    verdict: str
    confidence: float
    model_version: str


def _load_pointer(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_model(pointer: dict):
    return joblib.load(pointer["artifact_path"])


# Loaded once at process start. A new deployment (new container, per FR-7)
# is how a new Model Version takes effect — matches the pipeline's
# build-and-deploy model rather than hot-reloading inside a running pod.
_model, _pointer = (None, None)
_shadow_model, _shadow_pointer = (None, None)
# Running tally for the friendly /shadow/status endpoint, in addition to
# the Prometheus counters Grafana reads.
_shadow_tally = {"total": 0, "agree": 0, "disagree": 0}


@app.on_event("startup")
def _startup():
    global _model, _pointer, _shadow_model, _shadow_pointer

    pointer = _load_pointer(POINTER_PATH)
    if pointer is None:
        print(f"[serving] WARNING: no production model registered — {POINTER_PATH} not found.")
    else:
        _pointer = pointer
        _model = _load_model(pointer)
        print(f"[serving] loaded production model version {_pointer['model_version']}")

    shadow_pointer = _load_pointer(SHADOW_POINTER_PATH)
    if shadow_pointer is not None:
        _shadow_pointer = shadow_pointer
        _shadow_model = _load_model(shadow_pointer)
        SHADOW_LOADED.set(1)
        print(f"[serving] loaded shadow model version {_shadow_pointer['model_version']}")
    else:
        SHADOW_LOADED.set(0)
        print("[serving] no shadow model loaded")

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


@app.get("/shadow/status")
def shadow_status():
    if _shadow_pointer is None:
        return {"shadow_loaded": False}
    total = _shadow_tally["total"]
    agreement_rate = round(_shadow_tally["agree"] / total, 4) if total else None
    return {
        "shadow_loaded": True,
        "shadow_model_version": _shadow_pointer["model_version"],
        "production_model_version": _pointer["model_version"] if _pointer else None,
        "requests_scored": total,
        "agreement_rate": agreement_rate,
    }


def _score_shadow(features: pd.DataFrame, production_verdict: str) -> None:
    """Runs after the real response has already been sent — never on the
    critical path of a real caller's request."""
    proba = _shadow_model.predict_proba(features)[0][1]
    shadow_verdict = "fraud" if proba >= 0.5 else "not_fraud"
    agreement = "match" if shadow_verdict == production_verdict else "differ"

    SHADOW_PREDICTIONS_TOTAL.labels(verdict=shadow_verdict, agreement=agreement).inc()
    _shadow_tally["total"] += 1
    _shadow_tally["agree" if agreement == "match" else "disagree"] += 1


@app.post("/predict", response_model=FraudVerdict)
def predict(payload: TransactionPayload, background_tasks: BackgroundTasks):
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

    if _shadow_model is not None:
        background_tasks.add_task(_score_shadow, features, verdict)

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
