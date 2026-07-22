"""FR-8: Prediction Endpoint. Loads the Model Version named by
production_pointer.json (FR-4) — plain file read, no MLflow process
required — and serves fraud verdicts over REST.

Also exposes /metrics (FR-10/FR-11/FR-12): request latency, predicted-class
distribution, the current Drift Score (set by mlops/drift_check.py via
the shared drift_state.json file, since drift checks run on-demand and the
API process is what Prometheus actually scrapes), business-facing
transaction volume/value/card-type/device metrics for the client-facing
dashboard row, and confirmed precision/recall from the delayed
ground-truth loop (mlops/label_feedback.py), read the same on-demand way.

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
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from model.features import load_feature_definitions, transform

POINTER_PATH = os.environ.get("PRODUCTION_POINTER_PATH", "production_pointer.json")
SHADOW_POINTER_PATH = os.environ.get("SHADOW_POINTER_PATH", "models/shadow/shadow_pointer.json")
DRIFT_STATE_PATH = os.environ.get("DRIFT_STATE_PATH", "models/drift_state.json")
LABEL_FEEDBACK_STATE_PATH = os.environ.get("LABEL_FEEDBACK_STATE_PATH", "models/label_feedback_state.json")

app = FastAPI(title="Fraud Model Platform — Inference API")

PREDICTION_LATENCY = Histogram(
    "inference_latency_seconds", "Time to serve a single prediction"
)
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Predictions served, by verdict", ["verdict"]
)
TRANSACTION_AMOUNT = Histogram(
    "transaction_amount",
    "Transaction amount seen at inference time, by verdict",
    ["verdict"],
    buckets=(10, 25, 50, 100, 250, 500, 1000, 5000, 25000),
)
FRAUD_VALUE_TOTAL = Counter(
    "fraud_value_flagged_total", "Cumulative transaction amount flagged as fraud"
)
TRANSACTIONS_BY_CARD_TYPE = Counter(
    "transactions_by_card_type_total", "Transactions by card network and verdict", ["card4", "verdict"]
)
TRANSACTIONS_BY_DEVICE = Counter(
    "transactions_by_device_total", "Transactions by device type and verdict", ["device_type", "verdict"]
)
DRIFT_SCORE = Gauge("drift_score", "Most recent Drift Score computed by mlops/drift_check.py")
DRIFT_THRESHOLD = Gauge("drift_threshold", "Drift Score threshold that triggers retraining")

CONFIRMED_PRECISION = Gauge(
    "confirmed_precision", "Precision computed only from predictions whose true label has been confirmed"
)
CONFIRMED_RECALL = Gauge(
    "confirmed_recall", "Recall computed only from predictions whose true label has been confirmed"
)
LABEL_FEEDBACK_LAG_SECONDS = Gauge(
    "label_feedback_lag_seconds", "Average delay between serving a prediction and its label being confirmed"
)
LABEL_FEEDBACK_PENDING = Gauge(
    "label_feedback_pending", "Predictions served but not yet confirmed against a true label"
)

SHADOW_LOADED = Gauge("shadow_model_loaded", "1 if a shadow candidate is currently loaded, else 0")
SHADOW_PREDICTIONS_TOTAL = Counter(
    "shadow_predictions_total",
    "Shadow model predictions, scored against the same request as production",
    ["verdict", "agreement"],
)


class TransactionPayload(BaseModel):
    TransactionAmt: float
    dist1: float
    C1: float
    C2: float
    D1: float
    ProductCD: str
    card4: str
    card6: str
    P_emaildomain: str
    DeviceType: str


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


def _load_definitions(pointer: dict):
    path = pointer.get("feature_definitions_path")
    return load_feature_definitions(path) if path and os.path.exists(path) else None


# Loaded once at process start. A new deployment (new container, per FR-7)
# is how a new Model Version takes effect — matches the pipeline's
# build-and-deploy model rather than hot-reloading inside a running pod.
_model, _pointer, _definitions = (None, None, None)
_shadow_model, _shadow_pointer, _shadow_definitions = (None, None, None)
# Running tally for the friendly /shadow/status endpoint, in addition to
# the Prometheus counters Grafana reads.
_shadow_tally = {"total": 0, "agree": 0, "disagree": 0}


def _payload_to_row(payload: TransactionPayload) -> dict:
    return payload.model_dump()


@app.on_event("startup")
def _startup():
    global _model, _pointer, _definitions, _shadow_model, _shadow_pointer, _shadow_definitions

    pointer = _load_pointer(POINTER_PATH)
    if pointer is None:
        print(f"[serving] WARNING: no production model registered — {POINTER_PATH} not found.")
    else:
        _pointer = pointer
        _model = _load_model(pointer)
        _definitions = _load_definitions(pointer)
        print(f"[serving] loaded production model version {_pointer['model_version']}")

    shadow_pointer = _load_pointer(SHADOW_POINTER_PATH)
    if shadow_pointer is not None:
        _shadow_pointer = shadow_pointer
        _shadow_model = _load_model(shadow_pointer)
        _shadow_definitions = _load_definitions(shadow_pointer)
        SHADOW_LOADED.set(1)
        print(f"[serving] loaded shadow model version {_shadow_pointer['model_version']}")
    else:
        SHADOW_LOADED.set(0)
        print("[serving] no shadow model loaded")

    _refresh_on_demand_gauges()


def _refresh_on_demand_gauges() -> None:
    """Mirrors on-demand jobs' output files into Prometheus gauges — the
    same pattern for drift (mlops/drift_check.py) and confirmed
    precision/recall (mlops/label_feedback.py): those jobs run separately
    from this process, so the only way their results reach Prometheus is
    this API reading the file they last wrote."""
    if os.path.exists(DRIFT_STATE_PATH):
        with open(DRIFT_STATE_PATH) as f:
            state = json.load(f)
        DRIFT_SCORE.set(state.get("drift_score", 0.0))
        DRIFT_THRESHOLD.set(state.get("threshold", 0.0))

    if os.path.exists(LABEL_FEEDBACK_STATE_PATH):
        with open(LABEL_FEEDBACK_STATE_PATH) as f:
            state = json.load(f)
        if state.get("confirmed_precision") is not None:
            CONFIRMED_PRECISION.set(state["confirmed_precision"])
        if state.get("confirmed_recall") is not None:
            CONFIRMED_RECALL.set(state["confirmed_recall"])
        if state.get("avg_lag_seconds") is not None:
            LABEL_FEEDBACK_LAG_SECONDS.set(state["avg_lag_seconds"])
        LABEL_FEEDBACK_PENDING.set(state.get("pending_count", 0))


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


def _score_shadow(row: dict, production_verdict: str) -> None:
    """Runs after the real response has already been sent — never on the
    critical path of a real caller's request. Transforms with the shadow
    model's OWN feature definitions, not production's — a shadow candidate
    may have been trained with a different feature vocabulary."""
    import pandas as pd

    features = transform(pd.DataFrame([row]), _shadow_definitions)
    proba = _shadow_model.predict_proba(features)[0][1]
    shadow_verdict = "fraud" if proba >= 0.5 else "not_fraud"
    agreement = "match" if shadow_verdict == production_verdict else "differ"

    SHADOW_PREDICTIONS_TOTAL.labels(verdict=shadow_verdict, agreement=agreement).inc()
    _shadow_tally["total"] += 1
    _shadow_tally["agree" if agreement == "match" else "disagree"] += 1


@app.post("/predict", response_model=FraudVerdict)
def predict(payload: TransactionPayload, background_tasks: BackgroundTasks):
    import pandas as pd

    if _model is None:
        raise HTTPException(status_code=503, detail="No model loaded — see /health")

    row = _payload_to_row(payload)
    features = transform(pd.DataFrame([row]), _definitions)

    start = time.perf_counter()
    proba = _model.predict_proba(features)[0][1]
    PREDICTION_LATENCY.observe(time.perf_counter() - start)

    verdict = "fraud" if proba >= 0.5 else "not_fraud"
    PREDICTIONS_TOTAL.labels(verdict=verdict).inc()
    TRANSACTION_AMOUNT.labels(verdict=verdict).observe(payload.TransactionAmt)
    TRANSACTIONS_BY_CARD_TYPE.labels(card4=payload.card4, verdict=verdict).inc()
    TRANSACTIONS_BY_DEVICE.labels(device_type=payload.DeviceType, verdict=verdict).inc()
    if verdict == "fraud":
        FRAUD_VALUE_TOTAL.inc(payload.TransactionAmt)

    if _shadow_model is not None:
        background_tasks.add_task(_score_shadow, row, verdict)

    return FraudVerdict(
        verdict=verdict,
        confidence=round(float(proba if verdict == "fraud" else 1 - proba), 4),
        model_version=str(_pointer["model_version"]),
    )


@app.get("/metrics")
def metrics():
    _refresh_on_demand_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
