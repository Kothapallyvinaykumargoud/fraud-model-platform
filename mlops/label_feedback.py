"""Delayed ground-truth feedback loop.

A real bank doesn't know whether a transaction was actually fraud the
moment it scores it — that only gets confirmed later, when a chargeback
lands (real-world lag: days to weeks). Every metric this platform had
before this file (predictions_total, latency, drift score) only reflects
what the model *claimed*, never whether it was *right* — because nothing
ever played back a true label.

This simulates that lag using the real isFraud labels already sitting in
the held-out slice (never trained on): each run "serves" a few holdout rows
through the current production model, queues each prediction with a
deliberately delayed reveal time (scaled down to minutes here so it's
observable in a demo session, not the real days/weeks), then reveals
whatever's come due and folds it into a running confusion matrix. The
result — confirmed precision/recall, lagged behind real-time verdicts — is
written to a state file serving/app.py's /metrics endpoint reads, the same
pattern already used for drift_score (see DRIFT_STATE_PATH in
serving/app.py).

Usage:
    python -m mlops.label_feedback                  # simulate 20 new + reveal due (default)
    python -m mlops.label_feedback --simulate 50     # queue 50 new predictions
    python -m mlops.label_feedback --reveal-only     # only reveal, queue nothing new
"""
import argparse
import json
import os
import random
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd

from model.data import LABEL_COLUMN
from model.features import load_feature_definitions, transform
from model.register import PRODUCTION_POINTER_PATH

HOLDOUT_PATH = "models/candidate/holdout.csv"
QUEUE_PATH = "models/label_feedback_queue.jsonl"
STATE_PATH = "models/label_feedback_state.json"

# Real bank chargeback lag is days-to-weeks; compressed to minutes so the
# loop is observable by running this script a few times in one session.
MIN_REVEAL_DELAY_SECONDS = 60
MAX_REVEAL_DELAY_SECONDS = 300


def _load_production():
    if not os.path.exists(PRODUCTION_POINTER_PATH):
        raise SystemExit(f"[label_feedback] {PRODUCTION_POINTER_PATH} not found — nothing is registered yet.")
    with open(PRODUCTION_POINTER_PATH) as f:
        pointer = json.load(f)
    model = joblib.load(pointer["artifact_path"])
    definitions = load_feature_definitions(pointer["feature_definitions_path"])
    return model, definitions


def _queue_new_predictions(n: int) -> int:
    if not os.path.exists(HOLDOUT_PATH):
        raise SystemExit(f"[label_feedback] {HOLDOUT_PATH} not found — run `python -m model.train` first.")

    model, definitions = _load_production()
    holdout = pd.read_csv(HOLDOUT_PATH)
    batch = holdout.sample(n=min(n, len(holdout)), random_state=None).reset_index(drop=True)

    X = transform(batch, definitions)
    proba = model.predict_proba(X)[:, 1]

    now = datetime.now(timezone.utc)
    os.makedirs(os.path.dirname(QUEUE_PATH), exist_ok=True)
    with open(QUEUE_PATH, "a") as f:
        for i in range(len(batch)):
            delay = random.uniform(MIN_REVEAL_DELAY_SECONDS, MAX_REVEAL_DELAY_SECONDS)
            entry = {
                "predicted_verdict": "fraud" if proba[i] >= 0.5 else "not_fraud",
                "true_label": "fraud" if int(batch.loc[i, LABEL_COLUMN]) == 1 else "not_fraud",
                "served_at": now.isoformat(),
                "reveal_at": (now + timedelta(seconds=delay)).isoformat(),
            }
            f.write(json.dumps(entry) + "\n")
    return len(batch)


def _load_queue() -> list[dict]:
    if not os.path.exists(QUEUE_PATH):
        return []
    with open(QUEUE_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {"confirmed_tp": 0, "confirmed_fp": 0, "confirmed_fn": 0, "confirmed_tn": 0, "lag_seconds_sum": 0.0}
    with open(STATE_PATH) as f:
        state = json.load(f)
    for key in ("confirmed_tp", "confirmed_fp", "confirmed_fn", "confirmed_tn", "lag_seconds_sum"):
        state.setdefault(key, 0 if key != "lag_seconds_sum" else 0.0)
    return state


def _reveal_due(now: datetime) -> dict:
    queue = _load_queue()
    due = [e for e in queue if datetime.fromisoformat(e["reveal_at"]) <= now]
    pending = [e for e in queue if datetime.fromisoformat(e["reveal_at"]) > now]

    state = _load_state()
    for entry in due:
        predicted_fraud = entry["predicted_verdict"] == "fraud"
        actually_fraud = entry["true_label"] == "fraud"
        if predicted_fraud and actually_fraud:
            state["confirmed_tp"] += 1
        elif predicted_fraud and not actually_fraud:
            state["confirmed_fp"] += 1
        elif not predicted_fraud and actually_fraud:
            state["confirmed_fn"] += 1
        else:
            state["confirmed_tn"] += 1
        lag = (now - datetime.fromisoformat(entry["served_at"])).total_seconds()
        state["lag_seconds_sum"] += lag

    total_confirmed = state["confirmed_tp"] + state["confirmed_fp"] + state["confirmed_fn"] + state["confirmed_tn"]
    tp, fp, fn = state["confirmed_tp"], state["confirmed_fp"], state["confirmed_fn"]
    state["confirmed_precision"] = round(tp / (tp + fp), 4) if (tp + fp) else None
    state["confirmed_recall"] = round(tp / (tp + fn), 4) if (tp + fn) else None
    state["avg_lag_seconds"] = round(state["lag_seconds_sum"] / total_confirmed, 2) if total_confirmed else None
    state["total_confirmed"] = total_confirmed
    state["pending_count"] = len(pending)
    state["updated_at"] = now.isoformat()

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    with open(QUEUE_PATH, "w") as f:
        for entry in pending:
            f.write(json.dumps(entry) + "\n")

    return {"revealed": len(due), "state": state}


def run(simulate: int, reveal_only: bool) -> dict:
    queued = 0
    if not reveal_only:
        queued = _queue_new_predictions(simulate)
        print(f"[label_feedback] queued {queued} new prediction(s) awaiting label confirmation")

    result = _reveal_due(datetime.now(timezone.utc))
    print(
        f"[label_feedback] revealed {result['revealed']} — "
        f"confirmed_precision={result['state']['confirmed_precision']} "
        f"confirmed_recall={result['state']['confirmed_recall']} "
        f"pending={result['state']['pending_count']}"
    )
    return {"queued": queued, **result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", type=int, default=20, help="New predictions to queue this run")
    parser.add_argument("--reveal-only", action="store_true", help="Skip queuing new predictions, only reveal due ones")
    args = parser.parse_args()
    run(args.simulate, args.reveal_only)
