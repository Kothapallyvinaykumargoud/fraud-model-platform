"""FR-14: Drift-Triggered Retraining. FR-15: Auto-Retrain Promotion Gate.

Runs the Drift Check; if triggered, retrains a candidate and only lets it
replace the Production Pointer if it (a) passes the Validation Gate (FR-1)
and (b) matches or exceeds the current production Model Version's held-out
metric (PRD §4.6 FR-15 resolved assumption — a retrain that clears the bar
but is *worse* than what's running does not get promoted).

Designed to run on-demand on the EC2 box (brief/PRD architecture decision),
sharing it with MLflow's on-demand registration rather than running
continuously.

Usage:
    python -m mlops.retrain                  # only acts if drift is detected
    python -m mlops.retrain --simulate-drift  # force a drift-triggering check first
    python -m mlops.retrain --force           # skip the drift check, retrain now
"""
import argparse
import json
import os

from model.package import package
from model.register import register
from model.train import train
from model.validate import validate
from mlops.drift_check import run as check_drift

PRODUCTION_POINTER_PATH = "production_pointer.json"
PROMOTION_METRIC = "roc_auc"


def _current_production_metric() -> float:
    if not os.path.exists(PRODUCTION_POINTER_PATH):
        return 0.0  # nothing deployed yet — any validated candidate can become production
    with open(PRODUCTION_POINTER_PATH) as f:
        pointer = json.load(f)
    return pointer.get("metrics", {}).get(PROMOTION_METRIC, 0.0)


def run(threshold: float, simulate_drift: bool, force: bool) -> dict:
    if not force:
        drift_state = check_drift(threshold=threshold, simulate_drift=simulate_drift)
        if not drift_state["triggered"]:
            print("[retrain] drift below threshold — no retraining triggered")
            return {"retrained": False, "reason": "drift below threshold", "drift_state": drift_state}
        print(f"[retrain] drift triggered (score={drift_state['drift_score']}) — retraining")
    else:
        print("[retrain] --force — retraining without a drift check")

    candidate_dir = "models/candidate_retrain"
    metadata = train(data_path="data/creditcard.csv", out_dir=candidate_dir)

    passed, validation_record = validate(candidate_dir)
    if not passed:
        print(f"[retrain] candidate FAILED validation gate: {validation_record['failures']} — not promoted")
        return {"retrained": True, "promoted": False, "reason": "failed validation gate", "validation": validation_record}

    candidate_metric = metadata["metrics"][PROMOTION_METRIC]
    current_metric = _current_production_metric()
    if candidate_metric < current_metric:
        print(
            f"[retrain] candidate passed validation but {PROMOTION_METRIC}={candidate_metric:.4f} "
            f"< current production {current_metric:.4f} — not promoted (FR-15 regression check)"
        )
        return {
            "retrained": True,
            "promoted": False,
            "reason": "regression vs current production",
            "candidate_metric": candidate_metric,
            "current_metric": current_metric,
        }

    package_dir = package(candidate_dir)
    pointer = register(package_dir)
    print(f"[retrain] candidate PROMOTED — production is now version {pointer['model_version']}")
    return {"retrained": True, "promoted": True, "pointer": pointer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--simulate-drift", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.threshold, args.simulate_drift, args.force)
