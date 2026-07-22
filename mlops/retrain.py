"""FR-14: Drift-Triggered Retraining. FR-15: Auto-Retrain Promotion Gate.

Runs the Drift Check; if triggered, retrains a candidate and — if it (a)
passes the Validation Gate (FR-1) and (b) matches or exceeds the current
production Model Version's held-out metric (PRD §4.6 FR-15) — registers it
as a SHADOW candidate, not production. It does not touch real traffic.

This is deliberately NOT full auto-promotion: a retrain that clears both
gates still only earns a spot serving shadow traffic (scored on every real
request, logged, never returned to the caller — see serving/app.py). A
human reviews the shadow's live agreement rate with production and runs
mlops/promote_shadow.py to actually put it in front of real users. This is
the human-in-the-loop step a fully automatic pipeline was missing.

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

from model.data import DEFAULT_CSV_PATH
from model.package import package
from model.register import register_shadow
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
    metadata = train(data_path=DEFAULT_CSV_PATH, out_dir=candidate_dir)

    passed, validation_record = validate(candidate_dir)
    if not passed:
        print(f"[retrain] candidate FAILED validation gate: {validation_record['failures']} — discarded")
        return {"retrained": True, "shadowed": False, "reason": "failed validation gate", "validation": validation_record}

    candidate_metric = metadata["metrics"][PROMOTION_METRIC]
    current_metric = _current_production_metric()
    # Tolerance, not a strict inequality: RandomForestClassifier(n_jobs=-1)
    # doesn't guarantee bit-identical metrics across separate runs even with
    # the same random_state (parallel tree-building order varies), so two
    # independently-trained models on identical data can differ by ~1e-6
    # purely from floating-point noise. A strict "<" would reject that as a
    # false regression essentially every time. Only treat it as a real
    # regression if it exceeds this tolerance.
    REGRESSION_TOLERANCE = 0.005
    if candidate_metric < current_metric - REGRESSION_TOLERANCE:
        print(
            f"[retrain] candidate passed validation but {PROMOTION_METRIC}={candidate_metric:.4f} "
            f"< current production {current_metric:.4f} — not worth shadowing (FR-15 regression check)"
        )
        return {
            "retrained": True,
            "shadowed": False,
            "reason": "regression vs current production",
            "candidate_metric": candidate_metric,
            "current_metric": current_metric,
        }

    package_dir = package(candidate_dir)
    pointer = register_shadow(package_dir)
    print(f"[retrain] candidate registered as SHADOW — version {pointer['model_version']}")
    print("[retrain] production traffic is UNCHANGED. Review shadow agreement, then run mlops/promote_shadow.py")
    return {"retrained": True, "shadowed": True, "pointer": pointer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--simulate-drift", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.threshold, args.simulate_drift, args.force)
