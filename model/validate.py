"""FR-1: Validation Gate. A candidate model must clear a fixed metric
threshold before it can be packaged/registered. Every attempt — pass or
fail — is logged (PRD FR-1 consequence: "Every validation attempt is
logged with its metrics").

Usage:
    python -m model.validate --candidate models/candidate
"""
import argparse
import json
import os
from datetime import datetime, timezone

# Fixed thresholds. Deliberately lenient for a solo/synthetic-data project —
# tighten once trained on the real Kaggle dataset.
THRESHOLDS = {
    "roc_auc": 0.80,
    "recall": 0.50,
}

VALIDATION_LOG_PATH = "models/validation_log.jsonl"


def validate(candidate_dir: str, thresholds: dict = None) -> tuple[bool, dict]:
    thresholds = thresholds or THRESHOLDS
    with open(os.path.join(candidate_dir, "metadata.json")) as f:
        metadata = json.load(f)
    metrics = metadata["metrics"]

    failures = [
        f"{name}={metrics[name]:.4f} < required {threshold}"
        for name, threshold in thresholds.items()
        if metrics.get(name, 0) < threshold
    ]
    passed = len(failures) == 0

    record = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": candidate_dir,
        "trained_at": metadata.get("trained_at"),
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": passed,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(VALIDATION_LOG_PATH), exist_ok=True)
    with open(VALIDATION_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    if passed:
        print(f"[validate] PASSED — {candidate_dir} clears all thresholds")
    else:
        print(f"[validate] FAILED — {candidate_dir}: {failures}")

    return passed, record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="models/candidate")
    args = parser.parse_args()
    passed, _ = validate(args.candidate)
    raise SystemExit(0 if passed else 1)
