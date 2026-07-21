"""FR-16: Version-Aware Rollback. Points production_pointer.json at a
specific prior Model Version and restores its artifact to
models/production/model.joblib — a real rollback of *which model answers
requests*, not just a container redeploy.

Looks up the requested version in models/registration_log.jsonl (appended
by model.register on every registration) to find that version's packaged
artifact, which model.package never deletes.

After running this, commit and push production_pointer.json + the restored
models/production/model.joblib — the build-and-deploy CI workflow (FR-6/7)
picks up the push and redeploys the rolled-back version, same as any other
change (PRD §4.7 FR-16 consequence: the API's reported version reflects the
rollback).

Usage:
    python -m mlops.rollback --version 2
    python -m mlops.rollback --list
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone

REGISTRATION_LOG_PATH = "models/registration_log.jsonl"
PRODUCTION_POINTER_PATH = "production_pointer.json"
PRODUCTION_MODEL_DIR = "models/production"


def _load_log() -> list[dict]:
    if not os.path.exists(REGISTRATION_LOG_PATH):
        raise SystemExit(f"[rollback] {REGISTRATION_LOG_PATH} not found — no registrations to roll back to.")
    with open(REGISTRATION_LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def list_versions():
    for entry in _load_log():
        print(
            f"v{entry['model_version']}  registered_at={entry['registered_at']}  "
            f"roc_auc={entry['metrics'].get('roc_auc')}  package_id={entry['package_id']}"
        )


def rollback(target_version: int) -> dict:
    log = _load_log()
    matches = [e for e in log if int(e["model_version"]) == int(target_version)]
    if not matches:
        available = sorted({int(e["model_version"]) for e in log})
        raise SystemExit(f"[rollback] version {target_version} not found in {REGISTRATION_LOG_PATH}. Available: {available}")

    entry = matches[-1]  # most recent registration record for that version
    package_dir = entry["package_dir"]
    source_artifact = os.path.join(package_dir, "model.joblib")
    if not os.path.exists(source_artifact):
        raise SystemExit(f"[rollback] {source_artifact} no longer exists — cannot restore version {target_version}.")

    os.makedirs(PRODUCTION_MODEL_DIR, exist_ok=True)
    restored_artifact = os.path.join(PRODUCTION_MODEL_DIR, "model.joblib")
    shutil.copy2(source_artifact, restored_artifact)

    pointer = {
        **entry,
        "artifact_path": restored_artifact,
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
        "rollback_from_log_entry": entry["registered_at"],
    }
    with open(PRODUCTION_POINTER_PATH, "w") as f:
        json.dump(pointer, f, indent=2)

    print(f"[rollback] production_pointer.json now points at version {target_version} (restored from {package_dir})")
    print("[rollback] commit + push to trigger redeploy via CI")
    return pointer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", type=int, help="Model Version to roll back to")
    group.add_argument("--list", action="store_true", help="List all known registered versions")
    args = parser.parse_args()

    if args.list:
        list_versions()
    else:
        rollback(args.version)
