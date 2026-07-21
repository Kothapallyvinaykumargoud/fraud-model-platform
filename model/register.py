"""FR-3: Model Registration. FR-4: Production Pointer.

Registers a packaged model as a new Model Version in MLflow, then writes
production_pointer.json — the file the serving path reads to know which
Model Version is current, without needing MLflow running (FR-4).

MLflow here uses a local file-based tracking store (`file:./mlruns`), not a
tracking *server* — there's no persistent MLflow process to start or stop.
"On-demand" (per the brief/PRD) means: this script does file I/O when run,
and nothing runs between invocations. That's what satisfies the RAM budget.

Usage:
    python -m model.register --package models/packages/pkg-XXXXXXXXXXXX
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn

MLFLOW_TRACKING_URI = "file:./mlruns"
REGISTERED_MODEL_NAME = "fraud-detector"
PRODUCTION_POINTER_PATH = "production_pointer.json"
PRODUCTION_MODEL_DIR = "models/production"
REGISTRATION_LOG_PATH = "models/registration_log.jsonl"


def _log_registration(pointer: dict) -> None:
    os.makedirs(os.path.dirname(REGISTRATION_LOG_PATH), exist_ok=True)
    with open(REGISTRATION_LOG_PATH, "a") as f:
        f.write(json.dumps(pointer) + "\n")


def register(package_dir: str) -> dict:
    with open(os.path.join(package_dir, "metadata.json")) as f:
        metadata = json.load(f)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = joblib.load(os.path.join(package_dir, "model.joblib"))

    with mlflow.start_run(run_name=metadata["package_id"]) as run:
        mlflow.log_params(
            {
                "model_type": metadata["model_type"],
                "data_source": metadata["data_source"],
                "content_hash": metadata["content_hash"],
            }
        )
        mlflow.log_metrics(metadata["metrics"])
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    client = mlflow.MlflowClient()
    # log_model + registered_model_name already created the version; fetch
    # its number from the run it was just logged under.
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise RuntimeError("Model registration did not produce a version — check MLflow logs.")
    model_version = versions[0].version

    # MLflow's file-based store writes *absolute* paths into its own
    # metadata, which breaks once mlruns/ is copied somewhere else (e.g.
    # into a Docker image). So the serving contract is a plain copied file,
    # not an mlflow:// resolution — MLflow here is for history/versioning,
    # not for runtime artifact loading.
    os.makedirs(PRODUCTION_MODEL_DIR, exist_ok=True)
    artifact_path = os.path.join(PRODUCTION_MODEL_DIR, "model.joblib")
    shutil.copy2(os.path.join(package_dir, "model.joblib"), artifact_path)

    pointer = {
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "run_id": run_id,
        "model_uri": f"models:/{REGISTERED_MODEL_NAME}/{model_version}",
        "artifact_path": artifact_path,
        "package_dir": package_dir,
        "package_id": metadata["package_id"],
        "metrics": metadata["metrics"],
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(PRODUCTION_POINTER_PATH, "w") as f:
        json.dump(pointer, f, indent=2)
    _log_registration(pointer)

    print(f"[register] registered {REGISTERED_MODEL_NAME} v{model_version} (run {run_id})")
    print(f"[register] wrote {PRODUCTION_POINTER_PATH} -> version {model_version} @ {artifact_path}")
    return pointer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    register(args.package)
