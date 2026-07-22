"""FR-3: Model Registration. FR-4: Production Pointer.

Registers a packaged model as a new Model Version in MLflow, then writes
production_pointer.json — the file the serving path reads to know which
Model Version is current, without needing MLflow running (FR-4).

MLflow here uses a local file-based tracking store (`file:./mlruns`), not a
tracking *server* — there's no persistent MLflow process to start or stop.
"On-demand" (per the brief/PRD) means: this script does file I/O when run,
and nothing runs between invocations. That's what satisfies the RAM budget.

Also supports shadow registration: a candidate can be registered into
MLflow (real version number, real audit trail) without becoming the
production pointer — see register_shadow() and mlops/promote_shadow.py.

Usage:
    python -m model.register --package models/packages/pkg-XXXXXXXXXXXX
    python -m model.register --package models/packages/pkg-XXXXXXXXXXXX --shadow
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

SHADOW_POINTER_PATH = "models/shadow/shadow_pointer.json"
SHADOW_MODEL_DIR = "models/shadow"

REGISTRATION_LOG_PATH = "models/registration_log.jsonl"


def _copy_feature_definitions(package_dir: str, dest_dir: str) -> str | None:
    """Feature definitions travel with the model artifact through every
    slot (production, shadow, and — see mlops/promote_shadow.py and
    mlops/rollback.py — a shadow promotion or rollback), the same way
    model.joblib does, so serving always transforms with the exact
    vocabulary this model version was trained on."""
    src = os.path.join(package_dir, "feature_definitions.json")
    if not os.path.exists(src):
        return None
    dest = os.path.join(dest_dir, "feature_definitions.json")
    shutil.copy2(src, dest)
    return dest


def _log_registration(pointer: dict) -> None:
    os.makedirs(os.path.dirname(REGISTRATION_LOG_PATH), exist_ok=True)
    with open(REGISTRATION_LOG_PATH, "a") as f:
        f.write(json.dumps(pointer) + "\n")


def _register_in_mlflow(package_dir: str) -> tuple[dict, str, str]:
    """Shared MLflow registration step for both production and shadow
    paths — every candidate gets a real Model Version and audit trail
    regardless of whether it ends up serving real traffic."""
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
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise RuntimeError("Model registration did not produce a version — check MLflow logs.")
    model_version = versions[0].version
    return metadata, run_id, model_version


def register(package_dir: str) -> dict:
    metadata, run_id, model_version = _register_in_mlflow(package_dir)

    # MLflow's file-based store writes *absolute* paths into its own
    # metadata, which breaks once mlruns/ is copied somewhere else (e.g.
    # into a Docker image). So the serving contract is a plain copied file,
    # not an mlflow:// resolution — MLflow here is for history/versioning,
    # not for runtime artifact loading.
    os.makedirs(PRODUCTION_MODEL_DIR, exist_ok=True)
    artifact_path = os.path.join(PRODUCTION_MODEL_DIR, "model.joblib")
    shutil.copy2(os.path.join(package_dir, "model.joblib"), artifact_path)
    feature_definitions_path = _copy_feature_definitions(package_dir, PRODUCTION_MODEL_DIR)

    pointer = {
        "status": "production",
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "run_id": run_id,
        "model_uri": f"models:/{REGISTERED_MODEL_NAME}/{model_version}",
        "artifact_path": artifact_path,
        "feature_definitions_path": feature_definitions_path,
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


def register_shadow(package_dir: str) -> dict:
    """Registers a candidate in MLflow like register(), but writes it to
    the shadow slot instead of production — real traffic keeps going to
    whatever's in production_pointer.json until a human explicitly runs
    mlops/promote_shadow.py."""
    metadata, run_id, model_version = _register_in_mlflow(package_dir)

    os.makedirs(SHADOW_MODEL_DIR, exist_ok=True)
    artifact_path = os.path.join(SHADOW_MODEL_DIR, "model.joblib")
    shutil.copy2(os.path.join(package_dir, "model.joblib"), artifact_path)
    feature_definitions_path = _copy_feature_definitions(package_dir, SHADOW_MODEL_DIR)

    pointer = {
        "status": "shadow",
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": model_version,
        "run_id": run_id,
        "model_uri": f"models:/{REGISTERED_MODEL_NAME}/{model_version}",
        "artifact_path": artifact_path,
        "feature_definitions_path": feature_definitions_path,
        "package_dir": package_dir,
        "package_id": metadata["package_id"],
        "metrics": metadata["metrics"],
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(SHADOW_POINTER_PATH, "w") as f:
        json.dump(pointer, f, indent=2)
    _log_registration(pointer)

    print(f"[register] registered {REGISTERED_MODEL_NAME} v{model_version} (run {run_id}) as SHADOW")
    print(f"[register] wrote {SHADOW_POINTER_PATH} -> version {model_version} @ {artifact_path}")
    print("[register] production traffic is UNCHANGED — run mlops/promote_shadow.py to promote")
    return pointer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--shadow", action="store_true", help="Register as shadow instead of production")
    args = parser.parse_args()
    register_shadow(args.package) if args.shadow else register(args.package)
