"""Promotes the current shadow candidate to production. This is the
deliberate human-in-the-loop step: mlops/retrain.py never calls this
itself — a person reviews the shadow's live agreement rate with
production (Grafana "Shadow vs Production Agreement" panel, or
/shadow/status on the running API) and decides.

Usage:
    python -m mlops.promote_shadow          # promote whatever's in the shadow slot
    python -m mlops.promote_shadow --clear   # after promoting, clear the shadow slot
"""
import argparse
import json
import os
import shutil
from datetime import datetime, timezone

from model.register import (
    PRODUCTION_MODEL_DIR,
    PRODUCTION_POINTER_PATH,
    SHADOW_MODEL_DIR,
    SHADOW_POINTER_PATH,
    _log_registration,
)


def promote(clear_shadow: bool = False) -> dict:
    if not os.path.exists(SHADOW_POINTER_PATH):
        raise SystemExit(f"[promote_shadow] {SHADOW_POINTER_PATH} not found — nothing to promote.")

    with open(SHADOW_POINTER_PATH) as f:
        shadow_pointer = json.load(f)

    os.makedirs(PRODUCTION_MODEL_DIR, exist_ok=True)
    production_artifact = os.path.join(PRODUCTION_MODEL_DIR, "model.joblib")
    shutil.copy2(shadow_pointer["artifact_path"], production_artifact)

    pointer = {
        **shadow_pointer,
        "status": "production",
        "artifact_path": production_artifact,
        "promoted_from_shadow_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(PRODUCTION_POINTER_PATH, "w") as f:
        json.dump(pointer, f, indent=2)
    _log_registration(pointer)

    print(f"[promote_shadow] version {pointer['model_version']} is now PRODUCTION")

    if clear_shadow:
        os.remove(SHADOW_POINTER_PATH)
        if os.path.exists(os.path.join(SHADOW_MODEL_DIR, "model.joblib")):
            os.remove(os.path.join(SHADOW_MODEL_DIR, "model.joblib"))
        print("[promote_shadow] shadow slot cleared")

    print("[promote_shadow] commit + push production_pointer.json and models/production/ to redeploy")
    return pointer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Clear the shadow slot after promoting")
    args = parser.parse_args()
    promote(clear_shadow=args.clear)
