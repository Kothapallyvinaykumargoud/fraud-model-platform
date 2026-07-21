"""FR-2: Model Packaging. Bundles a validated model artifact with versioned
metadata (training date, dataset slice info, validation metrics) so it can
be traced later without re-running validation (FR-2 consequence).

Usage:
    python -m model.package --candidate models/candidate
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone

PACKAGES_DIR = "models/packages"


def package(candidate_dir: str) -> str:
    with open(os.path.join(candidate_dir, "metadata.json")) as f:
        metadata = json.load(f)

    model_path = os.path.join(candidate_dir, "model.joblib")
    with open(model_path, "rb") as f:
        content_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    package_id = f"pkg-{content_hash}"
    package_dir = os.path.join(PACKAGES_DIR, package_id)
    os.makedirs(package_dir, exist_ok=True)

    shutil.copy2(model_path, os.path.join(package_dir, "model.joblib"))

    package_metadata = {
        **metadata,
        "package_id": package_id,
        "content_hash": content_hash,
        "packaged_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(package_dir, "metadata.json"), "w") as f:
        json.dump(package_metadata, f, indent=2)

    print(f"[package] {candidate_dir} -> {package_dir} (id={package_id})")
    return package_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="models/candidate")
    args = parser.parse_args()
    package(args.candidate)
