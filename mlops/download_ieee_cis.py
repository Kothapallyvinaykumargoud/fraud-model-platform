"""Downloads the IEEE-CIS Fraud Detection dataset from Kaggle, merges the
transaction and identity tables, stratified-samples it down to a small
slice (default 50,000 rows, real ~3.5% fraud rate preserved rather than
oversampled) with the curated columns model/features.py expects, and
writes data/ieee_cis_sample.csv.

Downloads only train_transaction.csv + train_identity.csv (~710MB total),
not the full competition bundle (~1.4GB, which also includes the unused
test set) — competition_download_files() pulls everything as one zip,
competition_download_file() lets us fetch just the two files this project
actually needs.

IEEE-CIS is a Kaggle *competition* dataset, not a plain open dataset:
downloading it needs (a) a Kaggle account with an API token, supplied as
the KAGGLE_API_TOKEN environment variable (from kaggle.com/settings/api —
Kaggle's current single-token flow; the older KAGGLE_USERNAME+KAGGLE_KEY
pair still works too if you have one), and (b) that account having already
accepted the competition rules at kaggle.com/competitions/
ieee-fraud-detection/rules. Kaggle enforces (b) at the download endpoint
specifically — authenticating and even listing the competition's files can
succeed with (a) alone; only the actual download 403s without (b), and
there's no way to script around it. Without either, run() raises with that
explanation rather than silently falling back to anything — same "fail
loudly" stance as model/data.py takes toward a missing real dataset, just
one step earlier in the chain.

Usage:
    KAGGLE_API_TOKEN=... python -m mlops.download_ieee_cis
    python -m mlops.download_ieee_cis --sample-size 50000
"""
import argparse
import os
import zipfile

import pandas as pd

from model.data import LABEL_COLUMN
from model.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

COMPETITION = "ieee-fraud-detection"
RAW_DIR = "data/.ieee_cis_raw"
OUTPUT_PATH = "data/ieee_cis_sample.csv"
RAW_FILES = ("train_transaction.csv", "train_identity.csv")


def _download_raw() -> None:
    # Deferred import: the `kaggle` package authenticates at import time
    # (exits immediately if no credentials are found), so importing it
    # eagerly at module load would break every other script that happens
    # to import something from mlops/ or model/, even ones that never
    # touch Kaggle.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # exits with Kaggle's own setup instructions if no valid credential is found

    os.makedirs(RAW_DIR, exist_ok=True)
    for filename in RAW_FILES:
        zip_path = os.path.join(RAW_DIR, f"{filename}.zip")
        try:
            api.competition_download_file(COMPETITION, filename, path=RAW_DIR, quiet=False)
        except Exception as e:
            raise SystemExit(
                f"[download_ieee_cis] download of {filename} failed — most likely your Kaggle "
                f"account hasn't accepted the competition rules yet. Visit "
                f"https://www.kaggle.com/competitions/{COMPETITION}/rules, click \"I understand "
                f"and accept\", then retry. Original error: {e}"
            )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        os.remove(zip_path)  # don't keep both the zip and the extracted CSV — disk is tight


def _merge_and_sample(sample_size: int, seed: int) -> pd.DataFrame:
    transaction = pd.read_csv(os.path.join(RAW_DIR, "train_transaction.csv"))
    identity = pd.read_csv(os.path.join(RAW_DIR, "train_identity.csv"))
    merged = transaction.merge(identity, on="TransactionID", how="left")

    keep = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [LABEL_COLUMN]
    merged = merged[keep]

    # Stratified, not oversampled: preserve the real (~3.5%) fraud rate so
    # validate()'s thresholds are tested against a realistic class balance
    # rather than an inflated one.
    fraud = merged[merged[LABEL_COLUMN] == 1]
    legit = merged[merged[LABEL_COLUMN] == 0]
    fraud_rate = len(fraud) / len(merged)
    n_fraud = max(1, int(sample_size * fraud_rate))
    n_legit = sample_size - n_fraud

    sample = pd.concat(
        [
            fraud.sample(n=min(n_fraud, len(fraud)), random_state=seed),
            legit.sample(n=min(n_legit, len(legit)), random_state=seed),
        ]
    ).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sample


def run(sample_size: int = 50000, seed: int = 42) -> str:
    if not os.path.exists(os.path.join(RAW_DIR, "train_transaction.csv")):
        _download_raw()

    sample = _merge_and_sample(sample_size, seed)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)
    print(
        f"[download_ieee_cis] wrote {len(sample)} rows to {OUTPUT_PATH} "
        f"(fraud rate {sample[LABEL_COLUMN].mean():.4f})"
    )
    return OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.sample_size, args.seed)
