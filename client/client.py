"""FR-9: Client Script. Stands in for "Banking Applications" — sends
realistic Transaction Payloads to the inference API and prints the Fraud
Verdict received. Supports a --drifted mode that sources payloads from the
held-out data with model.data.apply_synthetic_drift applied, for
exercising the drift detection loop (FR-13) — the same shift
mlops/drift_check.py --simulate-drift uses, so "drifted" means the same
thing in both places.

Usage:
    python -m client.client --url http://localhost:8000 --count 10
    python -m client.client --url http://localhost:8000 --count 10 --drifted
"""
import argparse
import sys

import pandas as pd
import requests

from model.data import apply_synthetic_drift
from model.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

HOLDOUT_PATH = "models/candidate/holdout.csv"


def _load_batch(count: int, drifted: bool) -> pd.DataFrame:
    try:
        df = pd.read_csv(HOLDOUT_PATH)
    except FileNotFoundError:
        print(
            f"[client] {HOLDOUT_PATH} not found — run `python -m model.train` first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    batch = df.sample(n=min(count, len(df)), random_state=None).reset_index(drop=True)
    if drifted:
        batch = apply_synthetic_drift(batch)
    return batch


def _row_to_payload(row: pd.Series) -> dict:
    # Real IEEE-CIS data has genuine missing values (dist1, DeviceType,
    # etc. are NaN for a large fraction of rows) — `requests` refuses to
    # encode a NaN float into JSON at all, so it has to be sanitized here,
    # not left for the server. Same fallbacks model/features.py's
    # transform() would apply anyway (0.0 for numeric, "missing" for
    # categorical), just applied before the wire instead of after.
    payload = {col: (0.0 if pd.isna(row[col]) else float(row[col])) for col in NUMERIC_COLUMNS}
    payload.update({col: ("missing" if pd.isna(row[col]) else str(row[col])) for col in CATEGORICAL_COLUMNS})
    return payload


def run(url: str, count: int, drifted: bool):
    batch = _load_batch(count, drifted)
    fraud_count = 0
    for _, row in batch.iterrows():
        payload = _row_to_payload(row)
        try:
            resp = requests.post(f"{url}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            verdict = resp.json()
        except requests.RequestException as e:
            print(f"[client] request failed: {e}", file=sys.stderr)
            continue

        if verdict["verdict"] == "fraud":
            fraud_count += 1
        print(
            f"[client] amount=${payload['TransactionAmt']:.2f} card={payload['card4']} device={payload['DeviceType']} "
            f"-> {verdict['verdict']} (confidence={verdict['confidence']}, model_version={verdict['model_version']})"
        )

    print(f"[client] {fraud_count}/{len(batch)} flagged as fraud" + (" (drifted batch)" if drifted else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--drifted", action="store_true")
    args = parser.parse_args()
    run(args.url, args.count, args.drifted)
