"""FR-9: Client Script. Stands in for "Banking Applications" — sends
realistic Transaction Payloads to the inference API and prints the Fraud
Verdict received. Supports a --drifted mode that sources payloads from the
held-out data with a synthetic shift applied, for exercising the drift
detection loop (FR-13).

Usage:
    python -m client.client --url http://localhost:8000 --count 10
    python -m client.client --url http://localhost:8000 --count 10 --drifted
"""
import argparse
import sys

import pandas as pd
import requests

from model.data import FEATURE_COLUMNS

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
        # Mirrors mlops/drift_check.py's synthetic shift so the client can
        # reproduce a drift-triggering batch on demand.
        for col in [c for c in FEATURE_COLUMNS if c.startswith("V")]:
            batch[col] = batch[col] + 3.0
        batch["Amount"] = batch["Amount"] * 2.5
    return batch


def run(url: str, count: int, drifted: bool):
    batch = _load_batch(count, drifted)
    fraud_count = 0
    for _, row in batch.iterrows():
        payload = {
            "Time": float(row["Time"]),
            "Amount": float(row["Amount"]),
            "V": [float(row[f"V{i}"]) for i in range(1, 29)],
        }
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
            f"[client] amount=${row['Amount']:.2f} -> {verdict['verdict']} "
            f"(confidence={verdict['confidence']}, model_version={verdict['model_version']})"
        )

    print(f"[client] {fraud_count}/{len(batch)} flagged as fraud" + (" (drifted batch)" if drifted else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--drifted", action="store_true")
    args = parser.parse_args()
    run(args.url, args.count, args.drifted)
