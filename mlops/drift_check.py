"""FR-13: Drift Check. Compares an incoming data batch against the training
baseline (the holdout slice, which was drawn from the same original
distribution as training but never trained on) and computes a Drift Score.

Method (resolves PRD §8 Open Question 1): Population Stability Index (PSI),
averaged across features — a standard, well-understood drift statistic.
PSI < 0.1: no meaningful shift. 0.1-0.25: moderate. > 0.25: significant.
Default threshold: 0.25 (--threshold to override). Numeric columns use the
usual quantile-bin PSI; categorical columns (card network, device type,
etc.) use a category-frequency PSI instead, since quantile bins don't mean
anything for a string column.

With --simulate-drift, draws a fresh sample from the holdout set and applies
the same synthetic shift as client.client --drifted (model.data.
apply_synthetic_drift), so the whole loop (check -> retrain -> validate ->
register -> redeploy) can be exercised without needing a real live feed
(brief decision: "Simulated drift from held-back data").

Usage:
    python -m mlops.drift_check                     # check against unshifted holdout sample
    python -m mlops.drift_check --simulate-drift     # force a drifted batch through
    python -m mlops.drift_check --batch new_data.csv # check a real incoming batch
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from model.data import apply_synthetic_drift
from model.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

HOLDOUT_PATH = "models/candidate/holdout.csv"
DRIFT_STATE_PATH = "models/drift_state.json"
DEFAULT_THRESHOLD = 0.25


def _psi_numeric(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for one numeric feature."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cmp_counts, _ = np.histogram(comparison, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cmp_pct = np.clip(cmp_counts / max(len(comparison), 1), 1e-6, None)

    return float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))


def _psi_categorical(reference: pd.Series, comparison: pd.Series) -> float:
    """PSI for one categorical feature: bins are the union of both sides'
    categories, not quantiles."""
    ref = reference.astype(str)
    cmp = comparison.astype(str)
    ref_counts = ref.value_counts()
    cmp_counts = cmp.value_counts()
    ref_total = max(len(ref), 1)
    cmp_total = max(len(cmp), 1)

    psi = 0.0
    for category in set(ref_counts.index) | set(cmp_counts.index):
        ref_pct = max(ref_counts.get(category, 0) / ref_total, 1e-6)
        cmp_pct = max(cmp_counts.get(category, 0) / cmp_total, 1e-6)
        psi += (cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)
    return float(psi)


def compute_drift_score(reference_df: pd.DataFrame, comparison_df: pd.DataFrame) -> dict:
    # Real IEEE-CIS data has heavy missingness in some columns (dist1 is
    # NaN for ~60% of rows) — np.quantile propagates NaN into every edge
    # it computes, which collapses to a single unique value and silently
    # reports 0 PSI regardless of actual drift. Filling with the same
    # sentinel model.features.transform() uses (0.0 / "missing") keeps
    # this measuring the same signal the model actually gets fed.
    per_feature = {
        col: _psi_numeric(
            reference_df[col].fillna(0.0).to_numpy(), comparison_df[col].fillna(0.0).to_numpy()
        )
        for col in NUMERIC_COLUMNS
    }
    per_feature.update(
        {
            col: _psi_categorical(
                reference_df[col].fillna("missing"), comparison_df[col].fillna("missing")
            )
            for col in CATEGORICAL_COLUMNS
        }
    )
    drift_score = float(np.mean(list(per_feature.values())))
    return {"drift_score": drift_score, "per_feature_psi": per_feature}


def _load_comparison_batch(batch_path: str, simulate_drift: bool, reference_df: pd.DataFrame) -> pd.DataFrame:
    if batch_path:
        return pd.read_csv(batch_path)

    sample = reference_df.sample(frac=0.5, random_state=None).reset_index(drop=True)
    if simulate_drift:
        sample = apply_synthetic_drift(sample)
    return sample


def run(threshold: float = DEFAULT_THRESHOLD, batch_path: str = None, simulate_drift: bool = False) -> dict:
    if not os.path.exists(HOLDOUT_PATH):
        raise SystemExit(f"[drift_check] {HOLDOUT_PATH} not found — run `python -m model.train` first.")

    reference_df = pd.read_csv(HOLDOUT_PATH)
    comparison_df = _load_comparison_batch(batch_path, simulate_drift, reference_df)

    result = compute_drift_score(reference_df, comparison_df)
    triggered = result["drift_score"] >= threshold

    state = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "drift_score": round(result["drift_score"], 4),
        "threshold": threshold,
        "triggered": triggered,
        "per_feature_psi": {k: round(v, 4) for k, v in result["per_feature_psi"].items()},
        "comparison_source": batch_path or ("simulated-drift" if simulate_drift else "holdout-sample"),
    }
    os.makedirs(os.path.dirname(DRIFT_STATE_PATH), exist_ok=True)
    with open(DRIFT_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    verdict = "TRIGGERED" if triggered else "ok"
    print(f"[drift_check] drift_score={state['drift_score']} threshold={threshold} -> {verdict}")
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch", default=None, help="CSV of incoming data to check; defaults to a holdout sample")
    parser.add_argument("--simulate-drift", action="store_true")
    args = parser.parse_args()
    state = run(args.threshold, args.batch, args.simulate_drift)
    raise SystemExit(0 if not state["triggered"] else 1)
