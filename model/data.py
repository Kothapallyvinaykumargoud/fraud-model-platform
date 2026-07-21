"""Loads the Kaggle Credit Card Fraud Detection (ULB) dataset, or generates a
schema-compatible synthetic dataset for local development when the real CSV
isn't present (the real dataset requires a Kaggle account to download:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Also owns the train/holdout split: the holdout slice is deliberately reserved
so mlops/drift_check.py has real, never-trained-on data to compare against
and to simulate drift with (per brief decision: "Simulated drift from
held-out data").
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
LABEL_COLUMN = "Class"
ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

DEFAULT_CSV_PATH = "data/creditcard.csv"


def _synthesize(n_rows: int = 20000, fraud_rate: float = 0.0017, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    def make_rows(n, fraud: bool):
        # V1-V28 are PCA components in the real dataset; fraud rows there skew
        # away from 0 on several components. We mimic that shape only closely
        # enough to make validation/drift logic exercisable, not to be a
        # realistic fraud model.
        shift = 2.5 if fraud else 0.0
        v = rng.normal(loc=shift, scale=1.0, size=(n, 28))
        time = rng.uniform(0, 172792, size=n)
        amount = rng.exponential(scale=88.0 if not fraud else 120.0, size=n)
        label = np.full(n, 1 if fraud else 0)
        return np.column_stack([v, time, amount, label])

    rows = np.vstack([make_rows(n_legit, False), make_rows(n_fraud, True)])
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=ALL_COLUMNS)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df


def load_dataset(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Loads the real Kaggle CSV if present, else falls back to a synthetic
    stand-in with the same schema. Prints which source was used so it's never
    silently ambiguous which data trained/validated a given model."""
    import os

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        missing = set(ALL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")
        print(f"[data] loaded real dataset from {csv_path} ({len(df)} rows)")
        return df[ALL_COLUMNS]

    print(
        f"[data] {csv_path} not found — using synthetic stand-in dataset. "
        "Download the real dataset from "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it "
        f"at {csv_path} for a real model."
    )
    return _synthesize()


def split_train_holdout(df: pd.DataFrame, holdout_frac: float = 0.2, seed: int = 42):
    """Splits into a training slice and a holdout slice. The holdout slice is
    never used for training — it's reserved for drift simulation (optionally
    shifted further by mlops/drift_check.py) and for the FR-15 promotion
    gate's held-out metric comparison."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - holdout_frac))
    return shuffled.iloc[:cut].reset_index(drop=True), shuffled.iloc[cut:].reset_index(drop=True)
