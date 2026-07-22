"""Loads a small stratified sample of the IEEE-CIS Fraud Detection dataset
(real Vesta e-commerce transactions — card network/type, product code,
purchaser email domain, device type, plus a few of IEEE-CIS's own
transaction-aggregate columns), or generates a schema-compatible synthetic
dataset for local development when the real sample isn't present.

The real sample is produced by mlops/download_ieee_cis.py, which needs a
Kaggle account with an API token AND that account having accepted the
IEEE-CIS competition rules on kaggle.com — see that script's docstring.

Also owns the train/holdout split: the holdout slice is deliberately
reserved so mlops/drift_check.py has real, never-trained-on data to compare
against, and apply_synthetic_drift(), the shared shift used by both
client.client --drifted and mlops/drift_check.py --simulate-drift so
"drifted" means the same thing in both places.
"""
import numpy as np
import pandas as pd

from model.features import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS

RAW_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
LABEL_COLUMN = "isFraud"
ALL_COLUMNS = RAW_FEATURE_COLUMNS + [LABEL_COLUMN]

DEFAULT_CSV_PATH = "data/ieee_cis_sample.csv"

_CARD_NETWORKS = ["visa", "mastercard", "amex", "discover"]
_CARD_TYPES = ["credit", "debit"]
_PRODUCT_CODES = ["W", "C", "R", "H", "S"]
_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "anonymous.com"]
_DEVICE_TYPES = ["desktop", "mobile", "unknown"]


def _synthesize(n_rows: int = 20000, fraud_rate: float = 0.035, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    def make_rows(n: int, fraud: bool) -> pd.DataFrame:
        # Shapes chosen only to be schema-compatible and to skew
        # fraud/legit apart enough to exercise validation/drift logic — not
        # a realistic fraud model. Real signal comes from the actual
        # IEEE-CIS sample once mlops/download_ieee_cis.py has run.
        return pd.DataFrame(
            {
                "TransactionAmt": rng.exponential(scale=140.0 if fraud else 80.0, size=n),
                "dist1": rng.exponential(scale=50.0 if fraud else 15.0, size=n),
                "C1": rng.poisson(lam=3.0 if fraud else 1.0, size=n).astype(float),
                "C2": rng.poisson(lam=2.5 if fraud else 1.0, size=n).astype(float),
                "D1": rng.exponential(scale=5.0 if fraud else 60.0, size=n),
                "ProductCD": rng.choice(
                    _PRODUCT_CODES, size=n,
                    p=[0.3, 0.2, 0.2, 0.15, 0.15] if fraud else [0.55, 0.15, 0.1, 0.1, 0.1],
                ),
                "card4": rng.choice(_CARD_NETWORKS, size=n),
                "card6": rng.choice(_CARD_TYPES, size=n, p=[0.75, 0.25]),
                "P_emaildomain": rng.choice(
                    _EMAIL_DOMAINS, size=n,
                    p=[0.15, 0.15, 0.15, 0.15, 0.4] if fraud else [0.4, 0.2, 0.15, 0.15, 0.1],
                ),
                "DeviceType": rng.choice(
                    _DEVICE_TYPES, size=n,
                    p=[0.3, 0.4, 0.3] if fraud else [0.5, 0.45, 0.05],
                ),
                "isFraud": np.full(n, 1 if fraud else 0),
            }
        )

    df = pd.concat([make_rows(n_legit, False), make_rows(n_fraud, True)], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_dataset(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Loads the real IEEE-CIS sample if present, else falls back to a
    synthetic stand-in with the same schema. Prints which source was used
    so it's never silently ambiguous which data trained/validated a given
    model."""
    import os

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        missing = set(ALL_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")
        print(f"[data] loaded real IEEE-CIS sample from {csv_path} ({len(df)} rows)")
        return df[ALL_COLUMNS]

    print(
        f"[data] {csv_path} not found — using synthetic stand-in dataset. "
        "Run `python -m mlops.download_ieee_cis` (needs KAGGLE_USERNAME/KAGGLE_KEY "
        f"and an account that has accepted the competition rules) for a real sample at {csv_path}."
    )
    return _synthesize()


def split_train_holdout(df: pd.DataFrame, holdout_frac: float = 0.2, seed: int = 42):
    """Splits into a training slice and a holdout slice. The holdout slice
    is never used for training — it's reserved for drift simulation
    (optionally shifted further by apply_synthetic_drift) and for the
    FR-15 promotion gate's held-out metric comparison."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - holdout_frac))
    return shuffled.iloc[:cut].reset_index(drop=True), shuffled.iloc[cut:].reset_index(drop=True)


def apply_synthetic_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Shared drift simulation for client.client --drifted and
    mlops/drift_check.py --simulate-drift, so a "drifted" batch means the
    same thing in both places. Shifts amount/aggregate columns upward and
    skews the categorical mix toward patterns associated with fraud
    (anonymous email domain, mobile device) — a stand-in for a real
    attack-pattern shift, not a claim about real fraud demographics."""
    shifted = df.copy()
    for col in ["dist1", "C1", "C2", "D1"]:
        shifted[col] = shifted[col] * 3.0
    shifted["TransactionAmt"] = shifted["TransactionAmt"] * 2.5
    shifted["P_emaildomain"] = "anonymous.com"
    shifted["DeviceType"] = "mobile"
    return shifted
