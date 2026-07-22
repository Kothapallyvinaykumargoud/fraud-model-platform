"""FR-18: Feature Store. Centralizes feature transformation logic — the
same numeric columns and categorical vocabulary, versioned as one JSON
artifact, used by both training and serving instead of being duplicated in
each. A model trained against one version of these definitions must be
served with that same version, which is why register()/register_shadow()
copy feature_definitions.json alongside model.joblib rather than letting
each side reconstruct the transform independently.

This is intentionally NOT a managed feature platform (no SageMaker Feature
Store, no online/offline store split) — just one file-based module that
both sides import, sized for a single-operator project.

Usage:
    definitions = fit_feature_definitions(train_df)   # at train time
    X = transform(train_df, definitions)
    save_feature_definitions(definitions, path)

    definitions = load_feature_definitions(path)       # at serving time
    X = transform(incoming_df, definitions)
"""
import json

FEATURE_STORE_VERSION = "v1"

# Raw columns as they appear in the IEEE-CIS sample (see model/data.py) —
# a deliberately small, named-feature subset (not the full ~400 columns)
# chosen to read as "bank-like": amount, product, card network/type,
# purchaser email domain, device, plus a few of IEEE-CIS's own aggregate
# features (dist1 = address/zip distance, C1/C2 = count aggregates,
# D1 = days-since-last-transaction).
NUMERIC_COLUMNS = ["TransactionAmt", "dist1", "C1", "C2", "D1"]
CATEGORICAL_COLUMNS = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]
UNKNOWN_CATEGORY = "__unknown__"


def fit_feature_definitions(train_df, max_categories: int = 20) -> dict:
    """Builds the categorical vocabulary from a training slice. Values not
    in the vocabulary at serving time fall into UNKNOWN_CATEGORY rather
    than growing the feature space or erroring."""
    vocab = {
        col: train_df[col].fillna("missing").astype(str).value_counts().head(max_categories).index.tolist()
        for col in CATEGORICAL_COLUMNS
    }
    return {
        "version": FEATURE_STORE_VERSION,
        "numeric_columns": NUMERIC_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "vocab": vocab,
    }


def model_feature_names(definitions: dict) -> list[str]:
    names = list(definitions["numeric_columns"])
    for col in definitions["categorical_columns"]:
        names.extend(f"{col}={cat}" for cat in definitions["vocab"][col])
        names.append(f"{col}={UNKNOWN_CATEGORY}")
    return names


def transform(df, definitions: dict):
    """Raw (possibly mixed-type) columns in, one all-numeric row per input
    row out — the exact matrix shape the model was trained on, regardless
    of which categories this particular batch happens to contain."""
    import pandas as pd

    out = {}
    for col in definitions["numeric_columns"]:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in definitions["categorical_columns"]:
        # fillna BEFORE astype(str): once cast to string, a real NaN
        # becomes the literal string "nan" and fillna can no longer see it
        # (pandas only fills actual nulls) — same category of bug either
        # way (falls through to UNKNOWN_CATEGORY), but "missing" is the
        # intended, readable sentinel, not an accidental "nan" string.
        values = df[col].fillna("missing").astype(str)
        vocab = definitions["vocab"][col]
        for cat in vocab:
            out[f"{col}={cat}"] = (values == cat).astype(float)
        out[f"{col}={UNKNOWN_CATEGORY}"] = (~values.isin(vocab)).astype(float)

    result = pd.DataFrame(out, index=df.index)
    return result[model_feature_names(definitions)]


def save_feature_definitions(definitions: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(definitions, f, indent=2)


def load_feature_definitions(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
