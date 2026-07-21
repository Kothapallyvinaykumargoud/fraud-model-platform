"""Trains a fraud-detection classifier. Deliberately simple — model quality
is a means, not the goal (PRD §7 SM-C1 counter-metric: don't optimize
prediction accuracy over the platform work).

Usage:
    python -m model.train [--data data/creditcard.csv] [--out models/candidate]
"""
import argparse
import json
import os
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model.data import FEATURE_COLUMNS, LABEL_COLUMN, load_dataset, split_train_holdout


def train(data_path: str, out_dir: str, seed: int = 42) -> dict:
    df = load_dataset(data_path)
    train_df, holdout_df = split_train_holdout(df, holdout_frac=0.2, seed=seed)

    # Holdout is reserved for drift simulation — carve the model's own
    # test split out of the training slice only.
    X = train_df[FEATURE_COLUMNS]
    y = train_df[LABEL_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "average_precision": float(average_precision_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_holdout_rows": int(len(holdout_df)),
        "fraud_rate_train": float(y_train.mean()),
    }

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(out_dir, "model.joblib"))
    holdout_df.to_csv(os.path.join(out_dir, "holdout.csv"), index=False)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_path if os.path.exists(data_path) else "synthetic",
        "metrics": metrics,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "model_type": "RandomForestClassifier",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[train] wrote candidate model + metadata to {out_dir}")
    print(f"[train] metrics: {json.dumps(metrics, indent=2)}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--out", default="models/candidate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.out, args.seed)
