"""FR-19: Explainability. A validation-time SHAP feature-attribution
report, computed once per training run on a small sample — not per live
request. Per-request SHAP would add real CPU cost to every /predict call,
and the 1GB RAM box has no budget for that (see README "RAM budget"); a
report alongside the candidate model answers "what does this model version
generally weigh" without touching the serving path's latency at all.

Usage: called from model/train.py right after fitting; writes
shap_summary.json alongside the candidate model.
"""


def compute_shap_summary(clf, X_sample, feature_names: list[str], max_rows: int = 200, top_n: int = 10) -> dict:
    import numpy as np
    import shap

    sample = X_sample.sample(n=min(max_rows, len(X_sample)), random_state=42)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(sample)

    # Binary RandomForestClassifier: some shap/sklearn version combos return
    # a list of per-class arrays ([not_fraud, fraud]), others a single
    # (n_rows, n_features) array already scoped to the positive class —
    # handle both rather than pinning to one shap version's shape.
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(values).mean(axis=0)

    ranked = sorted(zip(feature_names, mean_abs.tolist()), key=lambda kv: kv[1], reverse=True)
    return {
        "sample_size": int(len(sample)),
        "top_features": [
            {"feature": name, "mean_abs_shap": round(value, 6)} for name, value in ranked[:top_n]
        ],
    }
