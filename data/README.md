Put a small, real sample of the IEEE-CIS Fraud Detection dataset here as
`ieee_cis_sample.csv` — real Vesta e-commerce transactions with named,
bank-like features (card network/type, product code, purchaser email
domain, device type, transaction amount) rather than anonymized PCA
components.

Produced by:

    KAGGLE_API_TOKEN=... python -m mlops.download_ieee_cis

This needs a Kaggle account with an API token (kaggle.com/settings/api —
"Generate New Token", a single string, Kaggle's current auth flow) AND
that account having accepted the competition rules at
kaggle.com/competitions/ieee-fraud-detection/rules. The two are checked at
different points: authenticating (and even listing the competition's
files) can succeed on the token alone — only the actual file download
403s if the rules haven't been accepted, and there's no way to script
around that part.

Without `ieee_cis_sample.csv`, `model/data.py` falls back to a synthetic,
schema-compatible dataset — fine for exercising the pipeline, not for a
model worth deploying.

(`creditcard.csv`, the earlier Kaggle ULB dataset this project started
with, may still be sitting in this folder from before the IEEE-CIS switch
— it's no longer read by anything and can be deleted.)
