Put the Kaggle Credit Card Fraud Detection (ULB) dataset here as `creditcard.csv`:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Without it, `model/data.py` falls back to a synthetic, schema-compatible
dataset — fine for exercising the pipeline, not for a model worth deploying.
