"""Runs train -> validate -> package -> register as one call, so CI (and
humans) don't have to shell-parse intermediate directory names between
steps. Unconditional — no drift check or regression gate (that's
mlops/retrain.py's job, for the drift-triggered path on the EC2 box).

Usage:
    python -m model.pipeline [--data data/creditcard.csv]
"""
import argparse
import sys

from model.package import package
from model.register import register
from model.train import train
from model.validate import validate


def run(data_path: str = "data/creditcard.csv", candidate_dir: str = "models/candidate") -> dict:
    train(data_path=data_path, out_dir=candidate_dir)

    passed, record = validate(candidate_dir)
    if not passed:
        print(f"[pipeline] validation failed: {record['failures']} — not registering")
        sys.exit(1)

    package_dir = package(candidate_dir)
    pointer = register(package_dir)
    return pointer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/creditcard.csv")
    args = parser.parse_args()
    run(args.data)
