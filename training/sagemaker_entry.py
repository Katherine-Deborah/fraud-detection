"""SageMaker script-mode training entry point for the Random Forest fraud
classifier -- runs inside the AWS-managed SKLearn training container
(Session 6). Reuses the exact same data_prep.py chronological split /
feature engineering / evaluation code as the local train_random_forest.py
(Session 5), same hyperparameters, so the SageMaker-trained model is
comparable apples-to-apples, not a reimplementation.

Deployed via training/sagemaker_launch.py with source_dir=training/,
dependencies=[data_generation/] -- inside the container this file and
data_prep.py sit flat in /opt/ml/code (no "training." package prefix),
so the import below is bare (`from data_prep import ...`), unlike the
local scripts which use `from training.data_prep import ...`.

Does NOT talk to MLflow directly -- the training container has no network
route back to the local docker-compose MLflow server. Instead this script
writes model.joblib + metrics.json into SM_MODEL_DIR, which SageMaker tars
into model.tar.gz and uploads to S3; training/register_sagemaker_model.py
then downloads that artifact after the job completes and logs it into the
local MLflow registry.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from data_prep import evaluate_model, load_dataset, prepare_splits  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # SageMaker passes each Estimator `hyperparameters` dict entry as a
    # "--key value" CLI arg automatically -- these must match the keys
    # used in training/sagemaker_launch.py.
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--fpr-target", type=float, default=0.02)
    # SageMaker sets these env vars inside the container; the defaults let
    # this script also run as a local smoke test before launching for real.
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    args = parser.parse_args()

    data_path = Path(args.train) / "transactions.parquet"
    print(f"loading dataset from {data_path} ...")
    df = load_dataset(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test, _fill_values = prepare_splits(df)

    print(f"training RandomForestClassifier on {len(X_train):,} rows...")
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    train_seconds = time.time() - start

    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(
        y_val.to_numpy(), val_score, y_test.to_numpy(), test_score, args.fpr_target
    )
    metrics["train_seconds"] = train_seconds

    params = {
        "model_type": "RandomForestClassifier",
        "class_imbalance_strategy": "class_weight=balanced_subsample",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "n_features": X_train.shape[1],
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "fpr_target": args.fpr_target,
        "trained_via": "sagemaker",
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.joblib")
    with open(model_dir / "metrics.json", "w") as f:
        json.dump({"params": params, "metrics": metrics}, f, indent=2)

    print(f"trained in {train_seconds:.1f}s")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
