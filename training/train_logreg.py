"""Train the Logistic Regression baseline fraud classifier and log the run
to MLflow.

Class imbalance: `class_weight="balanced"` (not SMOTE) -- see
docs/model_comparison.md for the full rationale; in short, at a ~0.2%
fraud rate with 3.5M+ training rows, reweighting the loss is simpler than
generating synthetic minority samples and carries no risk of a synthetic
point leaking across the chronological train/val/test boundary.

Usage:
    python training/train_logreg.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data_prep import (  # noqa: E402
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    evaluate_model,
    load_dataset,
    prepare_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment-name", default=MLFLOW_EXPERIMENT_NAME)
    parser.add_argument("--fpr-target", type=float, default=0.02)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print("loading dataset + preparing chronological splits...")
    df = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, _ = prepare_splits(df)

    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_imbalance_strategy": "class_weight=balanced",
                "C": args.C,
                "max_iter": args.max_iter,
                "n_features": X_train.shape[1],
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "fpr_target": args.fpr_target,
            }
        )

        print(f"training on {len(X_train):,} rows...")
        start = time.time()
        # StandardScaler is required here, not optional: raw feature scales
        # span orders of magnitude (e.g. time_since_last_txn_sec in the
        # hundreds of thousands of seconds vs. 0/1 flag columns), which both
        # stalls lbfgs convergence and lets large-scale features dominate
        # the learned coefficients regardless of actual signal strength.
        # Fit on train only, applied unchanged to val/test.
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=args.C, max_iter=args.max_iter),
        )
        model.fit(X_train, y_train)
        train_seconds = time.time() - start
        mlflow.log_metric("train_seconds", train_seconds)

        val_score = model.predict_proba(X_val)[:, 1]
        test_score = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(
            y_val.to_numpy(), val_score, y_test.to_numpy(), test_score, args.fpr_target
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"trained in {train_seconds:.1f}s")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
