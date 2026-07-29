"""Train the Random Forest fraud classifier and log the run to MLflow.

Class imbalance: `class_weight="balanced_subsample"` (not SMOTE) -- see
docs/model_comparison.md. Unlike plain "balanced" (which reweights once
against the whole training set), "balanced_subsample" recomputes weights
per bootstrap sample, which matters here since each tree's bootstrap draw
from a ~0.2%-fraud population can itself land at a different effective
imbalance ratio.

No feature scaling is needed (tree splits are scale-invariant), unlike
train_logreg.py.

Usage:
    python training/train_random_forest.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

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
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print("loading dataset + preparing chronological splits...")
    df = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, _ = prepare_splits(df)

    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_params(
            {
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
            }
        )

        print(f"training on {len(X_train):,} rows...")
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
