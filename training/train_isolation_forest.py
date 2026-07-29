"""Train the Isolation Forest fraud detector and log the run to MLflow.

Unsupervised: fit does **not** see `is_fraud` at all, so neither class
weighting nor SMOTE applies here -- the model's only imbalance-related
knob is `contamination`, the expected anomaly fraction, which we set from
the *training split's own* fraud rate (a reasonable prior in a real
deployment: you know roughly how rare fraud is even without per-row
labels at serving time). Labels are used only for threshold selection
(on validation) and evaluation (on test), exactly like the other 3 models,
so the model comparison in docs/model_comparison.md stays apples-to-apples.

Anomaly scores from `decision_function` run the opposite direction from
the other models' fraud probabilities (higher = more normal, lower/more
negative = more anomalous), so we negate it into a "fraud score" before
handing it to the shared evaluate_model() helper.

Usage:
    python training/train_isolation_forest.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest

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
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print("loading dataset + preparing chronological splits...")
    df = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, _ = prepare_splits(df)
    contamination = float(y_train.mean())

    with mlflow.start_run(run_name="isolation_forest"):
        mlflow.log_params(
            {
                "model_type": "IsolationForest",
                "class_imbalance_strategy": "unsupervised, contamination set from train fraud rate",
                "n_estimators": args.n_estimators,
                "contamination": contamination,
                "n_features": X_train.shape[1],
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "fpr_target": args.fpr_target,
            }
        )

        print(f"training on {len(X_train):,} rows (unsupervised, contamination={contamination:.5f})...")
        start = time.time()
        model = IsolationForest(
            n_estimators=args.n_estimators,
            contamination=contamination,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train)  # no y -- unsupervised
        train_seconds = time.time() - start
        mlflow.log_metric("train_seconds", train_seconds)

        # Negate: decision_function is high for normal points, low/negative
        # for anomalies -- flip sign so higher score means "more fraud-like",
        # matching the convention evaluate_model()/roc_curve expect.
        val_score = -model.decision_function(X_val)
        test_score = -model.decision_function(X_test)
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
