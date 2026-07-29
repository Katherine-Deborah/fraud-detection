"""Pull the latest run of each of the 4 Session 5 models from MLflow, build
a comparison table + bar chart, and write docs/model_comparison.md.

Run this after all 4 training scripts (train_logreg.py,
train_random_forest.py, train_isolation_forest.py, train_lstm.py) have
logged at least one run to the "fraud-detection" MLflow experiment.

Usage:
    python training/compare_models.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data_prep import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI  # noqa: E402

RUN_NAME_TO_LABEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "isolation_forest": "Isolation Forest",
    "lstm": "LSTM",
}
METRIC_COLUMNS = ["auc_pr", "precision", "recall", "f1", "val_recall_at_fpr_target", "threshold"]
PLOT_PATH = REPO_ROOT / "docs" / "eda" / "model_comparison.png"
DOC_PATH = REPO_ROOT / "docs" / "model_comparison.md"


def fetch_latest_runs(tracking_uri: str, experiment_name: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"No MLflow experiment named {experiment_name!r} -- run the 4 train_*.py scripts first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise SystemExit("Experiment exists but has no runs yet -- run the 4 train_*.py scripts first.")

    rows = []
    for run_name in RUN_NAME_TO_LABEL:
        matches = runs[runs["tags.mlflow.runName"] == run_name]
        if matches.empty:
            print(f"WARNING: no run found for {run_name!r} -- skipping (run its training script)")
            continue
        latest = matches.iloc[0]  # already sorted DESC by start_time
        row = {"run_name": run_name, "label": RUN_NAME_TO_LABEL[run_name]}
        for col in METRIC_COLUMNS:
            row[col] = latest.get(f"metrics.{col}")
        row["train_seconds"] = latest.get("metrics.train_seconds")
        row["class_imbalance_strategy"] = latest.get("params.class_imbalance_strategy")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(df["label"], df["auc_pr"], color="#4C72B0")
    axes[0].set_title("AUC-PR (test set)")
    axes[0].set_ylabel("AUC-PR")
    axes[0].tick_params(axis="x", rotation=20)

    width = 0.35
    x = range(len(df))
    axes[1].bar([i - width / 2 for i in x], df["precision"], width, label="Precision", color="#DD8452")
    axes[1].bar([i + width / 2 for i in x], df["recall"], width, label="Recall", color="#55A868")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(df["label"], rotation=20)
    axes[1].set_title(f"Precision / Recall at the FPR<=2% threshold (test set)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_doc(df: pd.DataFrame, doc_path: Path, plot_path: Path) -> None:
    best_auc_pr = df.loc[df["auc_pr"].idxmax()]
    best_recall = df.loc[df["recall"].idxmax()]

    table_lines = [
        "| Model | AUC-PR | Precision | Recall | F1 | Recall @ FPR<=2% (val) | Threshold | Train time (s) | Imbalance strategy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        table_lines.append(
            f"| {row['label']} | {row['auc_pr']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['val_recall_at_fpr_target']:.4f} | {row['threshold']:.4f} | "
            f"{row['train_seconds']:.1f} | {row['class_imbalance_strategy']} |"
        )
    table = "\n".join(table_lines)
    relative_plot_path = plot_path.relative_to(doc_path.parent)

    content = f"""# Model Comparison — Session 5

All 4 models were trained on a **chronological** train/val/test split
(70/15/15 by timestamp span, not row count) of the synthetic transaction
dataset (see `docs/dataset.md`) -- no future transactions leak into
training. The classification threshold for each model was selected on the
**validation** set to hit the target false-positive rate from
PROJECT.md #6 (recall >= 90% at FPR <= 2%), then applied unchanged to the
held-out **test** set to produce the precision/recall/F1 numbers below.
AUC-PR is threshold-independent and computed directly on test. Plain
accuracy is not reported anywhere, deliberately -- at a ~0.2% fraud rate
it would be >99% for a model that never flags anything.

Class imbalance is handled via **class weighting**, not SMOTE, for every
model -- see the "Imbalance strategy" column below and the training
scripts (`training/train_*.py`) for the per-model mechanism and rationale.

## Results

{table}

*(Numbers above are pulled live from MLflow by `training/compare_models.py`
-- rerun it after retraining to refresh this table.)*

![Model comparison]({relative_plot_path.as_posix()})

## Recommendation

**{best_auc_pr['label']}** is the recommended production candidate: it has
the highest AUC-PR ({best_auc_pr['auc_pr']:.4f}) of the 4 models, which
matters more than any single-threshold metric here since AUC-PR summarizes
ranking quality across the full precision/recall tradeoff -- the threshold
actually deployed can still be tuned independently at serving time (and
will be, formally, in Session 7's staged rollout).

{best_recall['label']} achieves the highest recall at the FPR<=2% operating
point ({best_recall['recall']:.4f}), which is the more relevant number if
the deployment goal is "catch as much fraud as possible within a fixed
false-positive budget" rather than overall ranking quality -- worth
revisiting once real (non-synthetic) traffic and a real cost-of-a-false-
positive number are available.

Isolation Forest's numbers reflect that it never sees `is_fraud` during
training (contamination is set from the training split's fraud rate, not
learned) -- it's included as the unsupervised anomaly-detection baseline
PROJECT.md calls for, not because it's expected to beat the 3 supervised
models on this labeled dataset.

## What I'd try with more time

- Hyperparameter search (currently single fixed configs per model) --
  particularly Random Forest depth/leaf-size and the LSTM's hidden size /
  sequence length, both chosen as reasonable defaults rather than tuned.
- Train the LSTM on the full 5M-row dataset rather than the 10,000-account
  (~1M row) subsample used here (a deliberate local-GPU scale tradeoff,
  documented in `training/train_lstm.py`) -- likely via SageMaker in
  Session 6, which is exactly the kind of workload a managed training job
  is for.
- An explicit padding mask for the LSTM's left-padded short sequences,
  instead of relying on the network to learn to ignore normalized-zero
  padding steps.
"""
    doc_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment-name", default=MLFLOW_EXPERIMENT_NAME)
    args = parser.parse_args()

    df = fetch_latest_runs(args.tracking_uri, args.experiment_name)
    print(df[["label", "auc_pr", "precision", "recall", "f1"]].to_string(index=False))

    plot_comparison(df, PLOT_PATH)
    write_doc(df, DOC_PATH, PLOT_PATH)
    print(f"\nwrote {PLOT_PATH}")
    print(f"wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
