"""Session 9: snapshots a sample of the training feature distribution to
disk, as the reference dataset for the Airflow drift_report task (see
dags/fraud_pipeline_dag.py).

Deliberately decoupled from serving/model_metadata.json (Session 8's
export, which is about *serving* the current production model) -- this
script is a *monitoring* concern: "what did the feature distribution look
like when the currently-deployed model was trained." It happens to reuse
the same chronological train split and sentinel fill values as training
(training/data_prep.py), but persists its own copy of the fill values
alongside the reference sample so the two artifacts can evolve
independently and the DAG doesn't need to know about serving/ at all.

X_train itself is ~3.5M rows -- far more than a drift comparison needs and
too large to comfortably commit or ship around. A fixed-seed random sample
keeps the reference file small while still faithfully representing the
training distribution's shape (this is a distributional reference, not a
lookup table where every row matters).

Run once; regenerate only if the underlying dataset or feature schema
changes:
    .venv/Scripts/python training/export_reference_distribution.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data_prep import FEATURE_COLUMNS, load_dataset, prepare_splits  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "processed"
SAMPLE_SIZE = 100_000
SEED = 42


def main() -> None:
    df = load_dataset()
    X_train, _y_train, _X_val, _y_val, _X_test, _y_test, fill_values = prepare_splits(df)

    n = min(SAMPLE_SIZE, len(X_train))
    sample = X_train.sample(n=n, random_state=SEED)[FEATURE_COLUMNS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features_path = OUT_DIR / "reference_features.parquet"
    fill_values_path = OUT_DIR / "reference_fill_values.json"

    sample.to_parquet(features_path, index=False)
    fill_values_path.write_text(json.dumps(fill_values, indent=2), encoding="utf-8")

    print(f"Sampled {n:,} of {len(X_train):,} training rows -> {features_path}")
    print(f"Sentinel fill values -> {fill_values_path}: {fill_values}")


if __name__ == "__main__":
    main()
