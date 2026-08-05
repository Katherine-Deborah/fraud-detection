"""Session 8, step 1 of 2 for ONNX export: pull the current production
model + its metrics out of MLflow and dump it to a plain joblib file.

Split into its own process/script deliberately: importing mlflow and
onnx/skl2onnx/onnxruntime in the *same* Python process segfaults on this
machine (STATUS_ACCESS_VIOLATION), reproduced down to `import mlflow`
followed by `from skl2onnx import convert_sklearn` with no other code in
between -- almost certainly a native protobuf descriptor-pool conflict
between the two packages' bundled generated proto code, not anything
specific to this project. Keeping mlflow's import confined to this script,
and onnx's to training/export_to_onnx.py, sidesteps it entirely: this
script never imports onnx/skl2onnx/onnxruntime, and export_to_onnx.py
never imports mlflow.

Run this first, then training/export_to_onnx.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import registry  # noqa: E402

STAGING_DIR = REPO_ROOT / "serving" / "_export_staging"


def fetch() -> None:
    client = registry.get_client()
    prod_version = registry.get_current_production(client)
    if prod_version is None:
        raise SystemExit("No model version currently tagged production -- run Session 7's registry walkthrough first.")

    algorithm = prod_version.tags.get(registry.TAG_ALGORITHM, "?")
    print(f"Fetching production model: v{prod_version.version} ({algorithm}), run_id={prod_version.run_id}")

    model_uri = f"runs:/{prod_version.run_id}/model"
    sk_model = mlflow.sklearn.load_model(model_uri)
    metrics = registry.source_run_metrics(client, prod_version)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    model_path = STAGING_DIR / "model.joblib"
    joblib.dump(sk_model, model_path)

    metadata = {
        "model_version": prod_version.version,
        "algorithm": algorithm,
        "source_run_id": prod_version.run_id,
        "threshold": metrics["threshold"],
        "test_auc_pr": metrics.get("auc_pr"),
        "test_recall": metrics.get("recall"),
    }
    metadata_path = STAGING_DIR / "source_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {model_path.relative_to(REPO_ROOT)} and {metadata_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    fetch()
