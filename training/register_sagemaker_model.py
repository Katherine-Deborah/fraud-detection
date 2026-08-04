"""Pull the model artifact from a completed Session 6 SageMaker Training Job
and register it in the local MLflow registry alongside the Session 5
locally-trained models, tagged trained_via=sagemaker.

Usage:
    python training/register_sagemaker_model.py \\
        --model-data s3://fraud-detection-sagemaker-<account-id>/.../model.tar.gz \\
        --job-name fraud-rf-sagemaker-2026-...

Both values are printed by training/sagemaker_launch.py when the job
finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import joblib
import mlflow
import mlflow.sklearn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data_prep import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI  # noqa: E402


def download_and_extract(model_data_uri: str, dest_dir: Path) -> None:
    parsed = urlparse(model_data_uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    tar_path = dest_dir / "model.tar.gz"
    boto3.client("s3").download_file(bucket, key, str(tar_path))
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data", required=True, help="S3 URI of model.tar.gz from the training job")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment-name", default=MLFLOW_EXPERIMENT_NAME)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"downloading {args.model_data} ...")
        download_and_extract(args.model_data, tmp_path)

        model = joblib.load(tmp_path / "model.joblib")
        with open(tmp_path / "metrics.json") as f:
            payload = json.load(f)
        params, metrics = payload["params"], payload["metrics"]

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment_name)

        with mlflow.start_run(run_name="random_forest_sagemaker"):
            mlflow.set_tags({"trained_via": "sagemaker", "sagemaker_job_name": args.job_name})
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.sklearn.log_model(model, artifact_path="model")

        print("registered in MLflow as run 'random_forest_sagemaker', tagged trained_via=sagemaker")
        for key, value in metrics.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
