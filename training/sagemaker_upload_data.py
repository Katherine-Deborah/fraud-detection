"""Create (if needed) the S3 bucket for Session 6 SageMaker training data,
attach a lifecycle rule so uploaded objects expire automatically, and
upload the raw transactions parquet used for training.

Run once per SageMaker session, after `aws configure` is set up:
    python training/sagemaker_upload_data.py --bucket fraud-detection-sagemaker-<account-id>

Cost/safety: see PROJECT.md §7 and docs/sagemaker.md. This bucket holds
only training input data (a few hundred MB), not model artifacts long
term -- the lifecycle rule expires objects under training-data/ after
EXPIRATION_DAYS so nothing accumulates if this project's SageMaker
experiments are repeated across several sessions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_TRANSACTIONS_PATH = REPO_ROOT / "data" / "raw" / "transactions.parquet"
EXPIRATION_DAYS = 14


def ensure_bucket(s3_client, bucket: str, region: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"bucket {bucket} already exists")
        return
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code not in ("404", "NoSuchBucket"):
            raise
    print(f"creating bucket {bucket} in {region} ...")
    if region == "us-east-1":
        s3_client.create_bucket(Bucket=bucket)
    else:
        s3_client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )


def apply_lifecycle_rule(s3_client, bucket: str) -> None:
    s3_client.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "expire-training-data",
                    "Filter": {"Prefix": "training-data/"},
                    "Status": "Enabled",
                    "Expiration": {"Days": EXPIRATION_DAYS},
                }
            ]
        },
    )
    print(f"lifecycle rule set: objects under training-data/ expire after {EXPIRATION_DAYS} days")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="S3 bucket name (must be globally unique)")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--prefix", default="training-data/rf-session6")
    args = parser.parse_args()

    if not RAW_TRANSACTIONS_PATH.exists():
        print(
            f"ERROR: {RAW_TRANSACTIONS_PATH} not found -- generate the dataset first "
            f"(python data_generation/generate_transactions.py)",
            file=sys.stderr,
        )
        sys.exit(1)

    session = boto3.session.Session(region_name=args.region)
    s3 = session.client("s3")

    ensure_bucket(s3, args.bucket, args.region)
    apply_lifecycle_rule(s3, args.bucket)

    prefix = args.prefix.rstrip("/")
    key = f"{prefix}/transactions.parquet"
    size_mb = RAW_TRANSACTIONS_PATH.stat().st_size / 1e6
    print(f"uploading {RAW_TRANSACTIONS_PATH} ({size_mb:.1f} MB) to s3://{args.bucket}/{key} ...")
    s3.upload_file(str(RAW_TRANSACTIONS_PATH), args.bucket, key)

    s3_uri = f"s3://{args.bucket}/{prefix}/"
    print(f"done. training input S3 URI (pass to sagemaker_launch.py --s3-input-uri): {s3_uri}")


if __name__ == "__main__":
    main()
