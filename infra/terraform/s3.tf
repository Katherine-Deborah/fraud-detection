# S3 bucket for SageMaker training data (PROJECT.md §5.5 / §7,
# training/sagemaker_upload_data.py's Session-6-created bucket, now
# managed as code). Scope is deliberately narrow -- this module owns only
# the training-job supporting resources (this bucket + the execution role
# in iam.tf), not the training job itself; see iam.tf's header comment for
# why a SageMaker Training Job isn't a Terraform resource here at all.

data "aws_caller_identity" "current" {}

locals {
  bucket_name = coalesce(var.bucket_name, "fraud-detection-sagemaker-${data.aws_caller_identity.current.account_id}")
}

resource "aws_s3_bucket" "training_data" {
  bucket = local.bucket_name
  tags   = var.tags
}

# Session 6 never turned this on and got away with it only because nothing
# sensitive was ever public by default -- explicit is better than relying
# on the account-level default (which can be changed by someone else later).
resource "aws_s3_bucket_public_access_block" "training_data" {
  bucket = aws_s3_bucket.training_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "training_data" {
  bucket = aws_s3_bucket.training_data.id

  rule {
    id     = "expire-all-objects"
    status = "Enabled"

    # No `filter` block == applies to the whole bucket, not just
    # training-data/. Session 6's script-created rule only covered
    # training-data/, leaving code/ and output/ (also written by
    # training/sagemaker_launch.py) to accumulate indefinitely --
    # documented as a known, unfixed gap in docs/sagemaker.md. This is
    # that fix.
    filter {}

    expiration {
      days = var.lifecycle_expiration_days
    }

    # Also expire abandoned multipart uploads -- a real (if minor) cost
    # leak `training/sagemaker_upload_data.py`'s rule never addressed,
    # since a failed/interrupted upload otherwise bills storage forever
    # with no object ever appearing to explain it.
    abort_incomplete_multipart_upload {
      days_after_initiation = var.lifecycle_expiration_days
    }
  }
}
