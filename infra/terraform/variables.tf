variable "region" {
  description = <<-EOT
    AWS region to create resources in. Deliberately has no default:
    Session 6 (docs/sagemaker.md) lost real time to region assumptions --
    a Service Quota increase was approved in ap-southeast-2 while the
    bucket/CLI default region was us-east-1/us-west-2. Pick the region
    where you actually have (or will request) the
    "ml.m5.large for training job usage" quota, and pass it explicitly:
      terraform apply -var="region=ap-southeast-2"
  EOT
  type        = string
}

variable "bucket_name" {
  description = <<-EOT
    S3 bucket name for SageMaker training data. Must be globally unique.
    Defaults to "fraud-detection-sagemaker-<account-id>" (null triggers
    that default in main.tf) -- both existing IAM policies in
    infra/aws_iam/ already scope S3 access to the
    "fraud-detection-sagemaker-*" prefix, so a custom value must keep
    that prefix or those policies need updating too.
  EOT
  type        = string
  default     = null
}

variable "lifecycle_expiration_days" {
  description = <<-EOT
    Days until objects in this bucket auto-expire. Session 6 originally
    scoped this rule to only the training-data/ prefix and left code/ and
    output/ (written by training/sagemaker_launch.py) uncovered indefinitely
    -- see docs/sagemaker.md's "Known gap, not fixed" note. This module
    applies the rule to the whole bucket instead, closing that gap.
  EOT
  type        = number
  default     = 14
}

variable "sagemaker_execution_role_name" {
  description = <<-EOT
    Must stay exactly "fraud-detection-sagemaker-execution" unless you also
    update infra/aws_iam/cli_user_permissions_policy.json's iam:PassRole
    Resource ARN, which is scoped to this exact role name.
  EOT
  type        = string
  default     = "fraud-detection-sagemaker-execution"
}

variable "tags" {
  description = "Tags applied to every resource this module creates."
  type        = map(string)
  default = {
    Project = "fraud-detection"
    Purpose = "sagemaker-training"
  }
}
