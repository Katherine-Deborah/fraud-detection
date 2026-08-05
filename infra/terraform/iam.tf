# SageMaker execution role only -- not the CLI user (infra/aws_iam/README.md's
# other IAM principal). The CLI user holds long-lived access keys created
# once, by hand, through the console (see that README's "Console steps");
# managing IAM Users/access keys as Terraform state is a real secret-
# handling liability for a portfolio project's state file, and the task
# scope here is "the SageMaker training job resources and the S3 bucket" --
# the CLI user is an operator identity, not a training-job resource. Reuses
# the exact trust/permissions policy JSON already reviewed and tested live
# in Session 6, via file(), rather than re-authoring them as a second,
# driftable copy in HCL.
#
# Also not a Terraform resource: the training job itself. The AWS provider
# has no `aws_sagemaker_training_job` resource -- Terraform models durable
# infrastructure, and a training job is a one-shot, imperative action with
# a start/end time and a result artifact, not a long-lived resource with a
# desired steady state. training/sagemaker_launch.py (Session 6) remains
# how a job actually gets launched, using the role/bucket this module
# creates.

resource "aws_iam_role" "sagemaker_execution" {
  name               = var.sagemaker_execution_role_name
  assume_role_policy = file("${path.module}/../aws_iam/sagemaker_execution_trust_policy.json")
  tags               = var.tags
}

resource "aws_iam_role_policy" "sagemaker_execution_permissions" {
  name   = "fraud-detection-sagemaker-permissions"
  role   = aws_iam_role.sagemaker_execution.id
  policy = file("${path.module}/../aws_iam/sagemaker_execution_permissions_policy.json")
}
