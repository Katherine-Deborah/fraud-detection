output "bucket_name" {
  description = "Pass to training/sagemaker_upload_data.py --bucket and training/sagemaker_launch.py."
  value       = aws_s3_bucket.training_data.bucket
}

output "sagemaker_execution_role_arn" {
  description = "Pass to training/sagemaker_launch.py --role-arn."
  value       = aws_iam_role.sagemaker_execution.arn
}
