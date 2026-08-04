"""Create the AWS Budget alert required by PROJECT.md §7 rule 1, before any
SageMaker resource is touched. Idempotent -- safe to rerun (updates the
existing budget instead of erroring if it already exists).

Usage:
    python infra/aws_budget/setup_budget.py --email you@example.com --limit 10

Creates one monthly cost budget (default $10) with two email
notifications: an ACTUAL-spend alert at 80% of the limit, and a
FORECASTED-spend alert at 100% -- so you find out both when real spend is
getting close, and when AWS's forecast says you're on track to exceed it
before you actually do.
"""

from __future__ import annotations

import argparse

import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Email address to receive budget alerts")
    parser.add_argument("--limit", type=float, default=10.0, help="Monthly budget limit in USD")
    parser.add_argument("--budget-name", default="fraud-detection-sagemaker")
    args = parser.parse_args()

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    budgets = boto3.client("budgets")

    budget = {
        "BudgetName": args.budget_name,
        "BudgetLimit": {"Amount": f"{args.limit:.2f}", "Unit": "USD"},
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST",
    }
    notifications = [
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80.0,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": args.email}],
        },
        {
            "Notification": {
                "NotificationType": "FORECASTED",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 100.0,
                "ThresholdType": "PERCENTAGE",
            },
            "Subscribers": [{"SubscriptionType": "EMAIL", "Address": args.email}],
        },
    ]

    try:
        budgets.create_budget(
            AccountId=account_id, Budget=budget, NotificationsWithSubscribers=notifications
        )
        print(f"created budget '{args.budget_name}': ${args.limit:.2f}/month, alerts to {args.email}")
    except budgets.exceptions.DuplicateRecordException:
        budgets.update_budget(AccountId=account_id, NewBudget=budget)
        print(f"budget '{args.budget_name}' already existed -- limit updated to ${args.limit:.2f}/month")
        print("(notification thresholds are not modified on update; delete and rerun if you need to change those)")

    print("\nverify in the console: https://console.aws.amazon.com/billing/home#/budgets")


if __name__ == "__main__":
    main()
