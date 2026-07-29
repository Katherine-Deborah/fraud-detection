"""Basic EDA on the generated synthetic transaction dataset.

Prints class balance and per-feature summary stats (fraud vs non-fraud),
and saves a few sanity-check plots to docs/eda/.

Usage:
    python data_generation/eda.py --data data/raw/transactions.parquet
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NUMERIC_FEATURES = [
    "amount", "hour_of_day", "day_of_week", "is_weekend",
    "time_since_last_txn_sec", "txn_count_last_1h", "txn_count_last_24h",
    "avg_amount_last_5_txns", "amount_to_avg_ratio",
    "geo_distance_from_last_txn_km", "home_distance_km",
    "is_new_device", "is_new_merchant_category",
    "cumulative_distinct_devices", "cumulative_distinct_merchant_categories",
    "account_txn_seq_num",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/raw/transactions.parquet")
    parser.add_argument("--plots-out", type=str, default="docs/eda")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    os.makedirs(args.plots_out, exist_ok=True)

    n = len(df)
    n_fraud = int(df["is_fraud"].sum())
    print(f"Total rows: {n:,}")
    print(f"Fraud rows: {n_fraud:,} ({n_fraud / n:.4%})")
    print(f"Unique accounts: {df['account_id'].nunique():,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    print("Per-feature mean, fraud vs non-fraud:")
    summary = df.groupby("is_fraud")[NUMERIC_FEATURES].mean().T
    summary.columns = ["non_fraud_mean", "fraud_mean"]
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))
    print()

    print("Fraud rate by merchant category:")
    print(df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False).to_string(float_format=lambda x: f"{x:.4%}"))

    # class balance bar chart
    fig, ax = plt.subplots(figsize=(4, 4))
    df["is_fraud"].value_counts().rename({False: "legitimate", True: "fraud"}).plot.bar(ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_title("Class balance (log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots_out, "class_balance.png"), dpi=120)
    plt.close(fig)

    # amount distribution fraud vs non-fraud
    fig, ax = plt.subplots(figsize=(6, 4))
    df.loc[~df["is_fraud"], "amount"].clip(upper=2000).hist(bins=60, alpha=0.6, label="legitimate", density=True, ax=ax)
    df.loc[df["is_fraud"], "amount"].clip(upper=2000).hist(bins=60, alpha=0.6, label="fraud", density=True, ax=ax)
    ax.set_title("Transaction amount distribution (clipped at $2,000)")
    ax.set_xlabel("amount ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots_out, "amount_distribution.png"), dpi=120)
    plt.close(fig)

    # velocity feature fraud vs non-fraud
    fig, ax = plt.subplots(figsize=(6, 4))
    df.loc[~df["is_fraud"], "txn_count_last_1h"].hist(bins=range(0, 12), alpha=0.6, label="legitimate", density=True, ax=ax)
    df.loc[df["is_fraud"], "txn_count_last_1h"].hist(bins=range(0, 12), alpha=0.6, label="fraud", density=True, ax=ax)
    ax.set_title("Transactions in trailing 1h, fraud vs legitimate")
    ax.set_xlabel("txn_count_last_1h")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.plots_out, "velocity_distribution.png"), dpi=120)
    plt.close(fig)

    print(f"\nSaved plots to {args.plots_out}/")


if __name__ == "__main__":
    main()
