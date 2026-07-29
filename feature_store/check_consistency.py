"""Confirms that features retrievable from the Feast *online* store (Redis,
what serving would use) match what training would see from the *offline*
source (data/raw/transactions.parquet) for the same account. Per
PROJECT.md Session 3: "this consistency check is the point of the whole
layer, don't skip it."

For a given account, the online store's current snapshot might reflect
either (a) the initial batch materialization (feature_store/materialize.py
-- exact values copied from the offline parquet), or (b) a subsequent
real-time push from the Kafka consumer (feature_store/online_features.py
-- geo features recomputed from city-centroid coordinates, everything
else recomputed to match offline exactly). Rather than assume which one
happened, this script uses `account_txn_seq_num` -- itself one of the 15
features -- to find the exact offline row the online snapshot corresponds
to (that account's Nth transaction), then compares every field against
that specific row. Geo features (`geo_distance_from_last_txn_km`,
`home_distance_km`) are checked with a tolerance instead of exact
equality; see the module docstring in online_features.py for why.

Usage:
    python feature_store/check_consistency.py --account-id acct_000000
    python feature_store/check_consistency.py --sample 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from feast import FeatureStore

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_REPO = REPO_ROOT / "feature_store" / "feature_repo"
TRANSACTIONS_PATH = REPO_ROOT / "data" / "raw" / "transactions.parquet"

FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend", "time_since_last_txn_sec",
    "txn_count_last_1h", "txn_count_last_24h", "avg_amount_last_5_txns",
    "amount_to_avg_ratio", "geo_distance_from_last_txn_km", "home_distance_km",
    "is_new_device", "is_new_merchant_category", "cumulative_distinct_devices",
    "cumulative_distinct_merchant_categories", "account_txn_seq_num",
]
GEO_FIELDS = {"geo_distance_from_last_txn_km", "home_distance_km"}
GEO_TOLERANCE_KM = 20.0


def check_account(store: FeatureStore, offline_df: pd.DataFrame, account_id: str) -> tuple[bool, str]:
    resp = store.get_online_features(
        features=[f"account_transaction_features:{f}" for f in FEATURES],
        entity_rows=[{"account_id": account_id}],
    ).to_dict()

    if resp["account_txn_seq_num"][0] is None:
        return False, f"{account_id}: no online features found (never materialized or streamed)"

    online_seq = int(resp["account_txn_seq_num"][0])
    acct_rows = offline_df[offline_df.account_id == account_id]
    target = acct_rows[acct_rows.account_txn_seq_num == online_seq]
    if len(target) != 1:
        return False, f"{account_id}: no unique offline row with account_txn_seq_num={online_seq}"
    target = target.iloc[0]

    mismatches = []
    for field in FEATURES:
        online_val = resp[field][0]
        offline_val = target[field]
        if field in GEO_FIELDS:
            ok = offline_val == -1 and online_val == -1 or abs(float(online_val) - float(offline_val)) <= GEO_TOLERANCE_KM
        else:
            ok = abs(float(online_val) - float(offline_val)) <= max(1e-4, abs(float(offline_val)) * 1e-6)
        if not ok:
            mismatches.append(f"{field}: online={online_val} offline={offline_val}")

    if mismatches:
        return False, f"{account_id} (txn_seq={online_seq}): " + "; ".join(mismatches)
    return True, f"{account_id} (txn_seq={online_seq}): all {len(FEATURES)} features match"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--account-id", help="Check a single account")
    p.add_argument("--sample", type=int, default=0, help="Check N randomly sampled accounts instead")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.account_id and not args.sample:
        args.sample = 50

    store = FeatureStore(repo_path=str(FEATURE_REPO))
    offline_df = pd.read_parquet(TRANSACTIONS_PATH)

    if args.account_id:
        account_ids = [args.account_id]
    else:
        account_ids = list(
            offline_df["account_id"].drop_duplicates().sample(args.sample, random_state=args.seed)
        )

    passed = 0
    for account_id in account_ids:
        ok, msg = check_account(store, offline_df, account_id)
        print(("PASS  " if ok else "FAIL  ") + msg)
        passed += ok

    print(f"\n{passed}/{len(account_ids)} accounts consistent between online and offline stores.")
    if passed != len(account_ids):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
