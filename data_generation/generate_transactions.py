"""Synthetic credit-card transaction dataset generator.

Produces a parquet file of transactions with realistic (non-i.i.d.) fraud
patterns for the fraud detection pipeline project. See docs/dataset.md for
the schema, methodology, and the actual numbers from the committed run.

Usage:
    python data_generation/generate_transactions.py --n-rows 5000000 --out data/raw/transactions.parquet
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "dining", "gas_station", "online_retail", "electronics",
    "travel", "entertainment", "utilities", "pharmacy", "clothing",
    "jewelry", "cash_advance",
]
# lognormal (mean_log-ish center in dollars, sigma) per category
CATEGORY_AMOUNT_PARAMS = {
    "grocery": (45, 0.5), "dining": (35, 0.5), "gas_station": (40, 0.3),
    "online_retail": (80, 0.7), "electronics": (250, 0.8), "travel": (400, 0.9),
    "entertainment": (60, 0.6), "utilities": (120, 0.3), "pharmacy": (30, 0.5),
    "clothing": (70, 0.6), "jewelry": (600, 0.9), "cash_advance": (300, 0.7),
}
# categories overrepresented in the "new device + high amount" fraud pattern
HIGH_RISK_CATEGORIES = ["electronics", "jewelry", "cash_advance", "online_retail"]

CITIES = [
    ("New York", 40.7128, -74.0060), ("Los Angeles", 34.0522, -118.2437),
    ("Chicago", 41.8781, -87.6298), ("Houston", 29.7604, -95.3698),
    ("Phoenix", 33.4484, -112.0740), ("Philadelphia", 39.9526, -75.1652),
    ("San Antonio", 29.4241, -98.4936), ("San Diego", 32.7157, -117.1611),
    ("Dallas", 32.7767, -96.7970), ("San Jose", 37.3382, -121.8863),
    ("Austin", 30.2672, -97.7431), ("Jacksonville", 30.3322, -81.6557),
    ("San Francisco", 37.7749, -122.4194), ("Columbus", 39.9612, -82.9988),
    ("Charlotte", 35.2271, -80.8431), ("Indianapolis", 39.7684, -86.1581),
    ("Seattle", 47.6062, -122.3321), ("Denver", 39.7392, -104.9903),
    ("Boston", 42.3601, -71.0589), ("Miami", 25.7617, -80.1918),
]
CITY_NAMES = np.array([c[0] for c in CITIES])
CITY_LATS = np.array([c[1] for c in CITIES])
CITY_LONS = np.array([c[2] for c in CITIES])

PERIOD_START = pd.Timestamp("2025-01-01")
PERIOD_END = pd.Timestamp("2025-12-31 23:59:59")
PERIOD_SECONDS = (PERIOD_END - PERIOD_START).total_seconds()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def generate_accounts(n_accounts: int, rng: np.random.Generator) -> pd.DataFrame:
    city_idx = rng.integers(0, len(CITIES), size=n_accounts)
    home_lat = CITY_LATS[city_idx] + rng.normal(0, 0.05, n_accounts)
    home_lon = CITY_LONS[city_idx] + rng.normal(0, 0.05, n_accounts)
    spend_multiplier = rng.lognormal(mean=0.0, sigma=0.4, size=n_accounts)
    # overdispersed activity level per account (some customers transact far more than others)
    activity_weight = rng.gamma(shape=2.0, scale=1.0, size=n_accounts)
    return pd.DataFrame({
        "account_id": [f"acct_{i:06d}" for i in range(n_accounts)],
        "home_city_idx": city_idx,
        "home_city": CITY_NAMES[city_idx],
        "home_lat": home_lat,
        "home_lon": home_lon,
        "spend_multiplier": spend_multiplier,
        "activity_weight": activity_weight,
    })


def generate_base_transactions(accounts: pd.DataFrame, n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    n_accounts = len(accounts)
    if n_rows < n_accounts:
        raise ValueError(
            f"n_rows ({n_rows}) must be >= n_accounts ({n_accounts}) -- "
            "every account gets at least one transaction."
        )
    weights = accounts["activity_weight"].to_numpy()
    n_i = np.maximum(1, np.round(weights / weights.sum() * n_rows)).astype(int)
    # Trim/pad rounding error so the total matches n_rows exactly. Only ever
    # add to accounts (always safe) or remove from accounts that have room
    # above the 1-per-account floor -- decrementing a random account already
    # at 1 and then re-clamping it back up to 1 (the previous approach)
    # could silently leave the final total short of n_rows by however many
    # accounts got clamped.
    diff = n_rows - n_i.sum()
    if diff > 0:
        idx = rng.integers(0, n_accounts, size=diff)
        np.add.at(n_i, idx, 1)
    elif diff < 0:
        to_remove = -diff
        while to_remove > 0:
            candidates = np.where(n_i > 1)[0]
            take = min(to_remove, len(candidates))
            idx = rng.choice(candidates, size=take, replace=False)
            n_i[idx] -= 1
            to_remove -= take

    account_idx = np.repeat(np.arange(n_accounts), n_i)
    total = len(account_idx)

    timestamps = PERIOD_START + pd.to_timedelta(rng.uniform(0, PERIOD_SECONDS, size=total), unit="s")
    category_idx = rng.integers(0, len(MERCHANT_CATEGORIES), size=total)
    categories = np.array(MERCHANT_CATEGORIES)[category_idx]

    cat_mean = np.array([CATEGORY_AMOUNT_PARAMS[c][0] for c in categories])
    cat_sigma = np.array([CATEGORY_AMOUNT_PARAMS[c][1] for c in categories])
    spend_mult = accounts["spend_multiplier"].to_numpy()[account_idx]
    amounts = rng.lognormal(mean=0, sigma=cat_sigma, size=total) * cat_mean * spend_mult
    amounts = np.round(amounts, 2)

    # ~3% of transactions are legitimate travel: a different random city, not the account's home
    is_travel = rng.random(total) < 0.03
    travel_city_idx = rng.integers(0, len(CITIES), size=total)
    home_city_idx = accounts["home_city_idx"].to_numpy()[account_idx]
    city_idx = np.where(is_travel, travel_city_idx, home_city_idx)
    location = CITY_NAMES[city_idx]
    home_lat = accounts["home_lat"].to_numpy()[account_idx]
    home_lon = accounts["home_lon"].to_numpy()[account_idx]
    lat = np.where(is_travel, CITY_LATS[travel_city_idx], home_lat) + rng.normal(0, 0.03, total)
    lon = np.where(is_travel, CITY_LONS[travel_city_idx], home_lon) + rng.normal(0, 0.03, total)

    # device evolution: small per-transaction chance an account starts using a new device
    is_new_device_event = rng.random(total) < 0.005

    df = pd.DataFrame({
        "account_id": accounts["account_id"].to_numpy()[account_idx],
        "timestamp": timestamps,
        "amount": amounts,
        "merchant_category": categories,
        "location": location,
        "lat": lat,
        "lon": lon,
        "_new_device_event": is_new_device_event,
    })
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    device_num = df.groupby("account_id")["_new_device_event"].cumsum()
    device_num = device_num.where(df.groupby("account_id").cumcount() > 0, 0)  # first txn always device 0
    df["device_id"] = df["account_id"] + "_dev" + device_num.astype(int).astype(str)
    df["is_fraud"] = False
    return df.drop(columns="_new_device_event")


def inject_fraud(base: pd.DataFrame, accounts: pd.DataFrame, n_rows: int, fraud_rate: float,
                  rng: np.random.Generator) -> pd.DataFrame:
    target_fraud = max(10, round(n_rows * fraud_rate))
    n_burst_events = max(1, round(target_fraud * 0.4 / 8))
    n_geo = max(1, round(target_fraud * 0.3))
    n_newdev = max(1, target_fraud - n_burst_events * 8 - n_geo)

    acct_lookup = accounts.set_index("account_id")
    fraud_rows = []

    # Pattern A: velocity burst — several rapid-fire transactions on one account within minutes
    burst_accounts = rng.choice(accounts["account_id"].to_numpy(), size=n_burst_events, replace=False)
    for acct in burst_accounts:
        row = acct_lookup.loc[acct]
        start = PERIOD_START + pd.Timedelta(seconds=rng.uniform(0, PERIOD_SECONDS - 3600))
        k = 8
        offsets = np.sort(rng.uniform(0, 600, size=k))  # burst within 10 minutes
        cats = rng.choice(HIGH_RISK_CATEGORIES, size=k)
        for i in range(k):
            mean, sigma = CATEGORY_AMOUNT_PARAMS[cats[i]]
            amt = rng.lognormal(0, sigma) * mean * row["spend_multiplier"] * rng.uniform(2, 5)
            fraud_rows.append({
                "account_id": acct, "timestamp": start + pd.Timedelta(seconds=offsets[i]),
                "amount": round(amt, 2), "merchant_category": cats[i],
                "location": row["home_city"],
                "lat": row["home_lat"] + rng.normal(0, 0.05), "lon": row["home_lon"] + rng.normal(0, 0.05),
                "device_id": f"{acct}_dev0", "is_fraud": True,
            })

    # Pattern B: geo-impossible — a transaction far from home, minutes after a normal one
    geo_accounts = rng.choice(accounts["account_id"].to_numpy(), size=n_geo, replace=True)
    for acct in geo_accounts:
        row = acct_lookup.loc[acct]
        anchor_time = PERIOD_START + pd.Timedelta(seconds=rng.uniform(0, PERIOD_SECONDS - 3600))
        far_city_idx = rng.integers(0, len(CITIES))
        cat = rng.choice(MERCHANT_CATEGORIES)
        mean, sigma = CATEGORY_AMOUNT_PARAMS[cat]
        amt = rng.lognormal(0, sigma) * mean * row["spend_multiplier"]
        fraud_rows.append({
            "account_id": acct, "timestamp": anchor_time + pd.Timedelta(minutes=rng.uniform(5, 45)),
            "amount": round(amt, 2), "merchant_category": cat,
            "location": CITY_NAMES[far_city_idx],
            "lat": CITY_LATS[far_city_idx] + rng.normal(0, 0.05),
            "lon": CITY_LONS[far_city_idx] + rng.normal(0, 0.05),
            "device_id": f"{acct}_dev0", "is_fraud": True,
        })

    # Pattern C: brand-new device + abnormally high amount
    newdev_accounts = rng.choice(accounts["account_id"].to_numpy(), size=n_newdev, replace=True)
    for acct in newdev_accounts:
        row = acct_lookup.loc[acct]
        cat = rng.choice(HIGH_RISK_CATEGORIES)
        mean, sigma = CATEGORY_AMOUNT_PARAMS[cat]
        amt = rng.lognormal(0, sigma) * mean * row["spend_multiplier"] * rng.uniform(4, 8)
        fraud_rows.append({
            "account_id": acct, "timestamp": PERIOD_START + pd.Timedelta(seconds=rng.uniform(0, PERIOD_SECONDS)),
            "amount": round(amt, 2), "merchant_category": cat,
            "location": row["home_city"],
            "lat": row["home_lat"] + rng.normal(0, 0.05), "lon": row["home_lon"] + rng.normal(0, 0.05),
            "device_id": f"{acct}_devFRAUD{rng.integers(0, 1_000_000)}", "is_fraud": True,
        })

    return pd.DataFrame(fraud_rows)


def compute_engineered_features(df: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    grp = df.groupby("account_id", sort=False)

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    prev_time = grp["timestamp"].shift(1)
    df["time_since_last_txn_sec"] = (df["timestamp"] - prev_time).dt.total_seconds().fillna(-1)

    prev_lat = grp["lat"].shift(1)
    prev_lon = grp["lon"].shift(1)
    dist = haversine_km(df["lat"], df["lon"], prev_lat, prev_lon)
    df["geo_distance_from_last_txn_km"] = np.where(prev_lat.isna(), -1, dist)

    home = accounts.set_index("account_id")[["home_lat", "home_lon"]]
    home_lat = df["account_id"].map(home["home_lat"])
    home_lon = df["account_id"].map(home["home_lon"])
    df["home_distance_km"] = haversine_km(df["lat"], df["lon"], home_lat, home_lon)

    df["avg_amount_last_5_txns"] = grp["amount"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    df["avg_amount_last_5_txns"] = df["avg_amount_last_5_txns"].fillna(df["amount"])
    df["amount_to_avg_ratio"] = df["amount"] / df["avg_amount_last_5_txns"].replace(0, np.nan)
    df["amount_to_avg_ratio"] = df["amount_to_avg_ratio"].fillna(1.0)

    count_1h = grp.rolling("1h", on="timestamp")["amount"].count()
    count_24h = grp.rolling("24h", on="timestamp")["amount"].count()
    # groupby-rolling's result is only correct to assign back positionally
    # (via .to_numpy(), below) because its row order happens to match df's
    # row order exactly -- true because df is sorted by (account_id,
    # timestamp) immediately above and groupby(sort=False) preserves that
    # order. That's an invariant of *this* code, not something pandas
    # guarantees in general, so assert it explicitly: if a future change
    # ever breaks the assumption, this fails loudly here instead of
    # silently misaligning txn_count_last_1h/24h with the rest of the row.
    assert (count_1h.index.get_level_values(1).to_numpy() == df["timestamp"].to_numpy()).all(), (
        "groupby-rolling result no longer aligns positionally with df -- "
        "txn_count_last_1h/24h assignment below would be silently wrong"
    )
    df["txn_count_last_1h"] = count_1h.to_numpy()
    df["txn_count_last_24h"] = count_24h.to_numpy()

    is_new_device = (grp["device_id"].transform(lambda s: ~s.duplicated())).astype(int)
    is_new_category = (grp["merchant_category"].transform(lambda s: ~s.duplicated())).astype(int)
    df["is_new_device"] = is_new_device
    df["is_new_merchant_category"] = is_new_category
    df["cumulative_distinct_devices"] = grp["device_id"].transform(
        lambda s: (~s.duplicated()).cumsum()
    )
    df["cumulative_distinct_merchant_categories"] = grp["merchant_category"].transform(
        lambda s: (~s.duplicated()).cumsum()
    )
    df["account_txn_seq_num"] = grp.cumcount() + 1

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=5_000_000)
    parser.add_argument("--n-accounts", type=int, default=50_000)
    parser.add_argument("--fraud-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/transactions.parquet")
    args = parser.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.n_accounts:,} accounts...")
    accounts = generate_accounts(args.n_accounts, rng)

    print(f"Generating {args.n_rows:,} base transactions...")
    base = generate_base_transactions(accounts, args.n_rows, rng)

    print(f"Injecting fraud patterns (target rate {args.fraud_rate:.3%})...")
    fraud = inject_fraud(base, accounts, args.n_rows, args.fraud_rate, rng)

    combined = pd.concat([base, fraud], ignore_index=True)
    combined = combined.sort_values(["account_id", "timestamp"]).reset_index(drop=True)

    print("Computing engineered features...")
    combined = compute_engineered_features(combined, accounts)

    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.insert(0, "transaction_id", [f"txn_{i:010d}" for i in range(len(combined))])
    combined = combined.drop(columns=["lat", "lon"])

    out_cols = [
        "transaction_id", "account_id", "timestamp", "amount", "merchant_category",
        "location", "device_id", "is_fraud",
        "hour_of_day", "day_of_week", "is_weekend",
        "time_since_last_txn_sec", "txn_count_last_1h", "txn_count_last_24h",
        "avg_amount_last_5_txns", "amount_to_avg_ratio",
        "geo_distance_from_last_txn_km", "home_distance_km",
        "is_new_device", "is_new_merchant_category",
        "cumulative_distinct_devices", "cumulative_distinct_merchant_categories",
        "account_txn_seq_num",
    ]
    combined = combined[out_cols]

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    combined.to_parquet(args.out, index=False)

    accounts_out = os.path.join(os.path.dirname(args.out), "accounts.parquet")
    accounts[["account_id", "home_city", "home_lat", "home_lon"]].to_parquet(accounts_out, index=False)

    fraud_rate_actual = combined["is_fraud"].mean()
    elapsed = time.time() - t0
    print(f"Wrote {len(combined):,} rows to {args.out}")
    print(f"Wrote {len(accounts):,} account home-location rows to {accounts_out}")
    print(f"Actual fraud rate: {fraud_rate_actual:.4%} ({combined['is_fraud'].sum():,} fraud rows)")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
