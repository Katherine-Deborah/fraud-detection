"""Real-time computation of the same 15 engineered features as
`data_generation/generate_transactions.py::compute_engineered_features`,
applied incrementally as transactions arrive from Kafka (one record at a
time, not a batch groupby-and-shift over the whole history).

Per-account rolling state (last transaction's time/location, last 5
amounts, seen devices/categories, transaction sequence number) is kept in
Redis, not an in-memory dict -- `docs/kafka.md` flagged in-memory
out-of-order tracking as something that "would need to be persisted (e.g.
in Redis, once Session 3 exists)" for a consumer restart not to lose it.
This uses the *same* Redis container as the Feast online store but a
different logical DB index (db=1 vs Feast's db=0) so this engine's
scratch state never shares key space with Feast's own serialized feature
keys.

One deliberate gap from full offline/online parity: `geo_distance_from_
last_txn_km` and `home_distance_km` were computed offline from small
per-transaction random jitter around each city's coordinates -- jitter
that exists purely so synthetic transactions in the same city don't all
land on one exact point; it carries no fraud signal. The Kafka message
only carries the city name (`location`), not that jitter, so this module
resolves `location` to its city-centroid coordinates instead (the same
20-city table the generator uses). Online geo features will therefore
differ from the offline (jittered) values by up to roughly the jitter
magnitude -- a few km, not zero. `feature_store/check_consistency.py`
checks the geo features with a tolerance for exactly this reason, while
every other feature is checked for an exact match.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import redis  # noqa: E402

from data_generation.generate_transactions import (  # noqa: E402
    CITY_LATS,
    CITY_LONS,
    CITY_NAMES,
    haversine_km,
)

CITY_COORDS = {
    name: (lat, lon) for name, lat, lon in zip(CITY_NAMES, CITY_LATS, CITY_LONS)
}

STATE_DB = 1  # separate from Feast's online store, which lives in db 0
AMOUNT_HISTORY = 5
ONE_HOUR_SEC = 3600
ONE_DAY_SEC = 86400


class OnlineFeatureEngine:
    def __init__(self, redis_host: str, redis_port: int, accounts_path: Path | str):
        self.r = redis.Redis(host=redis_host, port=redis_port, db=STATE_DB, decode_responses=True)
        accounts = pd.read_parquet(accounts_path)
        self._home = {
            row.account_id: (float(row.home_lat), float(row.home_lon))
            for row in accounts.itertuples(index=False)
        }

    def compute(self, record: dict) -> dict:
        """Given one validated raw transaction record (the same shape the
        Kafka consumer already validates -- see consumer/consume.py's
        REQUIRED_FIELDS), returns the 15 engineered features plus the
        entity key and event timestamp, ready to push to Feast."""
        account_id = record["account_id"]
        ts = pd.Timestamp(record["timestamp"])
        ts_epoch = ts.timestamp()
        amount = float(record["amount"])
        device_id = record["device_id"]
        category = record["merchant_category"]
        lat, lon = CITY_COORDS.get(record["location"], (None, None))
        if lat is not None:
            lat, lon = float(lat), float(lon)

        p = f"fstate:{account_id}:"
        pipe = self.r.pipeline()

        last = self.r.hgetall(p + "last")
        if last:
            time_since_last_txn_sec = ts_epoch - float(last["ts"])
            if lat is not None and last.get("lat") not in (None, ""):
                geo_distance_from_last_txn_km = float(
                    haversine_km(lat, lon, float(last["lat"]), float(last["lon"]))
                )
            else:
                geo_distance_from_last_txn_km = -1.0
        else:
            time_since_last_txn_sec = -1.0
            geo_distance_from_last_txn_km = -1.0

        if lat is not None:
            pipe.hset(p + "last", mapping={"ts": ts_epoch, "lat": lat, "lon": lon})

        prev_amounts = self.r.lrange(p + "amounts", 0, AMOUNT_HISTORY - 1)
        if prev_amounts:
            avg_amount_last_5_txns = sum(float(a) for a in prev_amounts) / len(prev_amounts)
        else:
            avg_amount_last_5_txns = amount
        amount_to_avg_ratio = amount / avg_amount_last_5_txns if avg_amount_last_5_txns else 1.0
        pipe.lpush(p + "amounts", amount)
        pipe.ltrim(p + "amounts", 0, AMOUNT_HISTORY - 1)

        events_key = p + "events"
        txn_count_last_1h = self.r.zcount(events_key, ts_epoch - ONE_HOUR_SEC, ts_epoch)
        txn_count_last_24h = self.r.zcount(events_key, ts_epoch - ONE_DAY_SEC, ts_epoch)
        # current record counts toward its own windows
        txn_count_last_1h += 1
        txn_count_last_24h += 1
        pipe.zadd(events_key, {record["transaction_id"]: ts_epoch})
        pipe.zremrangebyscore(events_key, "-inf", ts_epoch - ONE_DAY_SEC)

        is_new_device = 0 if self.r.sismember(p + "devices", device_id) else 1
        is_new_merchant_category = 0 if self.r.sismember(p + "categories", category) else 1
        pipe.sadd(p + "devices", device_id)
        pipe.sadd(p + "categories", category)

        pipe.execute()
        cumulative_distinct_devices = self.r.scard(p + "devices")
        cumulative_distinct_merchant_categories = self.r.scard(p + "categories")
        account_txn_seq_num = self.r.incr(p + "seq")

        home_lat, home_lon = self._home.get(account_id, (None, None))
        if lat is not None and home_lat is not None:
            home_distance_km = float(haversine_km(lat, lon, home_lat, home_lon))
        else:
            home_distance_km = -1.0

        return {
            "account_id": account_id,
            # Must be named "timestamp", not "event_timestamp" -- Feast's
            # push path looks up the column named by the underlying batch
            # source's `timestamp_field` (transactions_source in
            # feature_store/feature_repo/definitions.py sets that to
            # "timestamp" to match the offline parquet's column name).
            "timestamp": ts,
            "hour_of_day": ts.hour,
            "day_of_week": ts.dayofweek,
            "is_weekend": int(ts.dayofweek >= 5),
            "time_since_last_txn_sec": float(time_since_last_txn_sec),
            "txn_count_last_1h": float(txn_count_last_1h),
            "txn_count_last_24h": float(txn_count_last_24h),
            "avg_amount_last_5_txns": float(avg_amount_last_5_txns),
            "amount_to_avg_ratio": float(amount_to_avg_ratio),
            "geo_distance_from_last_txn_km": float(geo_distance_from_last_txn_km),
            "home_distance_km": float(home_distance_km),
            "is_new_device": is_new_device,
            "is_new_merchant_category": is_new_merchant_category,
            "cumulative_distinct_devices": int(cumulative_distinct_devices),
            "cumulative_distinct_merchant_categories": int(cumulative_distinct_merchant_categories),
            "account_txn_seq_num": int(account_txn_seq_num),
        }
