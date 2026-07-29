# Feature Store (Feast + Redis)

Session 3 adds a feature store so that training (offline) and serving
(online) read the *same* feature definitions and, for a given account, the
same values. Read this before Session 4+ rather than re-deriving how the
pieces fit together.

## Why Feast Push Sources

A `FeatureView` in Feast is backed by one source. To get both an offline
path (train against `data/raw/transactions.parquet`) and an online path fed
by Kafka in real time, the source is a **Push Source**
(`feature_store/feature_repo/definitions.py`) wrapping a `FileSource`
pointed at the parquet file:

- **Batch/offline:** `feature_store/materialize.py` runs
  `store.materialize(start, end)`, which reads the underlying `FileSource`
  and writes the *latest* row per `account_id` into Redis. This is the
  "training-time snapshot" — everything the offline dataset already knows,
  in bulk.
- **Streaming/online:** the Kafka consumer (`consumer/consume.py`) computes
  features for each arriving transaction and calls `store.push(...)`,
  which writes straight into the online store, keyed by the same
  `account_id`.

Both paths write into the same Redis-backed online store using the same
feature view, which is the actual point of a feature store: a
`/predict` call and a training job resolve `account_transaction_features`
for an account the same way regardless of which path last updated it.

## Redis: two logical databases, one container

`docker-compose.yml`'s `redis` service backs both:

- **db 0** — Feast's own online store (its serialized keys, written by
  `materialize()` / `push()`).
- **db 1** — `feature_store/online_features.py`'s own per-account rolling
  state (last transaction's time/location, last 5 amounts, seen
  devices/categories, sequence counter) needed to compute features
  incrementally. This is scratch state for the *computation*, not a
  feature store value itself, so it's kept out of Feast's key space
  entirely rather than risking a collision.

**Port note:** the container is mapped to host port **6380**, not Redis's
default 6379. This machine's WSL Ubuntu distro already runs its own
unrelated `redis-server` on `127.0.0.1:6379` (predates this project), and
WSL2's localhost-forwarding silently routed Windows-side clients (both
Feast's Redis client and plain `redis-py` scripts) to *that* instance
instead of this container's — confirmed by writing a probe key in one and
reading it from the other. Rather than depend on that other process never
being there, `feature_store/feature_repo/feature_store.yaml`'s
`connection_string` and every script in this doc use `localhost:6380`.
If you hit `dbsize()` mysteriously not matching `docker exec fraud-redis
redis-cli DBSIZE`, this port collision is almost certainly why — check
`netstat -ano | grep 6379` for a second listener before assuming Feast is
broken.

## Why `pandas<3` now

`feast==0.65.0` requires `pandas<3,>=1.4.3`; the project had `pandas==3.0.5`
pinned since Session 1. Downgraded to `pandas==2.3.3` (the latest `<3`
release) in `requirements.txt` and regenerated the dataset under it —
identical row count and fraud rate (5,010,000 rows, 0.1996%) to the
original 3.0.5 run, so no behavior changed.

## `accounts.parquet`: closing a data gap

`home_distance_km` needs each account's home city coordinates, but the
generator computed those in-memory and dropped them before writing
`transactions.parquet` (only the per-transaction `location` city name
survived). Fixed by having
`data_generation/generate_transactions.py` also write
`data/raw/accounts.parquet` (`account_id`, `home_city`, `home_lat`,
`home_lon`) and regenerating. `feature_store/online_features.py` loads
this as a static reference table at startup.

## Real-time feature computation (`feature_store/online_features.py`)

`OnlineFeatureEngine.compute()` reproduces all 15 engineered features from
`data_generation/generate_transactions.py::compute_engineered_features`,
one transaction at a time, using the Redis (db 1) state described above.
Verified by replaying full per-account transaction histories (including
accounts hit by all three injected fraud patterns) through the engine and
diffing every field against the offline-computed values — see "Consistency
check results" below.

**Geo-feature tolerance, not exact match.** Offline, `geo_distance_from_
last_txn_km` and `home_distance_km` were computed from small random jitter
added to each city's coordinates — jitter that exists only so synthetic
transactions in the same city don't all land on one exact point; it
carries no fraud signal. The Kafka message only carries the city name
(`location`), not that jitter (a real system would face the same
limitation geocoding a merchant city rather than reading device GPS), so
the online engine resolves `location` to city-centroid coordinates
instead. Every other feature matches exactly; these two are checked with a
20km tolerance in `check_consistency.py`, comfortably above the jitter
magnitude (~a few km) and still tight enough to catch a real bug.

**Not exactly-once.** Like the Session 2 consumer's at-least-once Kafka
delivery, `OnlineFeatureEngine` state updates are not idempotent — replaying
the same transaction twice (e.g. after a crash between processing and
offset commit) double-counts it into the rolling windows and counters.
This surfaced directly during development: rerunning the consumer against
the same topic offsets under a new consumer group (for testing) reprocessed
2,000 messages three times over, visibly inflating `account_txn_seq_num`
and the velocity counters before the state was reset. Production-grade
would need idempotent updates (e.g. a per-account, per-transaction_id
dedupe set) — out of scope here, but worth knowing before trusting a
long-running consumer's numbers after a crash/restart.

## Error handling in the consumer

A feature-store push failure is logged and counted (`feature_push_
failures` in the consumer's log lines) but does **not** stop ingestion —
the data lake write already durably captured the record; a missed push is
recoverable by rerunning `feature_store/materialize.py`. This mirrors the
existing "don't let a downstream concern block raw ingestion" philosophy
from `docs/kafka.md`.

## Running it

```bash
docker compose up -d kafka redis

# one-time (or periodic) apply of the feature definitions
cd feature_store/feature_repo && feast apply && cd ../..

# batch: seed the online store from the full offline dataset
python feature_store/materialize.py

# streaming: keep it fresh as transactions arrive
python consumer/consume.py &
python producer/produce.py --limit 5000 --rate 200

# confirm online == offline for a sample of accounts
python feature_store/check_consistency.py --sample 200
python feature_store/check_consistency.py --account-id acct_000000
```

## Consistency check results

Run against the committed dataset after materializing all 50,000 accounts
and streaming 2,000 transactions through Kafka:

- **200 randomly sampled accounts** (mostly materialize-only, i.e. never
  touched by the streamed subset): **200/200 pass**, all 15 features exact
  (including geo — materialize copies the offline row's stored jittered
  values verbatim, so there's no jitter mismatch to tolerate here).
- **5 accounts specifically touched by the Kafka streaming path** (i.e.
  their online snapshot was overwritten by `OnlineFeatureEngine` +
  `store.push`, not just `materialize`): **5/5 pass**, all 15 features
  within tolerance (geo features matched within a few km, as expected from
  the jitter gap above).

`check_consistency.py` uses `account_txn_seq_num` (itself one of the 15
features) to find which offline row an online snapshot corresponds to, so
it works correctly whether that account was last updated by `materialize`
or by a live push — see the script's docstring for why.

## What's stubbed for later sessions

- Feature *values* aren't validated for plausibility anywhere yet — that's
  Great Expectations' job in the Session 4 DAG.
- The online engine's state (Redis db 1) has no eviction/TTL policy beyond
  the 24h event-window trim; long-running in production would need a
  retention policy for the devices/categories sets too.
