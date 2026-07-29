# Kafka Ingestion Layer

Session 2 replaces the `docker-compose.yml` `hello-world` stub with a
single-node Kafka broker and adds a producer/consumer pair that streams the
Session 1 synthetic dataset through it.

## Broker setup

- **KRaft mode, no Zookeeper.** Kafka's own quorum controller replaces
  Zookeeper (Zookeeper is being phased out of Kafka entirely); running one
  container instead of two is simpler for a single-node local dev setup and
  is the direction the ecosystem has already moved.
- Image: `apache/kafka:3.9.0` (Apache's own image, not a third-party one).
- Single broker, single controller, `KAFKA_AUTO_CREATE_TOPICS_ENABLE=true`.
- Exposed on `localhost:9092`. The producer/consumer are plain local Python
  scripts (not containerized), so they connect to that host port directly —
  there was no need to solve container-to-container advertised-listener
  routing for this session, since nothing else in the stack talks to Kafka
  yet.
- No volume is mounted, so broker state (topics, committed offsets) is
  wiped every time the container is recreated (`docker compose down` /
  `up`) — fine for a local dev/demo loop, but means the "verified restart
  resumes from committed offset" behavior below only holds across a
  `docker compose stop`/`start` of the *same* container, not a full
  `down`/`up`.
- Topic: `transactions`, 3 partitions, replication factor 1 (single broker,
  so replication factor >1 isn't possible yet — revisit if/when the stack
  moves to a multi-broker setup). Auto-created on first produce, or manually:
  ```
  docker exec fraud-kafka /opt/kafka/bin/kafka-topics.sh \
    --bootstrap-server localhost:9092 \
    --create --topic transactions --partitions 3 --replication-factor 1
  ```

## Message schema (JSON, not Avro)

Chosen over Avro for this session to avoid standing up a Schema Registry
container just to get ingestion working — JSON keeps the infra surface
small while still being a fully documented, validated schema (see
`consumer/consume.py::REQUIRED_FIELDS`). Revisit Avro + Schema Registry
later only if schema evolution actually becomes a problem worth solving.

Only the **raw** transaction fields are streamed — not the 15 engineered
behavioral features from `docs/dataset.md`. Those features were computed
offline using an account's full transaction history (available because the
whole dataset already existed in a file); a real streaming producer
wouldn't have that future/full-history context. Feature computation on live
data is the feature store's job (Session 3), not the producer's.

```json
{
  "transaction_id": "txn_0000000000",
  "account_id": "acct_005127",
  "timestamp": "2025-01-01T00:00:03.942101",
  "amount": 179.92,
  "merchant_category": "electronics",
  "location": "Austin",
  "device_id": "acct_005127_dev0",
  "is_fraud": false
}
```

| Field | Type | Notes |
|---|---|---|
| `transaction_id` | string | |
| `account_id` | string | Used as the Kafka message **key**, so all of one account's events land on the same partition and stay in per-account order within that partition. |
| `timestamp` | string | ISO 8601, validated with `datetime.fromisoformat`. |
| `amount` | number | |
| `merchant_category` | string | |
| `location` | string | |
| `device_id` | string | |
| `is_fraud` | bool | Included even though a real-time system wouldn't know this at ingestion time — this is a synthetic-data replay whose purpose is to reproduce a labeled dataset end-to-end through the pipeline, not a live fraud feed. |

## Offset / replay strategy

- Consumer group: `fraud-lake-writer` (default).
- `auto.offset.reset=earliest` — only takes effect the *first* time a group
  has no committed offset. This makes a fresh demo (new group, or the topic
  re-created) replay from the beginning by default, which matters for
  reproducibility.
- `enable.auto.commit=False` — offsets are committed **manually**, and only
  *after* a record has been durably written to the data lake (or
  dead-lettered). This gives **at-least-once** delivery: a crash between
  writing and committing can cause the same message to be reprocessed on
  restart, but never silently dropped. Downstream consumers of the lake
  should dedupe by `transaction_id` if exact-once matters.
- To force a full replay for a demo without waiting for a new topic:
  either pass a new `--group-id`, or reset the existing group's offsets:
  ```
  docker exec fraud-kafka /opt/kafka/bin/kafka-consumer-groups.sh \
    --bootstrap-server localhost:9092 --group fraud-lake-writer \
    --reset-offsets --to-earliest --topic transactions --execute
  ```
- Verified restart behavior: running the consumer twice with the same
  group-id only processes messages produced *after* the first run's last
  committed offset — it does not re-land already-committed records.

## Error handling

| Scenario | Behavior |
|---|---|
| **Malformed message** (invalid JSON, missing required field, wrong field type, unparseable timestamp) | Logged as a warning, written to `data/lake/_dead_letter/dead_letter.jsonl` with the raw bytes, the error, and topic/partition/offset, then the offset is committed and the consumer moves on. It does **not** crash or block the stream. |
| **Consumer restart** | Resumes from the last committed offset (see above) — at-least-once, not at-most-once. |
| **Out-of-order timestamp** (a record arrives with an earlier timestamp than one already seen for the same `account_id`) | Logged as a warning and counted, but the record is **still landed** in the data lake — raw ingestion's job is to capture what arrived, not to enforce ordering. Per-account last-seen timestamps are tracked in memory only (reset on consumer restart); this is fine for a raw landing zone, but any future stateful ordering logic would need that state persisted (e.g. in Redis, once Session 3 exists), not kept in a local dict. |

The producer can deliberately inject malformed messages for testing via
`--inject-malformed-every N` (every Nth message is replaced with truncated
JSON, a wrong-typed field, or a message missing required fields).

## Data lake layout

Valid records land as JSON Lines, partitioned by the transaction's own date
(from its `timestamp` field, not wall-clock arrival time):

```
data/lake/
  dt=2025-01-01/transactions.jsonl
  dt=2025-01-02/transactions.jsonl
  ...
  _dead_letter/dead_letter.jsonl
```

`data/lake/` is gitignored, same as `data/raw/` — it's regenerated by
running the producer and consumer, not committed.

## Running it

```bash
docker compose up -d kafka

# consumer first (or in another terminal) — it will wait for messages
python consumer/consume.py

# replay 5000 transactions at 200/sec, with malformed messages injected for testing
python producer/produce.py --limit 5000 --rate 200 --inject-malformed-every 250
```

Both scripts default to `localhost:9092`; see `--help` on each for the full
list of options (rate, topic, group-id, output dir, offset-reset policy,
idle-timeout, etc).

## What's stubbed for later sessions

`consumer/consume.py::forward_to_feature_engineering()` is a documented
no-op today. Session 3 replaces it with a write into the Feast/Redis online
feature store, which is the actual point of that layer.
