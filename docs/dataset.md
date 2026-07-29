# Synthetic Transaction Dataset

**This is a synthetic dataset generated for a portfolio project. It does not
contain, derive from, or represent any real financial data or real
individuals.** All accounts, devices, transactions, and fraud events are
fabricated by `data_generation/generate_transactions.py`.

## How to regenerate

```bash
python data_generation/generate_transactions.py \
    --n-rows 5000000 --n-accounts 50000 --fraud-rate 0.002 --seed 42 \
    --out data/raw/transactions.parquet
```

`--n-rows` is fully configurable — use a smaller value (e.g. `50000`) for
fast iteration during development. The committed numbers below are from the
default full-scale run (`--n-rows 5000000 --seed 42`), executed once and
recorded here — not placeholders.

## Actual numbers from the committed run

| Metric | Value |
|---|---|
| Total rows | 5,010,000 |
| Fraud rows | 10,000 |
| Fraud rate | 0.1996% |
| Unique accounts | 50,000 |
| Date range | 2025-01-01 to 2025-12-31 |
| Generation time | ~88s (Python 3.11, single machine, vectorized pandas/numpy) |

Row count comes out slightly above the `--n-rows` target because injected
fraud transactions are *added* on top of the base population rather than
replacing existing rows, so the fraud rate stays close to the requested
value without thinning out normal traffic.

## Schema

### Raw fields

| Field | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique ID, assigned after final chronological sort (`txn_0000000001`, ...) |
| `account_id` | string | Synthetic account identifier (`acct_000000`, ...) |
| `timestamp` | datetime | Transaction time, uniformly distributed across 2025 with per-account activity-level weighting |
| `amount` | float | Dollar amount, log-normal per merchant category, scaled by a per-account spend multiplier |
| `merchant_category` | string | One of 12 categories (grocery, dining, gas_station, online_retail, electronics, travel, entertainment, utilities, pharmacy, clothing, jewelry, cash_advance) |
| `location` | string | Nearest city name for the transaction (usually the account's home city; ~3% of legitimate transactions are flagged as normal "travel" to a different city) |
| `device_id` | string | Device used, scoped to the account (`acct_000000_dev0`, `acct_000000_dev1`, ...) |
| `is_fraud` | bool | Label |

### Engineered behavioral features (15)

All computed strictly from the raw fields above, grouped by `account_id` and
ordered by `timestamp` — none of them reference `is_fraud`, so none of them
are a relabeled copy of the target (see "Avoiding label leakage" below).

| Feature | Description |
|---|---|
| `hour_of_day` | Hour of the transaction timestamp (0–23) |
| `day_of_week` | Day of week (0=Monday) |
| `is_weekend` | 1 if Saturday/Sunday |
| `time_since_last_txn_sec` | Seconds since this account's previous transaction. **`-1` sentinel** for an account's first transaction in the dataset (no prior history to compare against) |
| `txn_count_last_1h` | Count of this account's transactions in the trailing 1-hour window (velocity signal), current transaction included |
| `txn_count_last_24h` | Same, trailing 24-hour window |
| `avg_amount_last_5_txns` | Mean amount of this account's previous 5 transactions (current excluded). Falls back to the current transaction's own amount for an account's first transaction |
| `amount_to_avg_ratio` | `amount / avg_amount_last_5_txns` — how unusual this amount is relative to the account's recent ticket size |
| `geo_distance_from_last_txn_km` | Haversine distance between this and the account's previous transaction location. **`-1` sentinel** for the first transaction |
| `home_distance_km` | Haversine distance between this transaction and the account's home city |
| `is_new_device` | 1 if this is the first time this account has used this `device_id` |
| `is_new_merchant_category` | 1 if this is the first time this account has transacted in this `merchant_category` |
| `cumulative_distinct_devices` | Running count of distinct devices this account has used, as of this transaction |
| `cumulative_distinct_merchant_categories` | Running count of distinct merchant categories this account has used, as of this transaction |
| `account_txn_seq_num` | This transaction's 1-indexed position in the account's history (a tenure/experience proxy) |

**Sentinel convention:** `-1` means "not applicable / no prior transaction to
compare against," not a real distance or time value. Treat it as a missing-data
flag during modeling (e.g. a separate indicator feature or explicit handling),
not as a literal small number.

## Fraud injection methodology

The generator does **not** produce fraud via i.i.d. random noise — a model
trained on random noise wouldn't produce meaningful precision/recall. Instead,
after generating a normal (`is_fraud=False`) population of transactions, three
fraud patterns are injected on top, split roughly 40% / 30% / 30% of the fraud
budget:

1. **Velocity burst** — 8 rapid-fire transactions on one account within a
   10-minute window, at 2–5x the account's typical amount, concentrated in
   higher-risk categories.
2. **Geo-impossible sequence** — a transaction 5–45 minutes after a normal
   one, but at a randomly chosen city far from the account's home — a travel
   speed no real traveler could achieve.
3. **New device + high amount** — a transaction on a device never seen
   before for that account, at 4–8x the account's typical amount, skewed
   toward electronics/jewelry/cash_advance/online_retail.

Fraud rows are merged into the base population *before* engineered features
are computed, so downstream features (velocity, geo-distance, is_new_device,
etc.) reflect the fraud events naturally — they weren't hand-set to "look
fraudulent," they emerge from the same feature computation as everything
else.

### Signal strength (from the committed run — see `data_generation/eda.py` output)

| Feature | Non-fraud mean | Fraud mean |
|---|---|---|
| `amount` | $251.05 | $1,548.68 |
| `amount_to_avg_ratio` | 1.72 | 7.66 |
| `txn_count_last_1h` | 1.02 | 2.41 |
| `geo_distance_from_last_txn_km` | 121.41 km | 598.08 km |
| `home_distance_km` | 62.80 km | 591.15 km |
| `is_new_device` | 1% | 31% |

Fraud rate by category ranges from ~0.06% (grocery, dining, gas, etc.) to
~0.49% (electronics, online_retail, jewelry, cash_advance) — an ~8x lift,
not a deterministic split. These features separate fraud from legitimate
activity on average, but overlap significantly — this is deliberate, so
Session 5's model comparison is a real modeling exercise rather than a
lookup table.

Plots (class balance, amount distribution, velocity distribution, fraud vs.
legitimate) are saved to `docs/eda/` by `data_generation/eda.py`.

## Avoiding label leakage

- No engineered feature reads `is_fraud`; all are pure functions of
  `timestamp`, `amount`, `device_id`, `merchant_category`, and location,
  computed identically for fraud and legitimate rows.
- Fraud transactions are ordinary rows in the same table, sorted into the
  same per-account chronological sequence as everything else — the feature
  computation code has no branch that treats them differently.
- The account-level running features (`cumulative_distinct_devices`,
  `account_txn_seq_num`, etc.) are computed only from what an account has
  done *up to and including* the current row — no future information (e.g.
  from transactions later in time) leaks backward into earlier rows.
