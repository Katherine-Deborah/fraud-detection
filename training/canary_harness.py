"""Canary harness: unlike shadow mode, the candidate model's predictions
are actually **acted on** for a slice of traffic -- a deterministic 10% of
accounts, by hash(account_id). The other 90% keeps being served by the
current production model. This is the Canary stage of the Session 7
staged rollout (see docs/model_registry.md).

Same "no Session 8 serving layer yet" caveat as shadow_harness.py: traffic
here is the chronological test split, split further by account into a
canary slice and a production slice. Splitting by *account* (not by row)
matters -- an account's transactions should consistently land on one side
or the other, the way a real canary router would pin a user to a
treatment group for the duration of the rollout.

Usage:
    python training/canary_harness.py --algorithm random_forest
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import promotion_criteria, registry  # noqa: E402
from training.data_prep import (  # noqa: E402
    build_feature_matrix,
    compute_sentinel_fill_values,
    load_dataset,
    time_split,
)
from training.shadow_harness import _false_positive_rate, _score, find_candidate_version  # noqa: E402

CANARY_LOG_DIR = REPO_ROOT / "data" / "registry_canary_logs"
CANARY_FRACTION_BUCKETS = 10  # 1 of 10 hash buckets -> ~10% canary traffic


def _is_canary_account(account_id: str) -> bool:
    digest = hashlib.md5(account_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % CANARY_FRACTION_BUCKETS == 0


def run_canary(client, candidate_version) -> tuple[bool, dict]:
    production_version = registry.get_current_production(client)
    if production_version is None:
        raise SystemExit("no production model to canary against")

    print(f"loading candidate v{candidate_version.version} and production v{production_version.version}...")
    candidate_model = mlflow.sklearn.load_model(f"models:/{registry.REGISTERED_MODEL_NAME}/{candidate_version.version}")
    production_model = mlflow.sklearn.load_model(f"models:/{registry.REGISTERED_MODEL_NAME}/{production_version.version}")
    candidate_threshold = registry.source_run_metrics(client, candidate_version)["threshold"]
    production_threshold = registry.source_run_metrics(client, production_version)["threshold"]

    print("loading dataset + preparing chronological splits (test split as replayed traffic)...")
    df = load_dataset()
    train_df, _, test_df = time_split(df)
    fill_values = compute_sentinel_fill_values(train_df)
    test_df = test_df.reset_index(drop=True)
    X_test = build_feature_matrix(test_df, fill_values)
    y_test = test_df["is_fraud"].astype(int)

    is_canary = test_df["account_id"].apply(_is_canary_account)
    print(f"canary slice: {is_canary.sum():,} rows ({is_canary.mean():.1%} of accounts by hash bucket), "
          f"production slice: {(~is_canary).sum():,} rows")

    canary_score, canary_pred = _score(candidate_model, X_test[is_canary], candidate_threshold)
    prod_score, prod_pred = _score(production_model, X_test[~is_canary], production_threshold)

    y_canary = y_test[is_canary].reset_index(drop=True)
    y_prod = y_test[~is_canary].reset_index(drop=True)

    canary_recall = float(recall_score(y_canary, canary_pred, zero_division=0))
    canary_fpr = _false_positive_rate(y_canary, canary_pred)
    canary_fraud_count = int(y_canary.sum())
    production_recall = float(recall_score(y_prod, prod_pred, zero_division=0))
    production_fpr = _false_positive_rate(y_prod, prod_pred)

    CANARY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CANARY_LOG_DIR / f"{candidate_version.tags.get(registry.TAG_ALGORITHM)}_v{candidate_version.version}.parquet"
    pd.concat(
        [
            pd.DataFrame({"slice": "canary", "y_true": y_canary, "score": canary_score, "pred": canary_pred, "acted_on_by": "candidate"}),
            pd.DataFrame({"slice": "production", "y_true": y_prod, "score": prod_score, "pred": prod_pred, "acted_on_by": "production"}),
        ]
    ).to_parquet(log_path)
    print(f"wrote canary log -> {log_path}")

    passed, detail = promotion_criteria.gate_canary_to_production(
        canary_recall, canary_fpr, production_recall, canary_fraud_count
    )
    detail["production_fpr_on_canary_batch"] = production_fpr
    detail["n_canary_rows"] = int(is_canary.sum())
    detail["n_production_rows"] = int((~is_canary).sum())
    return passed, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True)
    args = parser.parse_args()

    client = registry.get_client()
    candidate_version = find_candidate_version(client, args.algorithm)
    if registry.get_stage(client, candidate_version.version) != registry.STAGE_CANARY:
        raise SystemExit(
            f"v{candidate_version.version} is in stage "
            f"{registry.get_stage(client, candidate_version.version)!r}, expected 'canary' "
            f"-- run shadow_harness.py first"
        )

    passed, detail = run_canary(client, candidate_version)
    print("\ncanary comparison:")
    for k, v in detail.items():
        print(f"  {k}: {v}")

    reason = (
        f"canary slice ({detail['n_canary_rows']:,} rows, {detail['canary_fraud_count']} fraud): "
        f"recall={detail['canary_recall']:.4f} fpr={detail['canary_fpr']:.4f} "
        f"vs production recall={detail['production_recall']:.4f}"
    )
    production_version = registry.get_current_production(client)
    if passed:
        registry.promote(client, candidate_version, registry.STAGE_PRODUCTION, reason, detail)
        registry.set_stage(client, production_version.version, registry.STAGE_ARCHIVED)
        registry.log_audit_event(
            client, "archive", production_version, registry.STAGE_PRODUCTION, registry.STAGE_ARCHIVED,
            True, f"superseded by v{candidate_version.version} ({candidate_version.tags.get(registry.TAG_ALGORITHM)})",
        )
    else:
        registry.reject(client, candidate_version, reason, detail)


if __name__ == "__main__":
    main()
