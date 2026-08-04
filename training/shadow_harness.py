"""Shadow-mode harness: the candidate model scores the same "live" traffic
as the current production model, side by side. Production's predictions
are the ones that count; the candidate's are logged for comparison only --
never acted on. This is the Shadow stage of the Session 7 staged rollout
(see docs/model_registry.md).

No real serving layer exists yet (that's Session 8), so "live traffic" here
is a replay of the chronological **test split** from training/data_prep.py
-- the same held-out period every model's Session 5/6 test metrics were
already computed on, and never used to fit or threshold-select any model.
Documented explicitly as a stand-in; Session 8's FastAPI gateway is the
natural place to point this harness at real traffic later.

Each model scores with its **own pre-selected threshold** (the one chosen
on validation during its own training run, logged as `metrics.threshold`)
-- not a threshold re-picked on this batch, which would just be tuning on
test data under a different name.

Usage:
    python training/shadow_harness.py --algorithm random_forest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import promotion_criteria, registry  # noqa: E402
from training.data_prep import load_dataset, prepare_splits  # noqa: E402

SHADOW_LOG_DIR = REPO_ROOT / "data" / "registry_shadow_logs"


def _false_positive_rate(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def _score(model, X, threshold: float) -> tuple[pd.Series, pd.Series]:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.Series(proba), pd.Series(pred)


def find_candidate_version(client, algorithm: str):
    versions = [
        mv
        for mv in client.search_model_versions(f"name='{registry.REGISTERED_MODEL_NAME}'")
        if mv.tags.get(registry.TAG_ALGORITHM) == algorithm
    ]
    if not versions:
        raise SystemExit(f"no registered version found for algorithm={algorithm!r} -- register it first")
    return max(versions, key=lambda v: int(v.version))


def run_shadow(client, candidate_version) -> tuple[bool, dict]:
    """Runs the shadow comparison for `candidate_version` against whatever
    is currently `production`. Returns (passed, detail) from
    promotion_criteria.gate_shadow_to_canary, and as a side effect writes
    the non-acted-on shadow log to SHADOW_LOG_DIR."""
    production_version = registry.get_current_production(client)
    if production_version is None:
        raise SystemExit(
            "no production model to shadow against -- the very first model "
            "bypasses shadow/canary entirely (bootstrap), see docs/model_registry.md"
        )

    print(f"loading candidate v{candidate_version.version} ({candidate_version.tags.get(registry.TAG_ALGORITHM)}) "
          f"and production v{production_version.version} ({production_version.tags.get(registry.TAG_ALGORITHM)})...")
    candidate_model = mlflow.sklearn.load_model(f"models:/{registry.REGISTERED_MODEL_NAME}/{candidate_version.version}")
    production_model = mlflow.sklearn.load_model(f"models:/{registry.REGISTERED_MODEL_NAME}/{production_version.version}")

    candidate_metrics = registry.source_run_metrics(client, candidate_version)
    production_metrics = registry.source_run_metrics(client, production_version)
    candidate_threshold = candidate_metrics["threshold"]
    production_threshold = production_metrics["threshold"]

    print("loading dataset + preparing chronological splits (using test split as replayed traffic)...")
    df = load_dataset()
    _, _, _, _, X_test, y_test, _ = prepare_splits(df)

    candidate_score, candidate_pred = _score(candidate_model, X_test, candidate_threshold)
    production_score, production_pred = _score(production_model, X_test, production_threshold)

    candidate_recall = float(recall_score(y_test, candidate_pred, zero_division=0))
    candidate_fpr = _false_positive_rate(y_test, candidate_pred)
    candidate_auc_pr = float(average_precision_score(y_test, candidate_score))
    production_recall = float(recall_score(y_test, production_pred, zero_division=0))
    production_fpr = _false_positive_rate(y_test, production_pred)

    SHADOW_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SHADOW_LOG_DIR / f"{candidate_version.tags.get(registry.TAG_ALGORITHM)}_v{candidate_version.version}.parquet"
    pd.DataFrame(
        {
            "y_true": y_test.reset_index(drop=True),
            "candidate_score": candidate_score,
            "candidate_pred": candidate_pred,
            "production_score": production_score,
            "production_pred": production_pred,
        }
    ).to_parquet(log_path)
    print(f"wrote shadow log ({len(y_test):,} rows, predictions logged, none acted on) -> {log_path}")

    passed, detail = promotion_criteria.gate_shadow_to_canary(candidate_recall, candidate_fpr, production_recall)
    detail["candidate_auc_pr_on_shadow_batch"] = candidate_auc_pr
    detail["production_fpr_on_shadow_batch"] = production_fpr
    detail["n_rows"] = len(y_test)
    return passed, detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, help="algorithm tag of the candidate version, e.g. random_forest")
    args = parser.parse_args()

    client = registry.get_client()
    candidate_version = find_candidate_version(client, args.algorithm)
    passed, detail = run_shadow(client, candidate_version)

    print("\nshadow comparison:")
    for k, v in detail.items():
        print(f"  {k}: {v}")

    reason = (
        f"shadow batch: candidate recall={detail['candidate_recall']:.4f} fpr={detail['candidate_fpr']:.4f} "
        f"vs production recall={detail['production_recall']:.4f}"
    )
    if passed:
        registry.promote(client, candidate_version, registry.STAGE_CANARY, reason, detail)
    else:
        registry.reject(client, candidate_version, reason, detail)


if __name__ == "__main__":
    main()
