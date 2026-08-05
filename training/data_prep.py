"""Shared data loading, chronological split, feature preparation, and
evaluation helpers used by every training/train_*.py script in Session 5.

Split strategy: a **chronological** train/val/test split on `timestamp`
(70/15/15 by time span, not by row count) -- shuffling here would leak
future transactions into training, which is the single most common and
easiest-to-miss bug in fraud-detection modeling (see SESSIONS.md Session 5
"Nuance to watch for"). The engineered features themselves were already
computed offline using only each account's *past* history (documented in
docs/dataset.md's "Avoiding label leakage" section), so this split doesn't
need to touch or recompute them -- it only has to make sure no test-period
row is used for fitting.

Class imbalance: handled via **class weighting**, not SMOTE, for every
supervised model (LogReg, RF, LSTM) -- decided explicitly for this session
rather than defaulted to. See docs/model_comparison.md for the per-model
rationale. Isolation Forest is unsupervised and uses neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# scikit-learn is imported lazily, inside the functions that actually need
# it (select_threshold_for_fpr / evaluate_at_threshold), not at module level.
# Session 9's Airflow drift_report task imports this module purely for its
# feature-matrix schema (FEATURE_COLUMNS, build_feature_matrix,
# load_dataset) -- none of which touch sklearn -- and the Airflow image
# (requirements-airflow.txt) deliberately doesn't carry a scikit-learn
# pin, so a module-level `from sklearn.metrics import ...` would break that
# import for no reason.

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.generate_transactions import MERCHANT_CATEGORIES  # noqa: E402

RAW_TRANSACTIONS_PATH = REPO_ROOT / "data" / "raw" / "transactions.parquet"

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "fraud-detection"

# The 15 engineered behavioral features from docs/dataset.md, plus the raw
# `amount` field (essential predictor, not itself one of the 15) and a
# derived `is_first_txn` flag (see SENTINEL_COLUMNS below). merchant_category
# is added separately as one-hot columns since it's categorical.
ENGINEERED_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "time_since_last_txn_sec",
    "txn_count_last_1h",
    "txn_count_last_24h",
    "avg_amount_last_5_txns",
    "amount_to_avg_ratio",
    "geo_distance_from_last_txn_km",
    "home_distance_km",
    "is_new_device",
    "is_new_merchant_category",
    "cumulative_distinct_devices",
    "cumulative_distinct_merchant_categories",
    "account_txn_seq_num",
]

# docs/dataset.md: "-1 means not applicable / no prior transaction to compare
# against ... treat it as a missing-data flag, not a literal small number."
# Both columns are -1 on exactly the same rows (an account's first
# transaction), so a single `is_first_txn` indicator captures that case for
# both; the sentinel value itself is then replaced with a neutral fill
# (the train-split median of the real, non-sentinel values) so linear/LSTM
# models don't interpret -1 as a real negative time/distance.
SENTINEL_COLUMNS = ["time_since_last_txn_sec", "geo_distance_from_last_txn_km"]

CATEGORY_COLUMNS = [f"category_{c}" for c in MERCHANT_CATEGORIES]
NUMERIC_FEATURE_COLUMNS = ["amount"] + ENGINEERED_FEATURES + ["is_first_txn"]
FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORY_COLUMNS

LABEL_COLUMN = "is_fraud"


def load_dataset(path: Path = RAW_TRANSACTIONS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    return df


def time_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split by timestamp span (not row count) so train, val,
    and test each cover contiguous, non-overlapping calendar periods."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    span = t_max - t_min
    train_cutoff = t_min + span * train_frac
    val_cutoff = t_min + span * (train_frac + val_frac)

    train_df = df[df["timestamp"] <= train_cutoff]
    val_df = df[(df["timestamp"] > train_cutoff) & (df["timestamp"] <= val_cutoff)]
    test_df = df[df["timestamp"] > val_cutoff]
    return train_df, val_df, test_df


def compute_sentinel_fill_values(train_df: pd.DataFrame) -> dict[str, float]:
    """Median of the real (non-sentinel) values, computed from the training
    split only -- val/test must never influence a fill value used to train
    on them."""
    fill_values = {}
    for col in SENTINEL_COLUMNS:
        real_values = train_df.loc[train_df[col] != -1, col]
        fill_values[col] = float(real_values.median())
    return fill_values


def build_feature_matrix(df: pd.DataFrame, fill_values: dict[str, float]) -> pd.DataFrame:
    df = df.copy()
    df["is_first_txn"] = (df["account_txn_seq_num"] == 1).astype(int)
    for col, fill_value in fill_values.items():
        df.loc[df[col] == -1, col] = fill_value

    category_dummies = pd.get_dummies(df["merchant_category"], prefix="category")
    for col in CATEGORY_COLUMNS:
        if col not in category_dummies:
            category_dummies[col] = 0
    category_dummies = category_dummies[CATEGORY_COLUMNS].astype(int)

    numeric = df[NUMERIC_FEATURE_COLUMNS].reset_index(drop=True)
    return pd.concat([numeric, category_dummies.reset_index(drop=True)], axis=1)


def prepare_splits(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, float]
]:
    """Full pipeline: chronological split -> train-derived sentinel fills ->
    feature matrices. Returns (X_train, y_train, X_val, y_val, X_test, y_test, fill_values)."""
    train_df, val_df, test_df = time_split(df)
    fill_values = compute_sentinel_fill_values(train_df)

    X_train = build_feature_matrix(train_df, fill_values)
    X_val = build_feature_matrix(val_df, fill_values)
    X_test = build_feature_matrix(test_df, fill_values)

    y_train = train_df[LABEL_COLUMN].reset_index(drop=True)
    y_val = val_df[LABEL_COLUMN].reset_index(drop=True)
    y_test = test_df[LABEL_COLUMN].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, fill_values


def select_threshold_for_fpr(
    y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = 0.02
) -> tuple[float, float, float]:
    """Pick the score threshold that maximizes recall subject to FPR <=
    fpr_target, chosen on a held-out (validation) set -- never on test.
    Returns (threshold, achieved_fpr, recall_at_that_fpr). If no threshold
    achieves the target FPR, falls back to the lowest achievable FPR."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    eligible = np.where(fpr <= fpr_target)[0]
    if len(eligible) > 0:
        best_idx = eligible[np.argmax(tpr[eligible])]
    else:
        best_idx = int(np.argmin(fpr))
    return float(thresholds[best_idx]), float(fpr[best_idx]), float(tpr[best_idx])


def evaluate_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_pred = (y_score >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_model(
    y_val: np.ndarray,
    val_score: np.ndarray,
    y_test: np.ndarray,
    test_score: np.ndarray,
    fpr_target: float = 0.02,
) -> dict[str, float]:
    """Threshold is selected on validation (at the target FPR from
    PROJECT.md #6), then applied unchanged to test. AUC-PR is
    threshold-independent and computed directly on test."""
    from sklearn.metrics import average_precision_score

    threshold, val_fpr, val_recall_at_fpr = select_threshold_for_fpr(y_val, val_score, fpr_target)
    test_metrics = evaluate_at_threshold(y_test, test_score, threshold)
    return {
        "auc_pr": float(average_precision_score(y_test, test_score)),
        "threshold": threshold,
        "val_achieved_fpr": val_fpr,
        "val_recall_at_fpr_target": val_recall_at_fpr,
        **test_metrics,
    }


if __name__ == "__main__":
    df = load_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test, fill_values = prepare_splits(df)
    print(f"train: {len(X_train):>9,} rows, fraud rate {y_train.mean():.4%}")
    print(f"val:   {len(X_val):>9,} rows, fraud rate {y_val.mean():.4%}")
    print(f"test:  {len(X_test):>9,} rows, fraud rate {y_test.mean():.4%}")
    print(f"sentinel fill values (from train median): {fill_values}")
    print(f"feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
