"""Train an LSTM fraud classifier over per-account transaction sequences
and log the run to MLflow.

Sequence construction: for every transaction, the model input is that
account's last `--seq-len` transactions (current one included), left-
padded with zeros if the account has fewer than `--seq-len` prior
transactions -- so this is a many-to-one sequence classifier, not a
per-timestep one. A window's *target* transaction determines which split
(train/val/test) it belongs to via the same chronological cutoffs as the
other models, but the window's input can legitimately reach back across a
split boundary into that account's earlier (e.g. train-period) history --
that isn't leakage, since it's still strictly-past information relative
to the target transaction, exactly like the offline-computed engineered
features (avg_amount_last_5_txns etc.) already do.

Class imbalance: `pos_weight` in `BCEWithLogitsLoss`, computed from the
train split's own class ratio -- SMOTE doesn't compose naturally with
sequence inputs (which minority-class transaction would you synthesize:
the whole window, or just the last step?), so weighting the loss is both
simpler and avoids that ambiguity. See docs/model_comparison.md.

Scale note: unlike the other 3 models (trained on the full 5M-row
dataset), this script trains on a `--n-accounts` subsample (default
10,000 of 50,000 accounts, ~1M rows) -- a deliberate, documented tradeoff
for local-GPU (4GB, e.g. GTX 1650 Ti) iteration speed. Sequence models
cost far more per epoch than the tabular models at the same row count.

Usage:
    python training/train_lstm.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data_prep import (  # noqa: E402
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    build_feature_matrix,
    compute_sentinel_fill_values,
    evaluate_model,
    load_dataset,
    time_split,
)


class AccountSequenceDataset(Dataset):
    """Slices a (seq_len, n_features) window ending at each requested row
    on the fly, rather than materializing all windows up front (5M rows x
    20 steps x ~29 features would be tens of GB) -- rows are pre-sorted by
    (account_id, timestamp) so each account occupies a contiguous block,
    letting `account_start` be derived in O(1) per row."""

    def __init__(
        self,
        row_indices: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        account_start: np.ndarray,
        seq_len: int,
    ):
        self.row_indices = row_indices
        self.features = features
        self.labels = labels
        self.account_start = account_start
        self.seq_len = seq_len
        self.n_features = features.shape[1]

    def __len__(self) -> int:
        return len(self.row_indices)

    def __getitem__(self, i: int):
        row = self.row_indices[i]
        start = self.account_start[row]
        window_start = max(start, row - self.seq_len + 1)
        window = self.features[window_start : row + 1]
        n_real = window.shape[0]
        if n_real < self.seq_len:
            pad = np.zeros((self.seq_len - n_real, self.n_features), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        return window, self.labels[row]


class FraudLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)  # logits, last layer's final hidden state


def build_sequence_arrays(
    df, fill_values: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, "pd.Series"]:
    df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
    features = build_feature_matrix(df, fill_values).to_numpy(dtype=np.float32)
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.float32)
    # Rows are contiguous per account after the sort above, so
    # (row_index - position_within_account) is constant across each
    # account's block and equals that block's starting row -- an O(1),
    # fully vectorized way to get each row's account-start offset without
    # a groupby lookup per row.
    pos_in_account = df.groupby("account_id").cumcount().to_numpy()
    account_start = np.arange(len(df)) - pos_in_account
    return features, labels, account_start, df["timestamp"]


def run_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def score_loader(model, loader, device) -> np.ndarray:
    model.eval()
    scores = []
    for x, _ in loader:
        logits = model(x.to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment-name", default=MLFLOW_EXPERIMENT_NAME)
    parser.add_argument("--fpr-target", type=float, default=0.02)
    parser.add_argument("--n-accounts", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    print("loading dataset + subsampling accounts...")
    df = load_dataset()
    rng = np.random.default_rng(args.seed)
    all_accounts = df["account_id"].unique()
    sampled_accounts = rng.choice(all_accounts, size=min(args.n_accounts, len(all_accounts)), replace=False)
    df = df[df["account_id"].isin(sampled_accounts)].reset_index(drop=True)
    print(f"  {len(sampled_accounts):,} accounts -> {len(df):,} rows, fraud rate {df[LABEL_COLUMN].mean():.4%}")

    train_cut_df, val_cut_df, test_cut_df = time_split(df)
    fill_values = compute_sentinel_fill_values(train_cut_df)
    train_cutoff = train_cut_df["timestamp"].max()
    val_cutoff = val_cut_df["timestamp"].max()

    features, labels, account_start, timestamps = build_sequence_arrays(df, fill_values)

    train_mask = (timestamps <= train_cutoff).to_numpy()
    val_mask = ((timestamps > train_cutoff) & (timestamps <= val_cutoff)).to_numpy()
    test_mask = (timestamps > val_cutoff).to_numpy()

    # StandardScaler equivalent (fit on train only): required for the same
    # reason as train_logreg.py -- raw feature scales span orders of
    # magnitude and would otherwise dominate/stall LSTM gradient updates.
    # Padding rows end up as -mean/std rather than exactly 0 after this;
    # a documented simplification (a separate "is padding" mask channel
    # would be the more precise fix) rather than added complexity here.
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std[std == 0] = 1.0
    features = (features - mean) / std

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f"  sequence splits -- train: {len(train_idx):,}  val: {len(val_idx):,}  test: {len(test_idx):,}")

    train_ds = AccountSequenceDataset(train_idx, features, labels, account_start, args.seq_len)
    val_ds = AccountSequenceDataset(val_idx, features, labels, account_start, args.seq_len)
    test_ds = AccountSequenceDataset(test_idx, features, labels, account_start, args.seq_len)

    # num_workers=0: the per-item work is plain numpy slicing (cheap), and
    # multiprocessing DataLoader workers add spawn overhead on Windows that
    # isn't worth it at this dataset size.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_pos = labels[train_idx].sum()
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    model = FraudLSTM(len(FEATURE_COLUMNS), args.hidden_size, args.num_layers).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run(run_name="lstm"):
        mlflow.log_params(
            {
                "model_type": "LSTM",
                "class_imbalance_strategy": f"pos_weight={pos_weight.item():.1f} in BCEWithLogitsLoss",
                "device": device,
                "n_accounts": len(sampled_accounts),
                "seq_len": args.seq_len,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_features": len(FEATURE_COLUMNS),
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "test_rows": len(test_idx),
                "fpr_target": args.fpr_target,
            }
        )

        start = time.time()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            print(f"  epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  ({time.time() - epoch_start:.1f}s)")
        train_seconds = time.time() - start
        mlflow.log_metric("train_seconds", train_seconds)

        val_score = score_loader(model, val_loader, device)
        test_score = score_loader(model, test_loader, device)
        metrics = evaluate_model(labels[val_idx], val_score, labels[test_idx], test_score, args.fpr_target)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"trained in {train_seconds:.1f}s")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
