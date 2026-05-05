"""
ExchangeNetwork: learns the task-exchange rule via behavior cloning.
ExchangeDataCollector: collects (obs_i, obs_j, label) pairs during env steps.
"""
import json
import math
import os
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from envs.utils.utils import cal_distance


# ---------------------------------------------------------------------------
# ExchangeNetwork
# ---------------------------------------------------------------------------

class ExchangeNetwork(nn.Module):
    """
    Predicts whether two agents should swap their discrete goal targets.

    Input features (per pair, concatenated = 20 dims):
        agent_i: gx-px, gy-py, v, theta, for_feature, vx_for, vy_for, px, py  (9)
        agent_j: same layout                                                   (9)
        derived: dist_ij, mutual_improve_score                                 (2)

    Output: single logit (sigmoid -> swap probability).
    """

    AGENT_FEAT_DIM = 9  # 7 base + 2 position
    PAIR_FEAT_DIM = AGENT_FEAT_DIM * 2 + 2  # 20

    def __init__(self, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.PAIR_FEAT_DIM, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """pair_features: [batch, 20] -> [batch, 1] logit."""
        return self.net(pair_features)

    @staticmethod
    def build_pair_features(
        obs_i: np.ndarray,
        obs_j: np.ndarray,
        px_i: float, py_i: float,
        px_j: float, py_j: float,
        gx_i: float, gy_i: float,
        gx_j: float, gy_j: float,
    ) -> np.ndarray:
        """
        Build the 20-dim feature vector for one (i, j) pair.

        Parameters:
            obs_i, obs_j: full observation rows (first 9 dims used)
            px/py: world positions
            gx/gy: goal positions
        Returns:
            np.ndarray of shape (20,)
        """
        feat_i = np.array(obs_i[:9], dtype=np.float32)
        feat_j = np.array(obs_j[:9], dtype=np.float32)

        dist_ij = float(cal_distance(px_i, py_i, px_j, py_j))

        # mutual_improve: positive means both agents get closer after swap
        d_i_current = cal_distance(px_i, py_i, gx_i, gy_i)
        d_j_current = cal_distance(px_j, py_j, gx_j, gy_j)
        d_i_after = cal_distance(px_i, py_i, gx_j, gy_j)
        d_j_after = cal_distance(px_j, py_j, gx_i, gy_i)
        mutual_improve = float(
            (d_i_current - d_i_after) + (d_j_current - d_j_after)
        )

        return np.concatenate([feat_i, feat_j, [dist_ij, mutual_improve]]).astype(np.float32)

    @torch.no_grad()
    def predict_swap_prob(self, pair_features: np.ndarray) -> np.ndarray:
        """
        pair_features: [batch, 20] numpy array
        Returns: [batch] swap probabilities in [0, 1]
        """
        x = torch.from_numpy(pair_features)
        logit = self.forward(x)
        prob = torch.sigmoid(logit).squeeze(-1).numpy()
        return prob

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
        self.to(device)
        self.eval()


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _load_exchange_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all JSONL files from data_dir, return (features, labels)."""
    features_list = []
    labels_list = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                feat = np.array(rec["features"], dtype=np.float32)
                label = int(rec["label"])
                features_list.append(feat)
                labels_list.append(label)

    if not features_list:
        return np.empty((0, ExchangeNetwork.PAIR_FEAT_DIM), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X = np.stack(features_list, axis=0)
    y = np.array(labels_list, dtype=np.float32)
    return X, y


def train_exchange_network(
    data_dir: str,
    model_save_path: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    val_split: float = 0.2,
) -> dict:
    """
    Train ExchangeNetwork from JSONL data.

    Returns: dict with 'best_val_acc', 'final_loss', 'num_samples'
    """
    X, y = _load_exchange_data(data_dir)
    if X.shape[0] < 32:
        print(f"[exchange_net] too few samples ({X.shape[0]}), skip training")
        return {"best_val_acc": 0.0, "final_loss": -1.0, "num_samples": X.shape[0]}

    # Train/val split
    n = X.shape[0]
    n_val = max(1, int(n * val_split))
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ExchangeNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device)).squeeze(-1)
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            val_acc = (val_preds == y_val.to(device)).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / max(1, n_batches)
            print(
                f"[exchange_net] epoch {epoch}/{epochs} "
                f"loss={avg_loss:.4f} val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"[exchange_net] model saved to {model_save_path} (best_val_acc={best_val_acc:.4f})")

    return {
        "best_val_acc": best_val_acc,
        "final_loss": total_loss / max(1, n_batches),
        "num_samples": n,
    }


# ---------------------------------------------------------------------------
# ExchangeDataCollector
# ---------------------------------------------------------------------------

class ExchangeDataCollector:
    """
    Collects exchange training data during env steps.

    Appends to a single file per process to avoid creating millions of tiny files.
    """

    FLUSH_INTERVAL = 5000  # flush buffer every N samples

    def __init__(self, data_dir: str = "./exchange_data", file_id: str = "all"):
        self.data_dir = data_dir
        self.buffer: List[dict] = []
        self.total_count = 0
        os.makedirs(self.data_dir, exist_ok=True)
        # Single consolidated file per process (avoids millions of tiny files)
        self._fpath = os.path.join(self.data_dir, f"exchange_data_{file_id}.jsonl")

    def add_candidate(
        self,
        obs_i: np.ndarray,
        obs_j: np.ndarray,
        px_i: float, py_i: float,
        px_j: float, py_j: float,
        gx_i: float, gy_i: float,
        gx_j: float, gy_j: float,
        swapped: bool,
    ):
        features = ExchangeNetwork.build_pair_features(
            obs_i, obs_j, px_i, py_i, px_j, py_j, gx_i, gy_i, gx_j, gy_j
        )
        self.buffer.append({
            "features": features.tolist(),
            "label": 1 if swapped else 0,
        })
        self.total_count += 1
        if len(self.buffer) >= self.FLUSH_INTERVAL:
            self._flush_buffer()

    def _flush_buffer(self):
        if not self.buffer:
            return
        with open(self._fpath, "a", encoding="utf-8") as f:
            for rec in self.buffer:
                f.write(json.dumps(rec) + "\n")
        self.buffer.clear()

    def flush_to_file(self, step_id: int = 0):
        """Final flush of any remaining buffered data."""
        self._flush_buffer()

    @property
    def sample_count(self) -> int:
        return len(self.buffer) + self.total_count
