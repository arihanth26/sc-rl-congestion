import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class WMTransitions(Dataset):
    def __init__(self, path):
        df = pd.read_parquet(path)

        X_obs = np.vstack(df["obs"].to_numpy()).astype(np.float32)         # (N, 8)
        A = df["action"].to_numpy().astype(np.int64)                      # (N,)
        Y_next = np.vstack(df["next_obs"].to_numpy()).astype(np.float32)  # (N, 8)
        Y_cost = df["cost"].to_numpy().astype(np.float32).reshape(-1, 1)  # (N, 1)

        # one-hot action (3)
        A_oh = np.zeros((len(A), 3), dtype=np.float32)
        A_oh[np.arange(len(A)), A] = 1.0

        self.X = np.hstack([X_obs, A_oh]).astype(np.float32)     # (N, 11)
        self.Y_next = Y_next.astype(np.float32)                  # (N, 8)
        self.Y_cost = Y_cost.astype(np.float32)                  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (
            torch.from_numpy(self.X[i]),
            torch.from_numpy(self.Y_next[i]),
            torch.from_numpy(self.Y_cost[i]),
        )


class WorldModel(nn.Module):
    def __init__(self, in_dim=11, hidden=128, out_state=8):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head_state = nn.Linear(hidden, out_state)
        self.head_cost = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_state(h), self.head_cost(h)


def train(
    data_path="data_processed/wm_transitions.parquet",
    out_model="models/world_model.pt",
    out_dir="outputs/day4_world_model",
    epochs=15,
    batch_size=2048,
    lr=1e-3,
    cost_weight=0.05,
    seed=42,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    ds = WMTransitions(data_path)

    # split
    n = len(ds)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.9 * n)
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_ds = torch.utils.data.Subset(ds, tr_idx)
    va_ds = torch.utils.data.Subset(ds, va_idx)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    tr_hist, va_hist = [], []

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0

        for X, Y_next, Y_cost in tr_loader:
            X = X.to(device)
            Y_next = Y_next.to(device)
            Y_cost = Y_cost.to(device)

            pred_next, pred_cost = model(X)

            loss_state = mse(pred_next, Y_next)
            loss_cost = mse(pred_cost, Y_cost)
            loss = loss_state + cost_weight * loss_cost

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * X.size(0)

        tr_loss /= len(tr_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for X, Y_next, Y_cost in va_loader:
                X = X.to(device)
                Y_next = Y_next.to(device)
                Y_cost = Y_cost.to(device)
                pred_next, pred_cost = model(X)

                loss_state = mse(pred_next, Y_next)
                loss_cost = mse(pred_cost, Y_cost)
                loss = loss_state + cost_weight * loss_cost
                va_loss += loss.item() * X.size(0)

        va_loss /= len(va_ds)

        tr_hist.append(tr_loss)
        va_hist.append(va_loss)

        print(f"Epoch {ep:02d}/{epochs} | train={tr_loss:.6f} | val={va_loss:.6f}")

    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": 11,
            "out_state": 8,
        },
        out_model,
    )
    print(f"\nSaved model: {out_model}")

    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(tr_hist, label="train")
    plt.plot(va_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, "wm_loss.png")
    plt.savefig(p, dpi=160)
    print(f"Saved: {p}")


if __name__ == "__main__":
    train()
