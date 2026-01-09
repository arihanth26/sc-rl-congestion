import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.train_world_model import WorldModel


def main(
    data_path="data_processed/wm_transitions.parquet",
    model_path="models/world_model.pt",
    out_dir="outputs/day4_world_model",
    n_points=4000,
    seed=42,
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(data_path)

    if len(df) > n_points:
        df = df.sample(n=n_points, random_state=seed).reset_index(drop=True)

    X_obs = np.vstack(df["obs"].to_numpy()).astype(np.float32)
    A = df["action"].to_numpy().astype(np.int64)
    Y_next = np.vstack(df["next_obs"].to_numpy()).astype(np.float32)
    Y_cost = df["cost"].to_numpy().astype(np.float32)

    A_oh = np.zeros((len(A), 3), dtype=np.float32)
    A_oh[np.arange(len(A)), A] = 1.0
    X = np.hstack([X_obs, A_oh]).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WorldModel().to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_next, pred_cost = model(torch.from_numpy(X).to(device))
        pred_next = pred_next.cpu().numpy()
        pred_cost = pred_cost.cpu().numpy().reshape(-1)

    # Metrics
    cost_mae = float(np.mean(np.abs(pred_cost - Y_cost)))
    q_mae = float(np.mean(np.abs(pred_next[:, :3] - Y_next[:, :3])))
    print(f"Cost MAE: {cost_mae:.4f}")
    print(f"Queue MAE (avg q1,q2,q3): {q_mae:.4f}")

    # Plots
    plt.figure()
    plt.scatter(Y_cost, pred_cost, s=3)
    plt.xlabel("true cost")
    plt.ylabel("predicted cost")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "wm_pred_vs_true_cost.png")
    plt.savefig(p1, dpi=160)
    print(f"Saved: {p1}")

    plt.figure()
    plt.scatter(Y_next[:, 0], pred_next[:, 0], s=3)
    plt.xlabel("true next q_fc1")
    plt.ylabel("predicted next q_fc1")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "wm_pred_vs_true_q_fc1.png")
    plt.savefig(p2, dpi=160)
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
