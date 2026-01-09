import os
import numpy as np
import pandas as pd

from src.env.fulfillment_env import FulfillmentRoutingEnv


def collect_rollouts(
    out_path="data_processed/wm_transitions.parquet",
    episodes=5,
    epsilon=0.30,
    max_steps=None,
    seed=42,
):
    """
    Collect transition data for training a world model.

    We mix:
      - random actions (prob epsilon)
      - cycling actions (0,1,2) otherwise

    Saved columns:
      obs, action, next_obs, reward, cost, late, queues, time_bucket
    """
    env = FulfillmentRoutingEnv(seed=seed)
    rng = np.random.default_rng(seed)

    rows = []

    for ep in range(episodes):
        obs, info = env.reset()
        t = 0

        while True:
            if (rng.random() < epsilon):
                action = env.action_space.sample()
            else:
                action = t % 3  # cycle

            next_obs, reward, done, trunc, step_info = env.step(action)

            rows.append(
                {
                    "episode": ep,
                    "t": t,
                    "time_bucket": step_info["time_bucket"],
                    "action": int(action),
                    "policy": step_info["policy"],
                    "reward": float(reward),
                    "cost": float(step_info["step_cost"]),
                    "late": float(step_info["late"]),
                    "processed": float(step_info["processed"]),
                    "arrivals": float(step_info["arrivals"]),
                    "obs": obs.tolist(),
                    "next_obs": next_obs.tolist(),
                    # store queues explicitly for convenience
                    "q_fc1": float(step_info.get("q_fc1", np.nan)),
                    "q_fc2": float(step_info.get("q_fc2", np.nan)),
                    "q_fc3": float(step_info.get("q_fc3", np.nan)),
                }
            )

            obs = next_obs
            t += 1

            if max_steps is not None and t >= max_steps:
                break
            if done or trunc:
                break

        print(f"Episode {ep} collected: {t} steps")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(df):,} | Cols: {df.shape[1]}")
    print("Head:\n", df.head(3))


if __name__ == "__main__":
    collect_rollouts()
