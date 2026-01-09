import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from src.sim_core import FulfillmentSim


class FulfillmentRoutingEnv(gym.Env):
    """
    Gym wrapper around FulfillmentSim using real demand from demand_yellow.parquet.

    Demand file expected columns:
      - time_bucket (datetime)
      - zone_group (int)
      - demand (int/float)

    We convert it into total system demand per time bucket:
      D(t) = sum_z demand(z, t)

    Action space (Discrete):
      0 -> Cheapest-first (route all demand to the FC with lowest per-order ship cost)
      1 -> Capacity-proportional split
      2 -> Threshold stress-aware split (avoid FCs above stress threshold, then cheapest among remaining)

    Observation vector:
      [q1, q2, q3, util1, util2, util3, D_t, D_roll]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        demand_path="data_processed/demand_yellow.parquet",
        start="2024-01-01 00:00:00",
        end="2024-01-07 23:55:00",
        bucket_minutes=5,
        fc_ids=("fc1", "fc2", "fc3"),
        capacities=(900.0, 700.0, 500.0),
        shipping_cost_per_order=(3.0, 4.0, 6.0),
        late_penalty_per_order=6.0,
        overflow_penalty_per_order=12.0,
        queue_variance_penalty=0.02,
        stress_threshold=1.25,
        rolling_window=12,  # 12*5min = 60 minutes
        seed=42,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        # --- Load and build demand series ---
        df = pd.read_parquet(demand_path)

        # Ensure timestamp
        df["time_bucket"] = pd.to_datetime(df["time_bucket"])

        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)

        df = df[(df["time_bucket"] >= start_ts) & (df["time_bucket"] <= end_ts)].copy()

        # Total demand per bucket (across all zones)
        demand_ts = (
            df.groupby("time_bucket", as_index=False)["demand"]
            .sum()
            .sort_values("time_bucket")
            .reset_index(drop=True)
        )

        if len(demand_ts) == 0:
            raise ValueError("No demand rows in the specified time range. Check start/end.")

        self.time_buckets = demand_ts["time_bucket"].to_list()
        self.D = demand_ts["demand"].to_numpy(dtype=float)

        # --- Build simulator ---
        self.sim = FulfillmentSim(
            fc_ids=fc_ids,
            capacities=capacities,
            shipping_cost_per_order=shipping_cost_per_order,
            late_penalty_per_order=late_penalty_per_order,
            overflow_penalty_per_order=overflow_penalty_per_order,
            queue_variance_penalty=queue_variance_penalty,
        )

        self.fc_ids = list(fc_ids)
        self.n_fc = len(self.fc_ids)

        self.stress_threshold = float(stress_threshold)
        self.rolling_window = int(rolling_window)

        # --- Gym spaces ---
        self.action_space = spaces.Discrete(3)

        obs_low = np.zeros(8, dtype=np.float32)
        obs_high = np.array(
            [1e9, 1e9, 1e9, 1e6, 1e6, 1e6, 1e9, 1e9],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # internal state
        self.t = 0
        self.recent_D = []

    # -------------------------
    # Routing rules (actions)
    # -------------------------
    def _route_cheapest(self, total_demand: float) -> np.ndarray:
        # route all to FC with minimum shipping cost
        j = int(np.argmin(self.sim.ship))
        routed = np.zeros(self.n_fc, dtype=float)
        routed[j] = total_demand
        return routed

    def _route_capacity_split(self, total_demand: float) -> np.ndarray:
        cap = np.maximum(self.sim.cap, 1e-9)
        frac = cap / cap.sum()
        return total_demand * frac

    def _route_threshold_stress(self, total_demand: float) -> np.ndarray:
        """
        Avoid FCs that are too stressed based on current backlog ratio:
          stress_i = q_i / cap_i

        If stress_i > threshold, exclude it.
        Then route all demand to the cheapest among remaining.
        If all are excluded, fall back to capacity split.
        """
        cap = np.maximum(self.sim.cap, 1e-9)
        stress = self.sim.q / cap

        feasible = np.where(stress <= self.stress_threshold)[0]
        if len(feasible) == 0:
            return self._route_capacity_split(total_demand)

        # among feasible, pick min shipping cost
        j = feasible[int(np.argmin(self.sim.ship[feasible]))]
        routed = np.zeros(self.n_fc, dtype=float)
        routed[j] = total_demand
        return routed

    # -------------------------
    # Observation / Reset / Step
    # -------------------------
    def _get_obs(self, total_demand: float) -> np.ndarray:
        q = self.sim.q.astype(np.float32)  # backlog after previous processing
        cap = np.maximum(self.sim.cap, 1.0).astype(np.float32)
        util = (q / cap).astype(np.float32)

        self.recent_D.append(float(total_demand))
        if len(self.recent_D) > self.rolling_window:
            self.recent_D.pop(0)
        D_roll = float(np.mean(self.recent_D))

        obs = np.concatenate(
            [q, util, np.array([total_demand, D_roll], dtype=np.float32)]
        )
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.t = 0
        self.recent_D = []

        total_demand = float(self.D[self.t])
        obs = self._get_obs(total_demand)

        info = {"time_bucket": str(self.time_buckets[self.t])}
        return obs, info

    def step(self, action: int):
        total_demand = float(self.D[self.t])

        if action == 0:
            routed = self._route_cheapest(total_demand)
            policy_name = "cheapest_first"
        elif action == 1:
            routed = self._route_capacity_split(total_demand)
            policy_name = "capacity_split"
        elif action == 2:
            routed = self._route_threshold_stress(total_demand)
            policy_name = "threshold_stress"
        else:
            raise ValueError(f"Invalid action {action}")

        metrics = self.sim.step(routed)

        # Reward: negative step cost (simple and correct)
        reward = -float(metrics["step_cost"])

        # advance time
        self.t += 1
        terminated = (self.t >= len(self.D))
        truncated = False

        # next obs uses next bucket's demand (if not terminated)
        if not terminated:
            next_total_demand = float(self.D[self.t])
            obs = self._get_obs(next_total_demand)
            time_bucket = str(self.time_buckets[self.t])
        else:
            obs = self._get_obs(0.0)  # dummy final obs
            time_bucket = str(self.time_buckets[-1])

        info = {
            "time_bucket": time_bucket,
            "policy": policy_name,
            "action": int(action),
            "total_demand": total_demand,
            "routed_to_fc": routed.tolist(),
            **metrics,
        }

        return obs, reward, terminated, truncated, info
