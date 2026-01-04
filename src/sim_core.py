import numpy as np


class FulfillmentSim:
    """
    Discrete-time queue simulator for a small fulfillment network.

    Each timestep:
    1) New orders are routed to FCs (arrivals)
    2) Queues increase
    3) Each FC processes up to its capacity
    4) Remaining queue is backlog (late)
    5) We compute step cost components
    """

    def __init__(
        self,
        fc_ids,
        capacities,
        shipping_cost_per_order,
        late_penalty_per_order=6.0,
        overflow_penalty_per_order=12.0,
        queue_variance_penalty=0.02,
    ):
        self.fc_ids = list(fc_ids)
        self.n_fc = len(self.fc_ids)

        self.cap = np.array(capacities, dtype=float)
        self.ship = np.array(shipping_cost_per_order, dtype=float)

        self.late_penalty = float(late_penalty_per_order)
        self.overflow_penalty = float(overflow_penalty_per_order)
        self.qvar_penalty = float(queue_variance_penalty)

        self.reset()

    def reset(self):
        self.q = np.zeros(self.n_fc, dtype=float)
        self.total_cost = 0.0
        self.total_arrivals = 0.0
        self.total_late = 0.0
        self.total_processed = 0.0

    def step(self, routed_to_fc):
        routed_to_fc = np.array(routed_to_fc, dtype=float)
        if routed_to_fc.shape[0] != self.n_fc:
            raise ValueError(f"routed_to_fc must have length {self.n_fc}")

        arrivals = float(routed_to_fc.sum())
        self.total_arrivals += arrivals

        # Add arrivals to queues
        self.q += routed_to_fc

        # Process up to capacity
        processed = np.minimum(self.q, self.cap)
        self.q -= processed

        backlog = self.q.copy()

        # Define "late" as remaining backlog after processing this timestep
        late = float(backlog.sum())
        self.total_late += late
        self.total_processed += float(processed.sum())

        # Cost components
        ship_cost = float((routed_to_fc * self.ship).sum())
        late_cost = late * self.late_penalty

        # Overflow penalty starts when any FC exceeds 2x its per-step capacity
        rho = backlog / np.maximum(self.cap, 1e-9)
        overflow = float(np.maximum(rho - 2.0, 0.0).sum())
        overflow_cost = overflow * self.overflow_penalty

        # Stability penalty: encourages balanced queues
        qvar_cost = float(np.var(backlog) * self.qvar_penalty)

        step_cost = ship_cost + late_cost + overflow_cost + qvar_cost
        self.total_cost += step_cost

        metrics = {
            "arrivals": arrivals,
            "processed": float(processed.sum()),
            "late": late,
            "ship_cost": ship_cost,
            "late_cost": late_cost,
            "overflow_cost": overflow_cost,
            "qvar_cost": qvar_cost,
            "step_cost": step_cost,
        }

        # convenient per-FC queue logging
        for i, fc in enumerate(self.fc_ids):
            metrics[f"q_{fc}"] = float(backlog[i])

        return metrics
