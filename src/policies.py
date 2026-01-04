import numpy as np

def make_zone_fc_cost(zones, n_fc, base_fc_cost):
    """
    We do NOT have geography yet, so we need a consistent cost structure.
    This creates a deterministic cost table that is stable across runs.

    Later, we will replace this with distance-based costs using zone centroids + FC coordinates.
    """
    zones = np.array(sorted(zones), dtype=int)
    rng = np.random.default_rng(42)

    # deterministic per-zone offsets
    offsets = rng.uniform(0.0, 0.4, size=(len(zones), n_fc))
    base = np.array(base_fc_cost, dtype=float)[None, :]
    cost = base + offsets
    zone_to_idx = {z: i for i, z in enumerate(zones)}
    return zones, zone_to_idx, cost


def policy_myopic_cheapest(demand_by_zone, zones, zone_to_idx, cost_table):
    """
    Route each zone's demand to the FC with the lowest cost for that zone.
    """
    n_fc = cost_table.shape[1]
    routed = np.zeros(n_fc, dtype=float)

    for z, d in demand_by_zone.items():
        if d <= 0:
            continue
        i = zone_to_idx[z]
        fc = int(np.argmin(cost_table[i]))
        routed[fc] += d
    return routed


def policy_static_split(total_demand, capacities):
    """
    Split all demand in proportion to FC capacities.
    """
    cap = np.array(capacities, dtype=float)
    w = cap / cap.sum()
    return total_demand * w


def policy_threshold_congestion(demand_by_zone, zones, zone_to_idx, cost_table, queues, capacities, rho_threshold=1.2):
    """
    Cheapest FC unless it is too congested (queue/capacity > threshold),
    then send to the next cheapest option.
    """
    n_fc = cost_table.shape[1]
    routed = np.zeros(n_fc, dtype=float)

    rho = queues / np.maximum(np.array(capacities, dtype=float), 1e-9)

    for z, d in demand_by_zone.items():
        if d <= 0:
            continue
        i = zone_to_idx[z]
        ranked_fcs = np.argsort(cost_table[i])  # cheapest to expensive
        chosen = None
        for fc in ranked_fcs:
            if rho[fc] <= rho_threshold:
                chosen = int(fc)
                break
        if chosen is None:
            chosen = int(ranked_fcs[0])  # fallback
        routed[chosen] += d

    return routed
