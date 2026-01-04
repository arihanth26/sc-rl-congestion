import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.sim_core import FulfillmentSim
from src.policies import (
    make_zone_fc_cost,
    policy_myopic_cheapest,
    policy_static_split,
    policy_threshold_congestion
)

def load_scenario(path="data_processed/scenario.json"):
    with open(path, "r") as f:
        return json.load(f)

def main():
    scen = load_scenario()
    demand_path = scen["dataset_path"]

    df = pd.read_parquet(demand_path)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])

    # filter to run window
    tmin = pd.to_datetime(scen["run_time_min"])
    tmax = pd.to_datetime(scen["run_time_max"])
    df = df[(df["time_bucket"] >= tmin) & (df["time_bucket"] < tmax)].copy()

    fc_ids = [fc["id"] for fc in scen["fcs"]]
    capacities = [fc["capacity_per_bucket"] for fc in scen["fcs"]]
    ship_cost = scen["shipping_cost_per_order"]

    # zones list (zone_group values excluding -1 if you want, but we keep it)
    zones = sorted(df["zone_group"].unique().tolist())
    zones_arr, zone_to_idx, cost_table = make_zone_fc_cost(zones, len(fc_ids), ship_cost)

    outdir = Path("outputs/day2")
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}

    for policy_name in ["myopic_cheapest", "static_split", "threshold_congestion"]:
        sim = FulfillmentSim(
            fc_ids=fc_ids,
            capacities=capacities,
            shipping_cost_per_order=ship_cost,
            late_penalty_per_order=scen["late_penalty_per_order"],
            overflow_penalty_per_order=scen["overflow_penalty_per_order"],
            queue_variance_penalty=scen["queue_variance_penalty"]
        )
        sim.reset()

        rows = []

        for t, g in df.groupby("time_bucket"):
            # demand per zone at this timestep
            demand_by_zone = dict(zip(g["zone_group"].astype(int), g["demand"].astype(float)))
            total_demand = float(g["demand"].sum())

            if policy_name == "myopic_cheapest":
                routed = policy_myopic_cheapest(demand_by_zone, zones_arr, zone_to_idx, cost_table)

            elif policy_name == "static_split":
                routed = policy_static_split(total_demand, capacities)

            else:
                routed = policy_threshold_congestion(
                    demand_by_zone, zones_arr, zone_to_idx, cost_table,
                    queues=sim.q, capacities=capacities,
                    rho_threshold=scen["policy_threshold_rho"]
                )

            metrics = sim.step(routed)
            metrics["time_bucket"] = t
            metrics["policy"] = policy_name
            metrics["routed_fc1"] = routed[0]
            metrics["routed_fc2"] = routed[1]
            metrics["routed_fc3"] = routed[2]
            rows.append(metrics)

        out = pd.DataFrame(rows)
        out.to_csv(outdir / f"{policy_name}_timeseries.csv", index=False)

        summary = {
            "policy": policy_name,
            "time_min": str(out["time_bucket"].min()),
            "time_max": str(out["time_bucket"].max()),
            "total_arrivals": sim.total_arrivals,
            "total_processed": sim.total_processed,
            "total_late": sim.total_late,
            "late_rate": float(sim.total_late / max(sim.total_arrivals, 1e-9)),
            "total_cost": sim.total_cost,
            "avg_cost_per_order": float(sim.total_cost / max(sim.total_arrivals, 1e-9))
        }
        results[policy_name] = summary

    # save summary
    with open(outdir / "baseline_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved baseline outputs to outputs/day2/")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
