from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTFIG = Path("reports/day2_figs")
OUTFIG.mkdir(parents=True, exist_ok=True)

POLICIES = ["myopic_cheapest", "static_split", "threshold_congestion"]

def load(policy):
    return pd.read_csv(f"outputs/day2/{policy}_timeseries.csv", parse_dates=["time_bucket"])

def plot_queues():
    for policy in POLICIES:
        df = load(policy)

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(df["time_bucket"], df["q_FC1"], label="FC1 queue")
        ax.plot(df["time_bucket"], df["q_FC2"], label="FC2 queue")
        ax.plot(df["time_bucket"], df["q_FC3"], label="FC3 queue")

        ax.set_title(f"Queue trajectories: {policy}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Backlog (orders)")
        ax.legend()

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(OUTFIG / f"queues_{policy}.png", dpi=200)
        plt.close()

def plot_late_and_cost():
    for policy in POLICIES:
        df = load(policy)

        # Backlog plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["time_bucket"], df["late"])
        ax.set_title(f"Backlog (late) over time: {policy}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Backlog (orders)")

        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(OUTFIG / f"late_{policy}.png", dpi=200)
        plt.close()

        # Cost plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["time_bucket"], df["step_cost"])
        ax.set_title(f"Step cost over time: {policy}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cost")

        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig(OUTFIG / f"cost_{policy}.png", dpi=200)
        plt.close()

def main():
    plot_queues()
    plot_late_and_cost()
    print("Saved Day 2 plots to reports/day2_figs/")

if __name__ == "__main__":
    main()
