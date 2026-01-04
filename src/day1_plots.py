import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INFILE = "data_processed/demand_yellow.parquet"
OUTDIR = Path("reports/day1_figs")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INFILE)

# 1) Total demand over time (all zones)
ts = df.groupby("time_bucket")["demand"].sum()

plt.figure()
plt.plot(ts.index, ts.values)
plt.title("Total demand over time (all zones)")
plt.xlabel("Time")
plt.ylabel("Trips (proxy for orders)")
plt.tight_layout()
plt.savefig(OUTDIR / "total_demand_timeseries.png", dpi=200)
plt.close()

# 2) Top zones by total demand
top = df.groupby("zone")["demand"].sum().sort_values(ascending=False).head(20)

plt.figure()
plt.bar(top.index.astype(str), top.values)
plt.title("Top 20 zones by total demand")
plt.xlabel("Zone")
plt.ylabel("Total trips")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTDIR / "top_zones.png", dpi=200)
plt.close()

# 3) Distribution of demand per (time_bucket, zone)
plt.figure()
plt.hist(df["demand"].values, bins=50)
plt.title("Demand per bucket per zone (distribution)")
plt.xlabel("Demand count in bucket")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUTDIR / "demand_hist.png", dpi=200)
plt.close()

print(f"Saved plots to {OUTDIR}")
print("Time range:", df["time_bucket"].min(), "->", df["time_bucket"].max())
print("Rows:", len(df), "| Zones:", df["zone"].nunique())
