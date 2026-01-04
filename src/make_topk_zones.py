import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data_processed/demand_yellow.parquet")
    ap.add_argument("--k", type=int, default=30, help="keep top K zones; rest -> OTHER")
    ap.add_argument("--outfile", default="data_processed/demand_yellow_top30.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.infile)

    # total demand per zone
    zone_totals = df.groupby("zone")["demand"].sum().sort_values(ascending=False)
    top_zones = set(zone_totals.head(args.k).index.tolist())

    df["zone_group"] = df["zone"].apply(lambda z: int(z) if z in top_zones else -1)  # -1 = OTHER

    # aggregate again after grouping
    out = (
        df.groupby(["time_bucket", "zone_group"], as_index=False)["demand"].sum()
        .sort_values(["time_bucket", "zone_group"])
    )

    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.outfile, index=False)

    print(f"Saved: {args.outfile}")
    print(f"Top-K zones: {args.k} (+ OTHER)")
    print("Rows:", len(out))
    print("Groups:", out["zone_group"].nunique())
    print("Time range:", out["time_bucket"].min(), "->", out["time_bucket"].max())

if __name__ == "__main__":
    main()
