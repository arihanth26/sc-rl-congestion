from __future__ import annotations

import argparse
from pathlib import Path
import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data_raw/tlc_yellow/yellow_tripdata_*.parquet",
                    help="glob for downloaded parquet files")
    ap.add_argument("--bucket_minutes", type=int, default=5, help="time bucket size in minutes")
    ap.add_argument("--out", default="data_processed/demand_yellow.parquet", help="output parquet path")
    args = ap.parse_args()

    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET threads TO 8;")

    query = f"""
    WITH base AS (
    SELECT
        CAST(tpep_pickup_datetime AS TIMESTAMP) AS pickup_ts,
        PULocationID AS zone
    FROM read_parquet('{args.glob}')
    WHERE
        PULocationID IS NOT NULL
        AND tpep_pickup_datetime IS NOT NULL
        AND CAST(tpep_pickup_datetime AS DATE) >= DATE '2024-01-01'
        AND CAST(tpep_pickup_datetime AS DATE) <  DATE '2024-04-01'
    ),
    binned AS (
    SELECT
        -- Time bucketing in LOCAL timestamp space (no epoch/to_timestamp, so no UTC shift)
        (
        date_trunc('hour', pickup_ts)
        + (floor(extract(minute FROM pickup_ts) / {args.bucket_minutes}) * {args.bucket_minutes}) * INTERVAL '1 minute'
        ) AS time_bucket,
        zone,
        COUNT(*)::INTEGER AS demand
    FROM base
    GROUP BY 1, 2
    )
    SELECT *
    FROM binned
    ORDER BY time_bucket, zone;
    """

    print("Building demand table...")
    df = con.execute(query).df()

    df.to_parquet(outpath, index=False)
    print(f"Saved demand table to: {outpath}")
    print(f"Rows: {len(df):,} | Cols: {df.shape[1]}")
    print(f"Time range: {df['time_bucket'].min()}  ->  {df['time_bucket'].max()}")
    print(f"Unique zones: {df['zone'].nunique():,}")


if __name__ == "__main__":
    main()
