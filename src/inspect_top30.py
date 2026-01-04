import pandas as pd

df = pd.read_parquet("data_processed/demand_yellow_top30.parquet")

print(df.head(10))
print("\n---")
print("Rows:", len(df))
print("Zone groups:", df["zone_group"].unique())
print("Time range:", df["time_bucket"].min(), "->", df["time_bucket"].max())
