import pandas as pd
import numpy as np


def main(path="data_processed/wm_transitions.parquet"):
    df = pd.read_parquet(path)

    print("Rows, Cols:", df.shape)
    print("Actions counts:\n", df["action"].value_counts().sort_index())
    print("Cost stats:\n", df["cost"].describe())
    print("Late stats:\n", df["late"].describe())

    # check vector lengths
    lens_obs = df["obs"].apply(len).value_counts().head()
    lens_next = df["next_obs"].apply(len).value_counts().head()
    print("\nobs length counts:\n", lens_obs)
    print("\nnext_obs length counts:\n", lens_next)

    # quick sanity check of queue columns
    print("\nQueue columns NA rate:")
    for c in ["q_fc1", "q_fc2", "q_fc3"]:
        print(c, float(df[c].isna().mean()))

    # show 3 sample rows
    print("\nSample rows:")
    print(df[["time_bucket", "action", "policy", "cost", "late", "q_fc1", "q_fc2", "q_fc3"]].head(10))


if __name__ == "__main__":
    main()
