from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import requests
from tqdm import tqdm

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"


@dataclass(frozen=True)
class Month:
    year: int
    month: int  # 1..12

    def __post_init__(self):
        if not (1 <= self.month <= 12):
            raise ValueError("month must be 1..12")

    def to_fname(self) -> str:
        return f"yellow_tripdata_{self.year}-{self.month:02d}.parquet"

    def next(self) -> "Month":
        if self.month == 12:
            return Month(self.year + 1, 1)
        return Month(self.year, self.month + 1)


def months_inclusive(start: Month, end: Month):
    cur = start
    while (cur.year, cur.month) <= (end.year, end.month):
        yield cur
        cur = cur.next()


def download_file(url: str, outpath: Path, timeout=180):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if outpath.exists() and outpath.stat().st_size > 0:
        print(f"✓ exists: {outpath.name}")
        return

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        with open(outpath, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=outpath.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--outdir", default="data_raw/tlc_yellow", help="output directory")
    args = ap.parse_args()

    sy, sm = map(int, args.start.split("-"))
    ey, em = map(int, args.end.split("-"))
    start = Month(sy, sm)
    end = Month(ey, em)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading yellow taxi parquet files from {args.start} to {args.end}...")

    for m in months_inclusive(start, end):
        fname = m.to_fname()
        url = f"{BASE}/{fname}"
        outpath = outdir / fname
        download_file(url, outpath)

    print("Done.")


if __name__ == "__main__":
    main()
