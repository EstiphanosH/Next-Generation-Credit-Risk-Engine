# scripts/fred_fetch.py
"""
Fetch macroeconomic data from FRED.
This is a simple wrapper around requests to the FRED API. Set FRED_API_KEY in env.
"""
import argparse
import os
from pathlib import Path
import requests
import logging
import time

logger = logging.getLogger("fred_fetch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, api_key: str, start: str = "2000-01-01", end: str = None):
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start}
    if end:
        params["observation_end"] = end
    resp = requests.get(FRED_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data


def main(args):
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set in environment")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in args.series:
        logger.info("Fetching %s", sid)
        data = fetch_series(sid, api_key, start=args.start, end=args.end)
        out_path = out_dir / f"{sid}.json"
        with open(out_path, "w") as f:
            import json
            json.dump(data, f)
        logger.info("Saved %s to %s", sid, out_path)
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", nargs="+", required=True)
    parser.add_argument("--out-dir", default="data/external")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    main(args)
