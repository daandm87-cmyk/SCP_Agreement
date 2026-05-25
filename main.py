"""
SCP_Agreement: main pipeline (v2).

Loops over Day 1 through Day N (default 10) at 00Z valid time each day.
For each day:
    1. Fetch GFS, ECMWF, CMC (any model can fail without killing the run)
    2. Regrid each to a common CONUS grid
    3. Compute per-model SCP_fixed
    4. Multi-model mean SCP, apply CIN factor (from any model that provides CIN)
    5. Compute agreement count (#models with SCP_fixed > threshold)
    6. Render PNG for this day

After the day loop, build a slider HTML page that shows all rendered days.

Usage:
    python main.py                 # Day 1-10 starting tomorrow 00Z
    python main.py --days 5        # Day 1-5
    python main.py --start-date 2026-05-25  # explicit start date
"""

import argparse
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

import scp_math
import gfs_fetcher
import ecmwf_fetcher
import cmc_fetcher
import render


# Common output grid (CONUS, 0.25 deg)
GRID_LATS = np.arange(24.0, 50.25, 0.25)
GRID_LONS = np.arange(-125.0, -65.0, 0.25)

# A model "agrees" at a grid point if its SCP_fixed exceeds this value
AGREEMENT_THRESHOLD = 2.0

# Default forecast window length (days)
DEFAULT_DAYS = 10


def default_start_valid_time() -> pd.Timestamp:
    """
    Default Day 1 target = tomorrow 00Z UTC.
    """
    now = pd.Timestamp.now('UTC').tz_localize(None).floor("h")
    tomorrow_00z = (now + pd.Timedelta(days=1)).normalize()
    return tomorrow_00z


def normalize_lons(lons):
    lons = np.asarray(lons)
    return np.where(lons > 180, lons - 360, lons)


def regrid_to_common(ds: xr.Dataset) -> xr.Dataset:
    """Interpolate a model Dataset to the common CONUS grid."""
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    lons_norm = normalize_lons(ds[lon_name].values)
    ds = ds.assign_coords({lon_name: lons_norm}).sortby(lon_name)

    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.sortby(lat_name)

    out = ds.interp(
        {lat_name: GRID_LATS, lon_name: GRID_LONS},
        method="linear",
    )
    if lat_name != "latitude":
        out = out.rename({lat_name: "latitude"})
    if lon_name != "longitude":
        out = out.rename({lon_name: "longitude"})
    return out


def compute_per_model_scp(ds: xr.Dataset) -> xr.DataArray:
    return xr.apply_ufunc(
        scp_math.compute_scp_fixed,
        ds["mlcape"], ds["srh_03"], ds["shear_06"],
        dask="forbidden",
    )


def try_fetch(fetch_fn, target, model_label):
    """Wrap a fetcher call and return None on failure (with log)."""
    try:
        return fetch_fn(target)
    except Exception as e:
        print(f"[WARN] {model_label} fetch failed for {target}: {e}")
        return None


def process_day(day_num: int, target_valid_time: pd.Timestamp) -> Optional[dict]:
    """
    Fetch all models for one valid time, compute mSCP + agreement, render PNG.

    Returns a dict with metadata (for the HTML index) or None if no model
    succeeded for this day.
    """
    print(f"\n=== Day {day_num} | valid {target_valid_time} ===")

    # Fetch all three (independently - any may fail)
    fetches = {
        "GFS":   try_fetch(gfs_fetcher.fetch_gfs,     target_valid_time, "GFS"),
        "ECMWF": try_fetch(ecmwf_fetcher.fetch_ecmwf, target_valid_time, "ECMWF"),
        "CMC":   try_fetch(cmc_fetcher.fetch_cmc,     target_valid_time, "CMC"),
    }

    succeeded = {k: v for k, v in fetches.items() if v is not None}
    if not succeeded:
        print(f"[ERROR] Day {day_num}: no model fetched successfully, skipping")
        return None

    print(f"[Day {day_num}] models succeeded: {list(succeeded.keys())}")

    # Regrid
    regridded = {name: regrid_to_common(ds) for name, ds in succeeded.items()}

    # Per-model SCP_fixed
    scps = {name: compute_per_model_scp(ds) for name, ds in regridded.items()}

    # Multi-model mean
    stacked = xr.concat(list(scps.values()), dim="model")
    mean_scp = stacked.mean(dim="model")

    # CIN factor: use whichever model(s) provide CIN. Average across
    # those that do; fall back to no CIN correction if no model has it.
    cin_arrays = []
    for name, ds in regridded.items():
        cin = ds["mlcin"]
        if not np.all(np.isnan(cin.values)):
            cin_arrays.append(cin)
    if cin_arrays:
        cin_mean = xr.concat(cin_arrays, dim="src").mean(dim="src", skipna=True)
        cin_fac = xr.apply_ufunc(scp_math.cin_factor, cin_mean,
                                 dask="forbidden")
    else:
        print("[WARN] No CIN data from any model; mSCP = mean_SCP (no CIN penalty)")
        cin_fac = xr.ones_like(mean_scp)

    mscp = mean_scp * cin_fac

    # Agreement count
    agreement = sum((scp > AGREEMENT_THRESHOLD).astype(int) for scp in scps.values())
    n_models = len(scps)

    # Render this day's PNG
    title_extra = "Models: " + ", ".join(succeeded.keys())
    runs_parts = []
    for name, ds in succeeded.items():
        runs_parts.append(f"{name} {ds.attrs.get('run_init', '?')}")
    title_extra += "  |  " + "  |  ".join(runs_parts)

    png_filename = f"map_day{day_num:02d}.png"
    render.render_day_map(
        mscp.values,
        agreement.values,
        n_models,
        lats=GRID_LATS,
        lons=GRID_LONS,
        valid_time_str=str(target_valid_time),
        title_extra=title_extra,
        png_filename=png_filename,
    )

    return {
        "day_num": day_num,
        "valid_time": str(target_valid_time),
        "png_filename": png_filename,
        "models": list(succeeded.keys()),
        "n_models": n_models,
    }


def parse_args():
    p = argparse.ArgumentParser(description="SCP_Agreement multi-model map (v2)")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS,
                   help=f"How many days to forecast (default {DEFAULT_DAYS})")
    p.add_argument("--start-date", type=str, default=None,
                   help="Day 1 date (YYYY-MM-DD), 00Z. Default: tomorrow 00Z.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.start_date:
        start = pd.Timestamp(args.start_date).normalize()
    else:
        start = default_start_valid_time()

    target_valid_times = [start + pd.Timedelta(days=i) for i in range(args.days)]
    print(f"=== SCP_Agreement pipeline (v2) ===")
    print(f"Forecasting {args.days} days from {target_valid_times[0]} "
          f"to {target_valid_times[-1]}")

    day_results = []
    for i, target in enumerate(target_valid_times, start=1):
        result = process_day(i, target)
        if result is not None:
            day_results.append(result)

    if not day_results:
        raise RuntimeError("No days were rendered successfully")

    # Build the slider HTML covering all successful days
    render.build_index_html(day_results)
    print(f"\n=== done: {len(day_results)} days rendered ===")


if __name__ == "__main__":
    main()
