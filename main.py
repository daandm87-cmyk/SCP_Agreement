"""
SCP_Agreement: main pipeline.

Usage:
    python main.py --valid-time 2026-05-25T00:00:00
    python main.py                   # defaults to next Monday 00Z

Pipeline:
    1. Fetch GFS  (mlcape, mlcin, srh_03, shear_06)
    2. Fetch ECMWF (mlcape proxy, srh_03 derived, shear_06 derived; CIN=NaN)
    3. Regrid both to a common 0.25 deg CONUS grid.
    4. Compute SCP_fixed per model.
    5. Multi-model mean SCP, then apply CIN factor correction using GFS CIN.
    6. Compute agreement count (#models with SCP_fixed > threshold).
    7. Render map.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import scp_math
import gfs_fetcher
import ecmwf_fetcher
import render


# Common output grid (CONUS, 0.25 deg)
GRID_LATS = np.arange(24.0, 50.25, 0.25)
GRID_LONS = np.arange(-125.0, -65.0, 0.25)

# Agreement threshold: a model "agrees" at a point if its SCP_fixed > this
AGREEMENT_THRESHOLD = 2.0


def default_valid_time() -> pd.Timestamp:
    """
    Default target: the next Monday at 00Z (capturing Sunday peak
    convection in the central US).
    """
    now = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    # weekday(): Mon=0 ... Sun=6
    days_until_monday = (0 - now.weekday()) % 7
    if days_until_monday == 0 and now.hour >= 0:
        days_until_monday = 7
    monday = (now + pd.Timedelta(days=days_until_monday)).normalize()
    return monday


def normalize_lons(lons):
    """Convert longitudes from [0, 360) to [-180, 180) if needed."""
    lons = np.asarray(lons)
    return np.where(lons > 180, lons - 360, lons)


def regrid_to_common(ds: xr.Dataset) -> xr.Dataset:
    """
    Interpolate a model Dataset to the common CONUS grid.
    Handles both [0, 360) and [-180, 180) longitude conventions.
    Handles descending or ascending latitude.
    """
    # Detect lat/lon coord names
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    # Normalize longitudes to [-180, 180)
    lons = ds[lon_name].values
    lons_norm = normalize_lons(lons)
    ds = ds.assign_coords({lon_name: lons_norm})
    ds = ds.sortby(lon_name)

    # Ensure latitudes ascending
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.sortby(lat_name)

    # Interpolate
    out = ds.interp(
        {lat_name: GRID_LATS, lon_name: GRID_LONS},
        method="linear",
    )
    # Standardize coordinate names
    if lat_name != "latitude":
        out = out.rename({lat_name: "latitude"})
    if lon_name != "longitude":
        out = out.rename({lon_name: "longitude"})
    return out


def compute_per_model_scp(ds: xr.Dataset) -> xr.DataArray:
    """Per-model SCP_fixed array."""
    return xr.apply_ufunc(
        scp_math.compute_scp_fixed,
        ds["mlcape"], ds["srh_03"], ds["shear_06"],
        dask="forbidden",
    )


def run_pipeline(target_valid_time: pd.Timestamp):
    print(f"\n=== SCP_Agreement pipeline ===")
    print(f"Target valid time: {target_valid_time}")

    # --- Fetch ---
    try:
        gfs = gfs_fetcher.fetch_gfs(target_valid_time)
    except Exception as e:
        print(f"[ERROR] GFS fetch failed: {e}")
        raise

    try:
        ecmwf = ecmwf_fetcher.fetch_ecmwf(target_valid_time)
    except Exception as e:
        print(f"[WARN] ECMWF fetch failed: {e}")
        print("[WARN] Proceeding with GFS only (no agreement layer)")
        ecmwf = None

    # --- Regrid ---
    print("[main] regridding to common grid...")
    gfs_g = regrid_to_common(gfs)
    ecmwf_g = regrid_to_common(ecmwf) if ecmwf is not None else None

    # --- Compute SCP_fixed per model ---
    print("[main] computing per-model SCP_fixed...")
    scp_gfs = compute_per_model_scp(gfs_g)

    model_scps = [scp_gfs]
    model_names = ["GFS"]
    if ecmwf_g is not None:
        scp_ecmwf = compute_per_model_scp(ecmwf_g)
        model_scps.append(scp_ecmwf)
        model_names.append("ECMWF")

    n_models = len(model_scps)

    # --- Multi-model mean SCP ---
    stacked = xr.concat(model_scps, dim="model")
    mean_scp = stacked.mean(dim="model")

    # --- Apply CIN factor using GFS CIN (consensus correction) ---
    # ECMWF doesn't provide CIN; we use GFS as the sole CIN voice in v1.
    print("[main] applying CIN factor correction...")
    cin_fac = xr.apply_ufunc(
        scp_math.cin_factor, gfs_g["mlcin"],
        dask="forbidden",
    )
    mscp = mean_scp * cin_fac

    # --- Agreement count: # models above threshold at each point ---
    agreement = sum((scp > AGREEMENT_THRESHOLD).astype(int) for scp in model_scps)

    # --- Render ---
    print("[main] rendering map...")
    title_extra = (
        f"Models: {', '.join(model_names)}  "
        f"|  GFS run {gfs.attrs.get('run_init','?')}"
    )
    if ecmwf is not None:
        title_extra += f"  |  ECMWF run {ecmwf.attrs.get('run_init','?')}"

    render.render_map(
        mscp.values,
        agreement.values,
        n_models,
        lats=GRID_LATS,
        lons=GRID_LONS,
        valid_time_str=str(target_valid_time),
        title_extra=title_extra,
    )

    print("=== done ===\n")


def parse_args():
    p = argparse.ArgumentParser(description="SCP_Agreement multi-model map")
    p.add_argument(
        "--valid-time",
        type=str,
        default=None,
        help="Target valid time (UTC), ISO format e.g. 2026-05-25T00:00:00. "
             "Default: next Monday 00Z.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.valid_time:
        target = pd.Timestamp(args.valid_time)
    else:
        target = default_valid_time()

    run_pipeline(target)


if __name__ == "__main__":
    main()
