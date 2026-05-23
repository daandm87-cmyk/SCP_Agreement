"""
ECMWF Open Data fetcher.

ECMWF Open Data does not output SRH, bulk shear, CIN, or LFC natively.
We pull what IS available:
    - CAPE (surface)
    - 10m u/v winds
    - u/v/geopotential height on pressure levels (1000, 925, 850, 700, 500 mb)

Then we DERIVE 0-3km SRH and 0-6km bulk shear at every grid point using
scp_math.grid_derive_srh_and_shear().

CIN is not available; this fetcher returns CIN as NaN. The consensus
CIN correction applied to the multi-model mean comes from GFS only.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.opendata import Client

import scp_math


# ECMWF model and cycle info
ECMWF_RUN_HOURS = (0, 12)   # ECMWF HRES runs at 00 and 12 UTC
ECMWF_AVAILABILITY_LAG_HOURS = 7   # conservative lag before files are posted
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]


def find_latest_ecmwf_run(target_valid_time: pd.Timestamp):
    """
    Most recent ECMWF HRES cycle that can forecast the target valid time.

    Returns
    -------
    (run_init, step_hours) : (pd.Timestamp, int)
    """
    now = pd.Timestamp.utcnow().tz_localize(None)
    target = target_valid_time.tz_localize(None) if target_valid_time.tzinfo else target_valid_time

    for hours_back in range(0, 72, 12):
        candidate_day = (now - pd.Timedelta(hours=hours_back)).floor("12h")
        # Build candidate using known run hours
        for run_hour in (12, 0):
            candidate = candidate_day.replace(hour=run_hour, minute=0, second=0, microsecond=0)
            if candidate > now:
                continue
            age_hours = (now - candidate).total_seconds() / 3600.0
            if age_hours < ECMWF_AVAILABILITY_LAG_HOURS:
                continue
            step = int((target - candidate).total_seconds() / 3600.0)
            # ECMWF Open Data HRES has 3-hourly steps to 144h, then 6-hourly to 240h
            if 0 <= step <= 240 and step % 3 == 0:
                return candidate, step

    raise RuntimeError(
        f"No usable ECMWF run found for target valid time {target_valid_time}"
    )


def _retrieve(client, target_path, **kwargs):
    """Thin wrapper around client.retrieve() to localize all calls."""
    client.retrieve(target=str(target_path), **kwargs)


def fetch_ecmwf(target_valid_time: pd.Timestamp,
                surface_elev_m: float = 100.0) -> xr.Dataset:
    """
    Fetch ECMWF fields and derive SRH/shear.

    Parameters
    ----------
    target_valid_time : pd.Timestamp
    surface_elev_m : float
        Approximate surface elevation to convert geopotential height (above
        sea level) to AGL. 100m is a crude CONUS-Plains average. v2 should
        replace with a proper terrain field.

    Returns
    -------
    xr.Dataset with the same variable names as the GFS fetcher:
        mlcape   (J/kg)     -- ECMWF surface CAPE used as proxy for MLCAPE
        mlcin    (J/kg)     -- all NaN; ECMWF Open Data doesn't output CIN
        srh_03   (m^2/s^2)  -- derived from pressure-level winds
        shear_06 (m/s)      -- derived from 500mb and 10m winds
    """
    target = pd.Timestamp(target_valid_time)
    run_init, step = find_latest_ecmwf_run(target)
    print(f"[ECMWF] run={run_init} step={step}h -> valid {target}")

    client = Client(source="ecmwf")
    date_str = run_init.strftime("%Y%m%d")
    time_int = run_init.hour

    with tempfile.TemporaryDirectory(prefix="ecmwf_") as tmpdir:
        tmp = Path(tmpdir)

        # Surface CAPE and 10m winds
        sfc_path = tmp / "sfc.grib2"
        _retrieve(
            client, sfc_path,
            date=date_str, time=time_int, step=step,
            type="fc", levtype="sfc",
            param=["cape", "10u", "10v"],
        )

        # Pressure level u, v, gh
        pl_path = tmp / "pl.grib2"
        _retrieve(
            client, pl_path,
            date=date_str, time=time_int, step=step,
            type="fc", levtype="pl",
            levelist=PRESSURE_LEVELS,
            param=["u", "v", "gh"],
        )

        # Open both
        ds_sfc = xr.open_dataset(sfc_path, engine="cfgrib", backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {"typeOfLevel": "surface"},
        }, errors="ignore")
        # CAPE might come under a different typeOfLevel; do a fallback open
        try:
            cape = ds_sfc["cape"]
        except KeyError:
            # Re-open without filter to grab whatever it is
            ds_sfc_all = xr.open_dataset(sfc_path, engine="cfgrib",
                                         backend_kwargs={"indexpath": ""})
            cape = ds_sfc_all["cape"]
            ds_sfc = ds_sfc_all

        u10 = ds_sfc["u10"] if "u10" in ds_sfc else ds_sfc["10u"]
        v10 = ds_sfc["v10"] if "v10" in ds_sfc else ds_sfc["10v"]

        ds_pl = xr.open_dataset(pl_path, engine="cfgrib",
                                backend_kwargs={"indexpath": ""})

    # Pull arrays out of pressure-level dataset
    # cfgrib names: u, v, gh on isobaricInhPa coord
    u_pl = ds_pl["u"]
    v_pl = ds_pl["v"]
    gh_pl = ds_pl["gh"]

    # Get pressure-level coord and ensure descending order (surface up)
    pres = u_pl["isobaricInhPa"].values
    order = np.argsort(-pres)
    pres_sorted = pres[order]
    u_arr = u_pl.values[order]
    v_arr = v_pl.values[order]
    gh_arr = gh_pl.values[order]

    # Insert 10m wind as the lowest level, using surface_elev_m + 10 as height
    # This gives the SRH integration a near-surface anchor.
    n_levels = u_arr.shape[0]
    n_lat = u_arr.shape[1]
    n_lon = u_arr.shape[2]

    # Build augmented profile: prepend 10m winds at height = surface_elev + 10
    u_aug = np.empty((n_levels + 1, n_lat, n_lon), dtype=float)
    v_aug = np.empty_like(u_aug)
    gh_aug = np.empty_like(u_aug)
    u_aug[0] = u10.values
    v_aug[0] = v10.values
    gh_aug[0] = surface_elev_m + 10.0
    u_aug[1:] = u_arr
    v_aug[1:] = v_arr
    gh_aug[1:] = gh_arr
    pres_aug = np.concatenate([[1013.0], pres_sorted])  # dummy surface pressure

    # Derive SRH and shear gridwise
    print(f"[ECMWF] deriving SRH/shear over {n_lat}x{n_lon} grid...")
    srh_03, shear_06 = scp_math.grid_derive_srh_and_shear(
        u_aug, v_aug, gh_aug, pres_aug, surface_elev_m=surface_elev_m
    )

    # Build output Dataset matching GFS fetcher's schema
    lat = u_pl["latitude"].values
    lon = u_pl["longitude"].values

    ds = xr.Dataset(
        data_vars={
            "mlcape":   (("latitude", "longitude"), cape.values),
            "mlcin":    (("latitude", "longitude"),
                         np.full((n_lat, n_lon), np.nan, dtype=float)),
            "srh_03":   (("latitude", "longitude"), srh_03),
            "shear_06": (("latitude", "longitude"), shear_06),
        },
        coords={"latitude": lat, "longitude": lon},
    )

    ds.attrs.update({
        "model": "ecmwf",
        "run_init": str(run_init),
        "forecast_hour": int(step),
        "valid_time": str(target),
        "notes": "SRH and shear derived from pressure-level winds. "
                 "CIN not available in ECMWF Open Data (NaN)."
    })
    return ds
