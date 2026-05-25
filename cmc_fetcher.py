"""
CMC GDPS data fetcher.

GDPS publishes ONE file per variable per forecast hour at
https://dd.weather.gc.ca/model_gem_global/15km/grib2/lat_lon/

Herbie's GDPS support requires `variable` and `level` as constructor
kwargs (not search strings like other models).

Fields pulled (14 files per valid time):
    CAPE   at SFC_0     (surface CAPE)
    CIN    at SFC_0     (surface CIN -- CMC does ship this)
    UGRD   at TGL_10    (10m u-wind)
    VGRD   at TGL_10    (10m v-wind)
    UGRD,VGRD at ISBL_1000, 925, 850, 700, 500  (10 files for SRH derivation)

CMC GDPS does NOT ship native HLCY, so we derive 0-3 km SRH from the
pressure-level winds using scp_math.grid_derive_srh_and_shear() (same
approach as ECMWF). Standard atmosphere heights are used to avoid
fetching geopotential height files.
"""

import pandas as pd
import numpy as np
import xarray as xr
import cfgrib
from herbie import Herbie

import scp_math


CMC_AVAILABILITY_LAG_HOURS = 7   # GDPS posts ~5-7h after init

# Pressure levels and approximate ISA heights (m) for SRH derivation.
# Using standard atmosphere instead of fetching geopotential heights
# saves 5 file downloads per valid time.
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]
ISA_HEIGHTS_M    = [110,   780,  1500, 3000, 5570]


def find_latest_cmc_run(target_valid_time: pd.Timestamp):
    """Most recent CMC GDPS cycle that can forecast the target valid time."""
    now = pd.Timestamp.now('UTC').tz_localize(None)
    target = (target_valid_time.tz_localize(None)
              if target_valid_time.tzinfo else target_valid_time)

    for hours_back in range(0, 72, 12):
        candidate_day = (now - pd.Timedelta(hours=hours_back)).floor("12h")
        for run_hour in (12, 0):
            candidate = candidate_day.replace(hour=run_hour, minute=0,
                                              second=0, microsecond=0)
            if candidate > now:
                continue
            age_hours = (now - candidate).total_seconds() / 3600.0
            if age_hours < CMC_AVAILABILITY_LAG_HOURS:
                continue
            fxx = int((target - candidate).total_seconds() / 3600.0)
            if 0 <= fxx <= 240:
                return candidate, fxx

    raise RuntimeError(
        f"No usable CMC GDPS run found for target valid time {target_valid_time}"
    )


def _open_grib(grib_path):
    """Open a single-variable GRIB file and return its DataArray."""
    datasets = cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": ""},
    )
    if not isinstance(datasets, list):
        datasets = [datasets]
    if not datasets:
        return None
    ds = datasets[0]
    data_vars = list(ds.data_vars)
    if not data_vars:
        return None
    return ds[data_vars[0]].squeeze(drop=True)


def _fetch_cmc_field(run_init, fxx, variable, level):
    """
    Download one GDPS field via Herbie's variable/level kwargs API.

    Returns a DataArray. Raises on failure.
    """
    H = Herbie(
        run_init,
        model="gdps",
        product="15km/grib2/lat_lon",
        fxx=fxx,
        variable=variable,
        level=level,
    )
    local_path = H.download(verbose=False)
    arr = _open_grib(local_path)
    if arr is None:
        raise RuntimeError(f"CMC {variable}@{level}: GRIB had no usable data")
    return arr


def fetch_cmc(target_valid_time: pd.Timestamp) -> xr.Dataset:
    """
    Fetch CMC GDPS fields and assemble into the SCP_Agreement Dataset schema.
    """
    target = pd.Timestamp(target_valid_time)
    run_init, fxx = find_latest_cmc_run(target)
    print(f"[CMC] run={run_init} F{fxx:03d} -> valid {target}")

    # --- Single-level fields ---
    cape = _fetch_cmc_field(run_init, fxx, "CAPE", "SFC_0")
    cin  = _fetch_cmc_field(run_init, fxx, "CIN",  "SFC_0")
    u10  = _fetch_cmc_field(run_init, fxx, "UGRD", "TGL_10")
    v10  = _fetch_cmc_field(run_init, fxx, "VGRD", "TGL_10")

    # --- Pressure-level winds for SRH derivation ---
    u_pl_list = []
    v_pl_list = []
    for p in PRESSURE_LEVELS:
        u_pl_list.append(_fetch_cmc_field(run_init, fxx, "UGRD", f"ISBL_{p}"))
        v_pl_list.append(_fetch_cmc_field(run_init, fxx, "VGRD", f"ISBL_{p}"))

    # Stack into (n_levels, n_lat, n_lon) arrays
    u_pl = np.stack([u.values for u in u_pl_list], axis=0)
    v_pl = np.stack([v.values for v in v_pl_list], axis=0)
    n_lev, n_lat, n_lon = u_pl.shape

    # Build geopotential-height proxy array using ISA heights, broadcast
    # to the full grid shape so scp_math.grid_derive_srh_and_shear() is happy.
    gh_pl = np.zeros_like(u_pl)
    for i, h in enumerate(ISA_HEIGHTS_M):
        gh_pl[i, :, :] = h

    # --- Augment with 10m winds at the bottom of the profile ---
    surface_elev_m = 100.0
    u_aug = np.empty((n_lev + 1, n_lat, n_lon), dtype=float)
    v_aug = np.empty_like(u_aug)
    gh_aug = np.empty_like(u_aug)
    u_aug[0] = u10.values
    v_aug[0] = v10.values
    gh_aug[0] = surface_elev_m + 10.0
    u_aug[1:] = u_pl
    v_aug[1:] = v_pl
    gh_aug[1:] = gh_pl
    pres_aug = np.concatenate([[1013.0], np.asarray(PRESSURE_LEVELS)])

    print(f"[CMC] deriving SRH/shear over {n_lat}x{n_lon} grid...")
    srh_03, shear_06 = scp_math.grid_derive_srh_and_shear(
        u_aug, v_aug, gh_aug, pres_aug, surface_elev_m=surface_elev_m
    )

    # --- Pull coords from one of the fetched fields (any will do) ---
    lat = (cape["latitude"].values if "latitude" in cape.coords
           else cape["lat"].values)
    lon = (cape["longitude"].values if "longitude" in cape.coords
           else cape["lon"].values)

    ds_out = xr.Dataset(
        data_vars={
            "mlcape":   (("latitude", "longitude"), cape.values),
            "mlcin":    (("latitude", "longitude"), cin.values),
            "srh_03":   (("latitude", "longitude"), srh_03),
            "shear_06": (("latitude", "longitude"), shear_06),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    ds_out.attrs.update({
        "model": "cmc",
        "run_init": str(run_init),
        "forecast_hour": int(fxx),
        "valid_time": str(target),
        "notes": "CAPE=surface, CIN=surface, SRH derived from "
                 "pressure-level winds (1000/925/850/700/500 mb) "
                 "using ISA heights.",
    })

    print(f"[CMC] Successfully assembled dataset")
    return ds_out
