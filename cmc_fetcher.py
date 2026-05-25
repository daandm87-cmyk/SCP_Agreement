"""
CMC GDPS data fetcher.

GDPS publishes ONE file per variable per forecast hour at
https://dd.weather.gc.ca/model_gem_global/15km/grib2/lat_lon/

For each valid time we make multiple small downloads, one per field.
Herbie's gdps support handles URL construction and download.

Fields pulled:
    - CAPE at surface
    - CIN at surface (may not be in open feed - handled defensively)
    - HLCY 3000-0 m above ground (0-3 km SRH, native)
    - UGRD/VGRD at 10 m above ground
    - UGRD/VGRD at 500 mb

Shear is derived from |V_500mb - V_10m|, matching GFS approach.
"""

import pandas as pd
import numpy as np
import xarray as xr
import cfgrib
from herbie import Herbie


CMC_AVAILABILITY_LAG_HOURS = 7   # GDPS posts ~5-7h after init


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


def _try_fetch_field(run_init, fxx, variable, level, optional=False):
    """
    Download a single GDPS field via Herbie. Returns DataArray or None.
    """
    search = f":{variable}:{level}:"
    try:
        H = Herbie(run_init, model="gdps", fxx=fxx,
                   product="15km/grib2/lat_lon")
        local_path = H.download(search=search, verbose=False)
        return _open_grib(local_path)
    except Exception as e:
        if optional:
            print(f"[CMC] optional field {variable}:{level} not available ({e})")
            return None
        raise RuntimeError(f"CMC fetch failed for {variable}:{level}: {e}")


def fetch_cmc(target_valid_time: pd.Timestamp) -> xr.Dataset:
    """
    Fetch CMC GDPS fields and assemble into the SCP_Agreement Dataset schema.
    """
    target = pd.Timestamp(target_valid_time)
    run_init, fxx = find_latest_cmc_run(target)
    print(f"[CMC] run={run_init} F{fxx:03d} -> valid {target}")

    cape = _try_fetch_field(run_init, fxx, "CAPE", "surface")
    if cape is None:
        raise RuntimeError("CMC CAPE not found")

    cin = _try_fetch_field(run_init, fxx, "CIN", "surface", optional=True)

    hlcy = _try_fetch_field(run_init, fxx, "HLCY", "3000-0 m above ground")
    if hlcy is None:
        raise RuntimeError("CMC HLCY (0-3km SRH) not found")

    u10 = _try_fetch_field(run_init, fxx, "UGRD", "10 m above ground")
    v10 = _try_fetch_field(run_init, fxx, "VGRD", "10 m above ground")
    if u10 is None or v10 is None:
        raise RuntimeError("CMC 10m winds not found")

    u500 = _try_fetch_field(run_init, fxx, "UGRD", "500 mb")
    v500 = _try_fetch_field(run_init, fxx, "VGRD", "500 mb")
    if u500 is None or v500 is None:
        raise RuntimeError("CMC 500mb winds not found")

    du = u500.values - u10.values
    dv = v500.values - v10.values
    shear_06_arr = np.sqrt(du * du + dv * dv)

    if cin is None:
        cin_arr = np.full_like(cape.values, np.nan, dtype=float)
    else:
        cin_arr = cin.values

    lat = (cape["latitude"].values if "latitude" in cape.coords
           else cape["lat"].values)
    lon = (cape["longitude"].values if "longitude" in cape.coords
           else cape["lon"].values)

    ds_out = xr.Dataset(
        data_vars={
            "mlcape":   (("latitude", "longitude"), cape.values),
            "mlcin":    (("latitude", "longitude"), cin_arr),
            "srh_03":   (("latitude", "longitude"), hlcy.values),
            "shear_06": (("latitude", "longitude"), shear_06_arr),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    ds_out.attrs.update({
        "model": "cmc",
        "run_init": str(run_init),
        "forecast_hour": int(fxx),
        "valid_time": str(target),
        "notes": "CAPE=surface, shear=|V500-V10|, CIN may be NaN.",
    })

    print(f"[CMC] Successfully assembled dataset")
    return ds_out
