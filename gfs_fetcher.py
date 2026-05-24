"""
GFS data fetcher.

Pulls fields needed for the SCP_Agreement product from the most recent
GFS run that can forecast the target valid time.

Inventory facts (confirmed from a live GFS pgrb2.0p25 inventory dump):
    - GFS DOES output: CAPE/CIN at 180-0 mb (MLCAPE/MLCIN), HLCY 0-3km,
      UGRD/VGRD at 10m surface and at pressure levels including 500 mb.
    - GFS does NOT output VUCSH/VVCSH 0-6km bulk shear in pgrb2.0p25.

Therefore:
    - We pull MLCAPE, MLCIN, and HLCY natively.
    - We pull u/v at 10m and at 500 mb, and COMPUTE 0-6 km bulk shear as
      |V_500 - V_10|. 500 mb ≈ 5500 m AGL, close enough to 6 km for an
      agreement-focused product. This is the same approximation we use
      for ECMWF, so the two models now use consistent methodology.

We do ONE Herbie download call (combined search) to avoid the multi-call
subset-cache bug.

LFC is not in standard GFS GRIB output; LFC factor correction is v2.
"""

import pandas as pd
import numpy as np
import xarray as xr
import cfgrib
from herbie import Herbie


# Single combined search: all 7 fields in one download.
GFS_SEARCH = (
    r":CAPE:180-0 mb above ground:"
    r"|:CIN:180-0 mb above ground:"
    r"|:HLCY:3000-0 m above ground:"
    r"|:UGRD:10 m above ground:"
    r"|:VGRD:10 m above ground:"
    r"|:UGRD:500 mb:"
    r"|:VGRD:500 mb:"
)


def find_latest_gfs_run(target_valid_time: pd.Timestamp):
    now = pd.Timestamp.now('UTC').tz_localize(None)
    target = target_valid_time.tz_localize(None) if target_valid_time.tzinfo else target_valid_time

    for hours_back in range(0, 60, 6):
        candidate = (now - pd.Timedelta(hours=hours_back)).floor("6h")
        age_hours = (now - candidate).total_seconds() / 3600.0
        if age_hours < 5:
            continue
        fxx = int((target - candidate).total_seconds() / 3600.0)
        if 0 <= fxx <= 384:
            return candidate, fxx

    raise RuntimeError(
        f"No usable GFS run found for target valid time {target_valid_time}"
    )


def _level_value_from_coords(da):
    """
    Extract the level value from a DataArray's coords. cfgrib stores it
    under one of several coord names depending on the GRIB level type.
    """
    for coord in ("isobaricInhPa", "heightAboveGround",
                  "pressureFromGroundLayer", "heightAboveGroundLayer"):
        if coord in da.coords:
            try:
                return float(da.coords[coord].values)
            except (ValueError, TypeError):
                pass
    return None


def _find_field(datasets, name_options, type_of_level=None, level_value=None):
    """
    Search a list of cfgrib Datasets for a field matching:
      - short name (case-insensitive) in name_options
      - GRIB_typeOfLevel attribute (if provided)
      - level value in coords (if provided)
    """
    name_set = {n.lower() for n in name_options}
    for ds in datasets:
        for var_name in ds.data_vars:
            if var_name.lower() not in name_set:
                continue
            da = ds[var_name]

            if type_of_level is not None:
                if da.attrs.get("GRIB_typeOfLevel", "") != type_of_level:
                    continue

            if level_value is not None:
                if _level_value_from_coords(da) != level_value:
                    continue

            # Strip scalar coords so later arithmetic doesn't get confused
            return da.squeeze(drop=True)
    return None


def fetch_gfs(target_valid_time: pd.Timestamp) -> xr.Dataset:
    target = pd.Timestamp(target_valid_time)
    run_init, fxx = find_latest_gfs_run(target)
    print(f"[GFS] run={run_init} F{fxx:03d} -> valid {target}")

    H = Herbie(
        run_init,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx,
    )

    # ONE download call only.
    local_path = H.download(search=GFS_SEARCH, verbose=False)
    print(f"[GFS] Subset downloaded -> {local_path}")

    datasets = cfgrib.open_datasets(
        str(local_path),
        backend_kwargs={"indexpath": ""},
    )
    if not isinstance(datasets, list):
        datasets = [datasets]
    print(f"[GFS] Parsed {len(datasets)} hypercubes")

    # --- Find each field ---

    mlcape = _find_field(
        datasets, ["cape", "mlcape"],
        type_of_level="pressureFromGroundLayer",
    )
    mlcin = _find_field(
        datasets, ["cin", "mlcin"],
        type_of_level="pressureFromGroundLayer",
    )
    srh_03 = _find_field(
        datasets, ["hlcy", "helicity"],
        type_of_level="heightAboveGroundLayer",
    )
    u_10m = _find_field(
        datasets, ["u", "u10", "10u"],
        type_of_level="heightAboveGround", level_value=10,
    )
    v_10m = _find_field(
        datasets, ["v", "v10", "10v"],
        type_of_level="heightAboveGround", level_value=10,
    )
    u_500 = _find_field(
        datasets, ["u"],
        type_of_level="isobaricInhPa", level_value=500,
    )
    v_500 = _find_field(
        datasets, ["v"],
        type_of_level="isobaricInhPa", level_value=500,
    )

    # Validate
    found = {
        "mlcape": mlcape, "mlcin": mlcin, "srh_03": srh_03,
        "u_10m": u_10m, "v_10m": v_10m,
        "u_500": u_500, "v_500": v_500,
    }
    missing = [k for k, v in found.items() if v is None]
    if missing:
        print(f"[GFS DEBUG] Missing fields: {missing}")
        print(f"[GFS DEBUG] What we got in {len(datasets)} hypercubes:")
        for i, ds in enumerate(datasets):
            for vn in ds.data_vars:
                da = ds[vn]
                tol = da.attrs.get("GRIB_typeOfLevel", "?")
                lvl = _level_value_from_coords(da)
                print(f"[GFS DEBUG]   [{i}] {vn} | typeOfLevel={tol} | level={lvl}")
        raise RuntimeError(f"Missing GFS fields: {missing}")

    # --- Compute 0-6 km bulk shear from 500 mb and 10 m winds ---
    # 500 mb ≈ 5500 m AGL, close enough to 6 km for agreement-focused product.
    du = u_500 - u_10m
    dv = v_500 - v_10m
    shear_06 = np.sqrt(du ** 2 + dv ** 2)
    shear_06.name = "shear_06"

    ds_out = xr.Dataset({
        "mlcape":   mlcape,
        "mlcin":    mlcin,
        "srh_03":   srh_03,
        "shear_06": shear_06,
    })
    ds_out.attrs.update({
        "model": "gfs",
        "run_init": str(run_init),
        "forecast_hour": int(fxx),
        "valid_time": str(target),
        "notes": "0-6km bulk shear computed from |V_500mb - V_10m|.",
    })

    print(f"[GFS] Successfully extracted all fields")
    return ds_out
