"""
GFS data fetcher.

Pulls fields needed for the SCP_Agreement product from the most recent
GFS run that can forecast the target valid time. Returns an xarray
Dataset with standardized variable names.

Fields pulled (from GFS pgrb2.0p25):
    - MLCAPE  (180-0 mb above ground)
    - MLCIN   (180-0 mb above ground)
    - 0-3km storm-relative helicity (HLCY 3000-0 m above ground)
    - 0-6km bulk shear u/v (VUCSH/VVCSH 0-6000 m above ground)

LFC is NOT in standard GFS GRIB output, so the LFC factor correction
is deferred to a later version. v1 uses CIN only.
"""

import pandas as pd
import numpy as np
import xarray as xr
from herbie import Herbie


# Herbie GRIB search strings for each field we need
GFS_SEARCH = {
    "mlcape":   r":CAPE:180-0 mb above ground:",
    "mlcin":    r":CIN:180-0 mb above ground:",
    "srh_03":   r":HLCY:3000-0 m above ground:",
    "ushear":   r":VUCSH:0-6000 m above ground:",
    "vshear":   r":VVCSH:0-6000 m above ground:",
}


def find_latest_gfs_run(target_valid_time: pd.Timestamp):
    """
    Find the most recent GFS cycle that can forecast the target valid time.

    GFS runs at 00, 06, 12, 18 UTC and we conservatively assume ~5 hours of
    processing lag before files are reliably available on NOMADS.

    Returns
    -------
    (run_init, fxx) : (pd.Timestamp, int)
        Init time of the run to use, and forecast hour from that run.
    """
    now = pd.Timestamp.utcnow().tz_localize(None)
    target = target_valid_time.tz_localize(None) if target_valid_time.tzinfo else target_valid_time

    for hours_back in range(0, 60, 6):
        candidate = (now - pd.Timedelta(hours=hours_back)).floor("6h")
        # Skip if too recent (data not yet posted)
        age_hours = (now - candidate).total_seconds() / 3600.0
        if age_hours < 5:
            continue
        fxx = int((target - candidate).total_seconds() / 3600.0)
        if 0 <= fxx <= 384:
            return candidate, fxx

    raise RuntimeError(
        f"No usable GFS run found for target valid time {target_valid_time}"
    )


def _open_field(H, search_str, var_name):
    """
    Pull a single field from a Herbie object via GRIB search string.
    Returns the field as a 2D xr.DataArray named `var_name`.
    """
    ds = H.xarray(search_str, remove_grib=False)
    # Herbie may return a Dataset or list of Datasets depending on matches.
    if isinstance(ds, list):
        if len(ds) == 0:
            raise RuntimeError(f"No match for search '{search_str}'")
        ds = ds[0]

    # Pick the first data variable (there should be exactly one)
    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        raise RuntimeError(f"No data variables in result for '{search_str}'")
    arr = ds[data_vars[0]].squeeze()
    arr.name = var_name
    return arr


def fetch_gfs(target_valid_time: pd.Timestamp) -> xr.Dataset:
    """
    Fetch all GFS fields needed for SCP at target_valid_time.

    Returns
    -------
    xr.Dataset with variables:
        mlcape   (J/kg)
        mlcin    (J/kg, negative)
        srh_03   (m^2/s^2)
        shear_06 (m/s, magnitude)
    Plus coords:
        latitude, longitude
    Attributes:
        run_init, forecast_hour, valid_time, model='gfs'
    """
    target = pd.Timestamp(target_valid_time)
    run_init, fxx = find_latest_gfs_run(target)
    print(f"[GFS] run={run_init} F{fxx:03d} -> valid {target}")

    H = Herbie(
        run_init,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx,
    )

    fields = {}
    for name, search in GFS_SEARCH.items():
        fields[name] = _open_field(H, search, name)

    # Combine shear u/v into magnitude
    ushear = fields.pop("ushear")
    vshear = fields.pop("vshear")
    shear_06 = np.sqrt(ushear ** 2 + vshear ** 2)
    shear_06.name = "shear_06"

    ds = xr.Dataset({
        "mlcape":  fields["mlcape"],
        "mlcin":   fields["mlcin"],
        "srh_03":  fields["srh_03"],
        "shear_06": shear_06,
    })

    ds.attrs.update({
        "model": "gfs",
        "run_init": str(run_init),
        "forecast_hour": int(fxx),
        "valid_time": str(target),
    })
    return ds
