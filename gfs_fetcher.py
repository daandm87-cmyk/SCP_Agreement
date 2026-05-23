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
    now = pd.Timestamp.now('UTC').tz_localize(None)
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


def fetch_gfs(target_valid_time: pd.Timestamp) -> xr.Dataset:
    """
    Fetch all GFS fields needed for SCP at target_valid_time.

    Downloads the full GRIB2 file and extracts needed fields to avoid
    cfgrib caching issues on GitHub Actions.

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

    # Download the full file and open it once (avoids cfgrib caching issues)
    try:
        ds_full = H.xarray(remove_grib=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch GFS GRIB2 file: {e}")

    # Extract and rename the fields we need by looking for them by name
    fields = {}
    
    # MLCAPE
    cape_var = None
    for var in ds_full.data_vars:
        if 'cape' in var.lower():
            cape_var = var
            break
    if cape_var:
        fields['mlcape'] = ds_full[cape_var].squeeze()
    else:
        raise RuntimeError("MLCAPE not found in GFS output")

    # MLCIN
    cin_var = None
    for var in ds_full.data_vars:
        if 'cin' in var.lower():
            cin_var = var
            break
    if cin_var:
        fields['mlcin'] = ds_full[cin_var].squeeze()
    else:
        raise RuntimeError("MLCIN not found in GFS output")

    # 0-3km SRH (HLCY)
    srh_var = None
    for var in ds_full.data_vars:
        if 'hlcy' in var.lower() or 'helicity' in var.lower():
            srh_var = var
            break
    if srh_var:
        fields['srh_03'] = ds_full[srh_var].squeeze()
    else:
        raise RuntimeError("0-3km SRH (HLCY) not found in GFS output")

    # 0-6km shear u/v components
    ushear_var = None
    vshear_var = None
    for var in ds_full.data_vars:
        if 'vucsh' in var.lower():
            ushear_var = var
        elif 'vvcsh' in var.lower():
            vshear_var = var
    
    if ushear_var and vshear_var:
        ushear = ds_full[ushear_var].squeeze()
        vshear = ds_full[vshear_var].squeeze()
        shear_06 = np.sqrt(ushear ** 2 + vshear ** 2)
        shear_06.name = "shear_06"
        fields['shear_06'] = shear_06
    else:
        raise RuntimeError("0-6km shear (VUCSH/VVCSH) not found in GFS output")

    # Combine into output Dataset
    ds = xr.Dataset({
        "mlcape":   fields["mlcape"],
        "mlcin":    fields["mlcin"],
        "srh_03":   fields["srh_03"],
        "shear_06": fields["shear_06"],
    })

    ds.attrs.update({
        "model": "gfs",
        "run_init": str(run_init),
        "forecast_hour": int(fxx),
        "valid_time": str(target),
    })
    return ds
