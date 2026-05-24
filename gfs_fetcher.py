"""
GFS data fetcher.

Pulls fields needed for the SCP_Agreement product from the most recent
GFS run that can forecast the target valid time. Returns an xarray
Dataset with standardized variable names.

Fields pulled (from GFS pgrb2.0p25):
    - MLCAPE  (180-0 mb above ground, "pressureFromGroundLayer" type)
    - MLCIN   (180-0 mb above ground, same)
    - 0-3km storm-relative helicity (HLCY 3000-0 m above ground)
    - 0-6km bulk shear u/v (VUCSH/VVCSH 0-6000 m above ground)

Strategy:
    - Download a subset GRIB file containing only these 5 fields in one
      Herbie call (combined regex search).
    - Open the local file directly with cfgrib.open_datasets() to avoid
      Herbie's internal cache-path quirks that were failing before.
    - cfgrib returns a list of Datasets (one per "hypercube" of compatible
      level types). Iterate and match each desired field by GRIB shortName
      plus level metadata.

LFC is NOT in standard GFS GRIB output, so the LFC factor correction
is deferred to a later version. v1 uses CIN only.
"""

import pandas as pd
import numpy as np
import xarray as xr
import cfgrib
from herbie import Herbie


# Combined regex matching all 5 fields we need
GFS_COMBINED_SEARCH = (
    r":CAPE:180-0 mb above ground:"
    r"|:CIN:180-0 mb above ground:"
    r"|:HLCY:3000-0 m above ground:"
    r"|:VUCSH:0-6000 m above ground:"
    r"|:VVCSH:0-6000 m above ground:"
)


def find_latest_gfs_run(target_valid_time: pd.Timestamp):
    """
    Find the most recent GFS cycle that can forecast the target valid time.

    GFS runs at 00, 06, 12, 18 UTC and we conservatively assume ~5 hours of
    processing lag before files are reliably available on NOMADS/AWS.

    Returns
    -------
    (run_init, fxx) : (pd.Timestamp, int)
    """
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


def _open_grib_as_list(grib_path):
    """
    Open a GRIB file with cfgrib and always return a list of Datasets.
    """
    result = cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": ""},
    )
    if not isinstance(result, list):
        result = [result]
    return result


def _attrs(da):
    """Convenience to grab GRIB metadata attrs from a DataArray."""
    return da.attrs


def _matches_level(da, type_of_level, top_level, bottom_level=0):
    """
    Check if a DataArray's GRIB level metadata matches the target layer.

    GRIB stores topLevel/bottomLevel in different units depending on the
    type-of-level (mb*100 for pressureFromGroundLayer, meters for
    heightAboveGroundLayer). We test both common encodings to be safe.
    """
    a = da.attrs
    if a.get("GRIB_typeOfLevel") != type_of_level:
        return False
    top = a.get("GRIB_topLevel")
    bot = a.get("GRIB_bottomLevel")
    # Accept multiple plausible unit encodings (e.g. 180 vs 18000 for mb)
    top_match = top in (top_level, top_level * 100)
    bot_match = bot == bottom_level
    return top_match and bot_match


def _find_field(datasets, short_name_options, level_check):
    """
    Search a list of cfgrib Datasets for a field matching a short name
    AND a level criterion. Returns the matched DataArray or None.
    """
    for ds in datasets:
        for var_name in ds.data_vars:
            if var_name.lower() not in short_name_options:
                continue
            da = ds[var_name]
            if level_check(da):
                return da.squeeze()
    return None


def _debug_dump(datasets, label):
    """Print every variable + level metadata from each Dataset for debugging."""
    print(f"[GFS DEBUG] ===== {label} =====")
    print(f"[GFS DEBUG] Total datasets: {len(datasets)}")
    for i, ds in enumerate(datasets):
        for vn in ds.data_vars:
            attrs = ds[vn].attrs
            tol = attrs.get("GRIB_typeOfLevel", "?")
            top = attrs.get("GRIB_topLevel", "?")
            bot = attrs.get("GRIB_bottomLevel", "?")
            print(f"[GFS DEBUG]  [{i}] {vn} | typeOfLevel={tol} | top={top} bot={bot}")


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

    # Download subset first, fall back to full file if subset fails
    try:
        local_path = H.download(search=GFS_COMBINED_SEARCH, verbose=False)
        print(f"[GFS] Subset downloaded: {local_path}")
        datasets = _open_grib_as_list(local_path)
    except Exception as e:
        print(f"[GFS] Subset download/open failed ({e}); falling back to full file")
        local_path = H.download(verbose=False)
        print(f"[GFS] Full file downloaded: {local_path}")
        datasets = _open_grib_as_list(local_path)

    print(f"[GFS] Parsed {len(datasets)} hypercubes from GRIB")

    # --- Match each field by shortName + level metadata ---

    mlcape = _find_field(
        datasets,
        {"cape", "mlcape"},
        lambda da: _matches_level(da, "pressureFromGroundLayer", 180, 0),
    )
    mlcin = _find_field(
        datasets,
        {"cin", "mlcin"},
        lambda da: _matches_level(da, "pressureFromGroundLayer", 180, 0),
    )
    srh_03 = _find_field(
        datasets,
        {"hlcy", "helicity"},
        lambda da: _matches_level(da, "heightAboveGroundLayer", 3000, 0),
    )
    ushear = _find_field(
        datasets,
        {"vucsh", "u_shr", "ushear"},
        lambda da: _matches_level(da, "heightAboveGroundLayer", 6000, 0),
    )
    vshear = _find_field(
        datasets,
        {"vvcsh", "v_shr", "vshear"},
        lambda da: _matches_level(da, "heightAboveGroundLayer", 6000, 0),
    )

    # Validate we got everything
    found = {
        "mlcape":  mlcape,
        "mlcin":   mlcin,
        "srh_03":  srh_03,
        "ushear":  ushear,
        "vshear":  vshear,
    }
    missing = [k for k, v in found.items() if v is None]
    if missing:
        _debug_dump(datasets, "Available variables in GFS GRIB")
        raise RuntimeError(
            f"Could not find these GFS fields after parsing: {missing}. "
            f"See [GFS DEBUG] output above for what's actually in the file."
        )

    # Combine shear u/v into magnitude
    shear_06 = np.sqrt(ushear ** 2 + vshear ** 2)
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
    })

    print(f"[GFS] Successfully extracted all 4 fields")
    return ds_out
