"""
GFS data fetcher.

Pulls fields needed for the SCP_Agreement product from the most recent
GFS run that can forecast the target valid time. Returns an xarray
Dataset with standardized variable names.

Fields pulled (from GFS pgrb2.0p25):
    - MLCAPE  (180-0 mb above ground)
    - MLCIN   (180-0 mb above ground)
    - 0-3km storm-relative helicity (HLCY 3000-0 m above ground)
    - 0-6km bulk shear u/v components (VUCSH/VVCSH)

This version logs detailed inventory diagnostics so we can see exactly
what's in the file if anything is missing.

LFC is NOT in standard GFS GRIB output, so the LFC factor correction
is deferred to a later version. v1 uses CIN only.
"""

import pandas as pd
import numpy as np
import xarray as xr
import cfgrib
from herbie import Herbie


# Several alternative search strings for the 0-6 km bulk shear lines.
# The exact text in GFS inventories has varied across GFS versions, so we
# try a list and use whichever yields a match.
SHEAR_SEARCH_OPTIONS = [
    # Standard pgrb2.0p25 notation
    r":VUCSH:6000-0 m above ground:|:VVCSH:6000-0 m above ground:",
    r":VUCSH:0-6000 m above ground:|:VVCSH:0-6000 m above ground:",
    # Looser variants
    r":VUCSH:.*6000.*m above ground:|:VVCSH:.*6000.*m above ground:",
    # Even looser - any VUCSH/VVCSH at all (last resort, returns all layers)
    r":VUCSH:|:VVCSH:",
]

# Base search for the non-shear fields
GFS_BASE_SEARCH = (
    r":CAPE:180-0 mb above ground:"
    r"|:CIN:180-0 mb above ground:"
    r"|:HLCY:3000-0 m above ground:"
)


def find_latest_gfs_run(target_valid_time: pd.Timestamp):
    """
    Most recent GFS cycle that can forecast the target valid time.
    GFS runs at 00, 06, 12, 18 UTC, with ~5 hours processing lag.
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
    """Open a GRIB file with cfgrib and always return a list of Datasets."""
    result = cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": ""},
    )
    if not isinstance(result, list):
        result = [result]
    return result


def _dump_inventory_diagnostics(H):
    """
    Print every inventory entry matching wind/shear/CAPE-related patterns
    so we can see exactly what's in this GFS run's GRIB file.
    """
    try:
        inv = H.inventory()
    except Exception as e:
        print(f"[GFS DEBUG] Could not load inventory: {e}")
        return

    print(f"[GFS DEBUG] Inventory has {len(inv)} total entries")
    print(f"[GFS DEBUG] Inventory columns: {list(inv.columns)}")

    # Find the column with the searchable text
    search_col = None
    for cand in ("search_this", "variable", "search", "name"):
        if cand in inv.columns:
            search_col = cand
            break
    if search_col is None:
        search_col = inv.columns[-1]
        print(f"[GFS DEBUG] No conventional search column found; using '{search_col}'")
    else:
        print(f"[GFS DEBUG] Using search column: '{search_col}'")

    # Dump matches for each pattern of interest
    patterns = ["CAPE", "CIN", "HLCY", "VUCSH", "VVCSH", "VWSH", "CSH", "shear"]
    for pat in patterns:
        matches = inv[inv[search_col].astype(str).str.contains(
            pat, na=False, regex=False
        )]
        print(f"[GFS DEBUG] '{pat}' -> {len(matches)} inventory entries")
        for s in matches[search_col].head(8).tolist():
            print(f"[GFS DEBUG]      '{s}'")


def _try_download(H, search_str, label):
    """
    Attempt a Herbie subset download with a search string. Returns the
    local file path on success, None on failure.
    """
    try:
        local_path = H.download(search=search_str, verbose=False)
        print(f"[GFS] {label}: subset downloaded -> {local_path}")
        return local_path
    except Exception as e:
        print(f"[GFS] {label}: subset failed ({e})")
        return None


def _find_field_by_name(datasets, short_name_options):
    """
    Find a field by short name. We rely on the search regex having already
    filtered to the right layer so name alone is sufficient.
    """
    for ds in datasets:
        for var_name in ds.data_vars:
            if var_name.lower() in {s.lower() for s in short_name_options}:
                return ds[var_name].squeeze()
    return None


def fetch_gfs(target_valid_time: pd.Timestamp) -> xr.Dataset:
    """
    Fetch all GFS fields needed for SCP at target_valid_time.
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

    # Always dump diagnostics so we know what's in the file
    _dump_inventory_diagnostics(H)

    # --- Step 1: download base fields (cape, cin, hlcy) ---
    base_path = _try_download(H, GFS_BASE_SEARCH, "base fields")
    if base_path is None:
        raise RuntimeError("GFS base-fields subset download failed")
    base_datasets = _open_grib_as_list(base_path)

    mlcape = _find_field_by_name(base_datasets, ["cape", "mlcape"])
    mlcin  = _find_field_by_name(base_datasets, ["cin", "mlcin"])
    srh_03 = _find_field_by_name(base_datasets, ["hlcy", "helicity"])

    if mlcape is None or mlcin is None or srh_03 is None:
        print("[GFS DEBUG] base datasets contents:")
        for i, ds in enumerate(base_datasets):
            for vn in ds.data_vars:
                print(f"[GFS DEBUG]   [{i}] {vn}")
        raise RuntimeError(
            f"Could not find base fields. "
            f"mlcape={mlcape is not None}, "
            f"mlcin={mlcin is not None}, "
            f"srh_03={srh_03 is not None}"
        )

    # --- Step 2: download shear fields, trying multiple search variants ---
    ushear = None
    vshear = None
    for i, shear_search in enumerate(SHEAR_SEARCH_OPTIONS):
        print(f"[GFS] Trying shear search variant {i+1}/{len(SHEAR_SEARCH_OPTIONS)}: "
              f"{shear_search!r}")
        shear_path = _try_download(H, shear_search, f"shear v{i+1}")
        if shear_path is None:
            continue
        shear_datasets = _open_grib_as_list(shear_path)
        print(f"[GFS] shear v{i+1}: parsed {len(shear_datasets)} hypercubes")
        for j, ds in enumerate(shear_datasets):
            for vn in ds.data_vars:
                print(f"[GFS DEBUG]   shear[{j}] {vn}")

        ushear = _find_field_by_name(shear_datasets, ["vucsh", "u_shr", "ushear"])
        vshear = _find_field_by_name(shear_datasets, ["vvcsh", "v_shr", "vshear"])

        if ushear is not None and vshear is not None:
            print(f"[GFS] shear found with variant {i+1}")
            break

    if ushear is None or vshear is None:
        raise RuntimeError(
            f"Could not find shear fields after trying {len(SHEAR_SEARCH_OPTIONS)} "
            f"search variants. ushear={ushear is not None}, "
            f"vshear={vshear is not None}. See [GFS DEBUG] above to see what's in "
            f"the inventory."
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
