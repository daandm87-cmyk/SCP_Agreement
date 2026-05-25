"""
CMC GDPS data fetcher.

Bypasses Herbie and downloads directly from MSC's open data portal. URL
pattern is documented and stable:

    https://dd.weather.gc.ca/model_gem_global/15km/grib2/lat_lon/
        {HH}/{hhh}/CMC_glb_{VAR}_{LVL}_latlon.15x.15_{YYYYMMDDHH}_P{hhh}.grib2

Where:
    HH         = run hour (00 or 12), 2 digits
    hhh        = forecast hour, 3 digits
    VAR        = CAPE, CIN, UGRD, VGRD, ...
    LVL        = level encoding (SFC_0, TGL_10, ISBL_500, ...)
    YYYYMMDDHH = run init date+hour

Fields pulled (14 files per valid time):
    CAPE   at SFC_0     (surface CAPE)
    CIN    at SFC_0     (surface CIN)
    UGRD,VGRD at TGL_10
    UGRD,VGRD at ISBL_1000, 925, 850, 700, 500   (10 files for SRH derivation)

SRH derived from pressure-level winds (same approach as ECMWF) since
CMC GDPS does not output HLCY natively. Standard atmosphere heights
are used for the geopotential proxy.
"""

import tempfile
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd
import numpy as np
import xarray as xr
import cfgrib

import scp_math


CMC_AVAILABILITY_LAG_HOURS = 7
CMC_BASE_URL = "https://dd.weather.gc.ca/today/model_gem_global/15km/grib2/lat_lon"

# Pressure levels and approximate ISA heights (m) for SRH derivation
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


def _build_cmc_url(run_init: pd.Timestamp, fxx: int,
                   variable: str, level: str) -> str:
    """Build the MSC GDPS file URL for one variable/level/forecast hour."""
    hh = f"{run_init.hour:02d}"
    fxx_str = f"{fxx:03d}"
    date_str = run_init.strftime("%Y%m%d%H")
    filename = (
        f"CMC_glb_{variable}_{level}_latlon.15x.15_{date_str}_P{fxx_str}.grib2"
    )
    return f"{CMC_BASE_URL}/{hh}/{fxx_str}/{filename}"


def _download_cmc_file(url: str, local_path: Path):
    """Download via urllib. Raises on HTTP/URL errors."""
    try:
        urllib.request.urlretrieve(url, str(local_path))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} for {url}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error for {url}: {e.reason}")


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


def _fetch_cmc_field(run_init, fxx, variable, level, tmp_dir):
    """
    Download one CMC field directly from MSC. Returns a DataArray.
    """
    url = _build_cmc_url(run_init, fxx, variable, level)
    local_path = tmp_dir / url.split("/")[-1]

    if not local_path.exists():
        try:
            _download_cmc_file(url, local_path)
        except RuntimeError as e:
            raise RuntimeError(f"CMC {variable}@{level}: {e}")

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

    # Use a single temp dir for this fetch
    with tempfile.TemporaryDirectory(prefix="cmc_") as td:
        tmp_dir = Path(td)

        # Single-level fields
        cape = _fetch_cmc_field(run_init, fxx, "CAPE", "SFC_0",  tmp_dir)
        cin  = _fetch_cmc_field(run_init, fxx, "CIN",  "SFC_0",  tmp_dir)
        u10  = _fetch_cmc_field(run_init, fxx, "UGRD", "TGL_10", tmp_dir)
        v10  = _fetch_cmc_field(run_init, fxx, "VGRD", "TGL_10", tmp_dir)

        # Pressure-level winds for SRH derivation
        u_pl_list = []
        v_pl_list = []
        for p in PRESSURE_LEVELS:
            u_pl_list.append(
                _fetch_cmc_field(run_init, fxx, "UGRD", f"ISBL_{p}", tmp_dir)
            )
            v_pl_list.append(
                _fetch_cmc_field(run_init, fxx, "VGRD", f"ISBL_{p}", tmp_dir)
            )

        # Materialize arrays while temp files still exist
        cape_arr = cape.values
        cin_arr  = cin.values
        u10_arr  = u10.values
        v10_arr  = v10.values
        u_pl = np.stack([u.values for u in u_pl_list], axis=0)
        v_pl = np.stack([v.values for v in v_pl_list], axis=0)

        lat = (cape["latitude"].values if "latitude" in cape.coords
               else cape["lat"].values)
        lon = (cape["longitude"].values if "longitude" in cape.coords
               else cape["lon"].values)

    # ----- after tempdir cleanup, just numpy from here -----

    n_lev, n_lat, n_lon = u_pl.shape

    # Build ISA height proxy on the grid
    gh_pl = np.zeros_like(u_pl)
    for i, h in enumerate(ISA_HEIGHTS_M):
        gh_pl[i, :, :] = h

    # Augment with 10m winds at the bottom of the profile
    surface_elev_m = 100.0
    u_aug = np.empty((n_lev + 1, n_lat, n_lon), dtype=float)
    v_aug = np.empty_like(u_aug)
    gh_aug = np.empty_like(u_aug)
    u_aug[0] = u10_arr
    v_aug[0] = v10_arr
    gh_aug[0] = surface_elev_m + 10.0
    u_aug[1:] = u_pl
    v_aug[1:] = v_pl
    gh_aug[1:] = gh_pl
    pres_aug = np.concatenate([[1013.0], np.asarray(PRESSURE_LEVELS)])

    print(f"[CMC] deriving SRH/shear over {n_lat}x{n_lon} grid...")
    srh_03, shear_06 = scp_math.grid_derive_srh_and_shear(
        u_aug, v_aug, gh_aug, pres_aug, surface_elev_m=surface_elev_m
    )

    ds_out = xr.Dataset(
        data_vars={
            "mlcape":   (("latitude", "longitude"), cape_arr),
            "mlcin":    (("latitude", "longitude"), cin_arr),
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
        "notes": "CAPE=surface, CIN=surface, SRH derived from pressure-level "
                 "winds using ISA heights. Downloaded directly from MSC.",
    })

    print(f"[CMC] Successfully assembled dataset")
    return ds_out
