"""
CMC GDPS data fetcher.

Bypasses Herbie and downloads directly from the MSC Datamart.

URL layout (verified against live directory listings on 2026-11-22, after the
MSC 2026-04-14 infrastructure migration and the GDPS 10.0.0 upgrade; the old
model_gem_global/.../CMC_glb_* layout returns 404 everywhere):

    https://{host}/{YYYYMMDD}/WXO-DD/model_gdps/15km/{HH}/{hhh}/
        {YYYYMMDD}T{HH}Z_MSC_GDPS_{VAR}_{LEVEL}_LatLon0.15_PT{hhh}H.grib2

    host      = dd.weather.gc.ca | dd.meteo.gc.ca | dd2.weather.gc.ca
                (all three served identical listings on 2026-11-22)
    YYYYMMDD  = run init date. The dated tree keeps ~30 days and also holds
                today's runs. (/today/model_gdps/15km/... exists too but only
                carries the current UTC day, so it can't reach yesterday's 12Z.)
    HH        = run hour, 00 or 12
    hhh       = forecast hour, 3 digits, hourly 000-240

Fields pulled (14 files per valid time), exact {VAR}_{LEVEL} tokens:

    CAPE_Sfc                          surface-based CAPE
    CIN_Sfc                           surface-based CIN
    WindU_AGL-10m,   WindV_AGL-10m    10 m winds
    WindU_IsbL-1000, WindV_IsbL-1000  isobaric winds; level is zero-padded
    WindU_IsbL-0925, WindV_IsbL-0925  to 4 digits (IsbL-0500, not IsbL-500)
    WindU_IsbL-0850, WindV_IsbL-0850
    WindU_IsbL-0700, WindV_IsbL-0700
    WindU_IsbL-0500, WindV_IsbL-0500

    e.g. https://dd.weather.gc.ca/20261122/WXO-DD/model_gdps/15km/00/024/
             20261122T00Z_MSC_GDPS_CAPE_Sfc_LatLon0.15_PT024H.grib2

Each file holds one GRIB2 message on a global 0.15 deg grid (2400 x 1201,
lat -90..90 ascending, lon 0..359.85). Fields are validated after opening by
their WMO GRIB2 codes (discipline/category/number) and level, not by cfgrib's
name, because cfgrib/ecCodes decodes CAPE and CIN as 'unknown': MSC encodes
them as 0/7/6 and 0/7/7 (standard WMO CAPE/CIN, J/kg) with the surface level
value left missing, which ecCodes' cape/cin concepts don't match. Observed on
2026-11-22 00Z F024: CAPE 0..5995 J/kg, CIN -1113..0 J/kg (already negative,
matching scp_math.cin_factor's convention). Winds decode normally as
u10/v10 (heightAboveGround 10) and u/v (isobaricInhPa).

SRH is derived from pressure-level winds (same approach as ECMWF) because
GDPS does not publish storm-relative helicity. Standard-atmosphere heights
are used for the geopotential proxy.
"""

import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import cfgrib

import scp_math


CMC_AVAILABILITY_LAG_HOURS = 5   # the HEAD probe confirms availability anyway
CMC_HOSTS = (
    "https://dd.weather.gc.ca",
    "https://dd.meteo.gc.ca",
    "https://dd2.weather.gc.ca",
)
HTTP_TIMEOUT_S = 120
USER_AGENT = "SCP_Agreement/2 (python-urllib)"

# Pressure levels and approximate ISA heights (m) for SRH derivation
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]
ISA_HEIGHTS_M    = [110,   780,  1500, 3000, 5570]

# key -> (VAR, LEVEL token, (discipline, category, number), (level coord, value))
FIELDS = {
    "cape": ("CAPE",  "Sfc",     (0, 7, 6), ("surface", None)),
    "cin":  ("CIN",   "Sfc",     (0, 7, 7), ("surface", None)),
    "u10":  ("WindU", "AGL-10m", (0, 2, 2), ("heightAboveGround", 10)),
    "v10":  ("WindV", "AGL-10m", (0, 2, 3), ("heightAboveGround", 10)),
}
for _p in PRESSURE_LEVELS:
    FIELDS[f"u{_p}"] = ("WindU", f"IsbL-{_p:04d}", (0, 2, 2), ("isobaricInhPa", _p))
    FIELDS[f"v{_p}"] = ("WindV", f"IsbL-{_p:04d}", (0, 2, 3), ("isobaricInhPa", _p))

_CODE_KEYS = ("discipline", "parameterCategory", "parameterNumber")

_inventory_printed = False


def candidate_cmc_runs(target_valid_time: pd.Timestamp):
    """Yield (run_init, fxx) tuples newest-first, walking back up to 72 h."""
    now = pd.Timestamp.now('UTC').tz_localize(None)
    target = (target_valid_time.tz_localize(None)
              if target_valid_time.tzinfo else target_valid_time)

    newest = now.floor("12h")
    for k in range(0, 7):                       # 7 cycles = 72 h back
        run_init = newest - pd.Timedelta(hours=12 * k)
        age_hours = (now - run_init).total_seconds() / 3600.0
        if age_hours < CMC_AVAILABILITY_LAG_HOURS:
            continue
        fxx = int((target - run_init).total_seconds() / 3600.0)
        if 0 <= fxx <= 240:
            yield run_init, fxx


def _build_cmc_url(host: str, run_init: pd.Timestamp, fxx: int,
                   var: str, level: str) -> str:
    """Build the MSC Datamart GDPS URL for one variable/level/forecast hour."""
    ymd = run_init.strftime("%Y%m%d")
    hh = f"{run_init.hour:02d}"
    hhh = f"{fxx:03d}"
    filename = (f"{ymd}T{hh}Z_MSC_GDPS_{var}_{level}"
                f"_LatLon0.15_PT{hhh}H.grib2")
    return f"{host}/{ymd}/WXO-DD/model_gdps/15km/{hh}/{hhh}/{filename}"


def _request(url: str, method: str = "GET"):
    req = urllib.request.Request(url, method=method,
                                 headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise RuntimeError(f"network error: {reason}") from None


def _exists(url: str) -> bool:
    try:
        with _request(url, method="HEAD"):
            return True
    except RuntimeError:
        return False


def _download(url: str, local_path: Path):
    with _request(url) as resp, open(local_path, "wb") as fh:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def _open_field(grib_path: Path, key: str) -> xr.DataArray:
    """Open a single-message GDPS GRIB and check its WMO code and level."""
    datasets = cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": "", "read_keys": list(_CODE_KEYS)},
    )
    if not isinstance(datasets, list):
        datasets = [datasets]
    found = [(ds, vn) for ds in datasets for vn in ds.data_vars]
    if len(found) != 1:
        raise RuntimeError(f"CMC {key}: expected 1 GRIB variable, got "
                           f"{[vn for _, vn in found]}")
    ds, vn = found[0]
    da = ds[vn]
    _, _, want_code, (want_tol, want_lev) = FIELDS[key]
    code = tuple(da.attrs.get(f"GRIB_{k}") for k in _CODE_KEYS)
    tol = da.attrs.get("GRIB_typeOfLevel")
    lev = ds.coords[tol].item() if tol in ds.coords else None
    if (code != want_code or tol != want_tol
            or (want_lev is not None and lev != want_lev)):
        raise RuntimeError(
            f"CMC {key}: GRIB is {code} {tol}={lev}, expected {want_code} "
            f"{want_tol}={want_lev} (MSC may have changed the encoding)")
    da = da.squeeze(drop=True)
    da.attrs["_level"] = f"{tol}={lev}" if want_lev is not None else tol
    return da


def _print_inventory(arrays: dict):
    print("[CMC] GRIB inventory (key: cfgrib name | WMO d/c/n | level | "
          "units | min..max):")
    for key, da in arrays.items():
        a = da.attrs
        code = "/".join(str(a.get(f"GRIB_{k}")) for k in _CODE_KEYS)
        units = a.get("GRIB_units", "?")
        if units == "unknown" and key in ("cape", "cin"):
            units = "J kg**-1 (WMO 0/7)"
        vals = da.values
        print(f"[CMC]   {key:5s}: {da.name:7s} | {code} | {a['_level']:22s} | "
              f"{units} | {np.nanmin(vals):.1f}..{np.nanmax(vals):.1f}")


def _find_run(target: pd.Timestamp):
    """HEAD-probe CAPE for each candidate run/host; return the first hit."""
    candidates = list(candidate_cmc_runs(target))
    if not candidates:
        raise RuntimeError(f"no GDPS run within F000-F240 of {target}")
    for run_init, fxx in candidates:
        for host in CMC_HOSTS:
            var, lvl = FIELDS["cape"][:2]
            if _exists(_build_cmc_url(host, run_init, fxx, var, lvl)):
                return run_init, fxx, host
    tried = ", ".join(f"{r:%d/%HZ}" for r, _ in candidates)
    raise RuntimeError(f"no GDPS run had CAPE for {target} "
                       f"(tried runs {tried} on {len(CMC_HOSTS)} hosts)")


def fetch_cmc(target_valid_time: pd.Timestamp) -> xr.Dataset:
    """
    Fetch CMC GDPS fields and assemble into the SCP_Agreement Dataset schema
    (mlcape, mlcin, srh_03, shear_06 on latitude/longitude).
    """
    global _inventory_printed
    target = pd.Timestamp(target_valid_time)

    run_init, fxx, host = _find_run(target)
    print(f"[CMC] run={run_init} F{fxx:03d} -> valid {target} ({host})")

    with tempfile.TemporaryDirectory(prefix="cmc_") as td:
        tmp_dir = Path(td)
        arrays = {}
        for key, (var, lvl, _, _) in FIELDS.items():
            url = _build_cmc_url(host, run_init, fxx, var, lvl)
            local = tmp_dir / url.rsplit("/", 1)[-1]
            try:
                _download(url, local)
            except RuntimeError as e:
                raise RuntimeError(f"CMC {var}_{lvl}: {e} ({url})") from None
            arrays[key] = _open_field(local, key)

        if not _inventory_printed:
            _print_inventory(arrays)
            _inventory_printed = True

        # Materialize arrays while temp files still exist
        cape = arrays["cape"]
        lat = cape["latitude"].values
        lon = cape["longitude"].values
        cape_arr = cape.values
        cin_arr  = arrays["cin"].values
        u10_arr  = arrays["u10"].values
        v10_arr  = arrays["v10"].values
        u_pl = np.stack([arrays[f"u{p}"].values for p in PRESSURE_LEVELS])
        v_pl = np.stack([arrays[f"v{p}"].values for p in PRESSURE_LEVELS])
        del arrays, cape

    # --- SUBSET to CONUS region BEFORE the expensive SRH derivation ---
    # The full CMC grid is global (~2.9M points); CONUS is ~4% of that.
    # ~25x speedup on grid_derive_srh_and_shear.
    CONUS_LAT = (20.0, 55.0)
    CONUS_LON = (-130.0, -60.0)
    lat, lon, subbed = scp_math.subset_to_region(
        lat, lon, CONUS_LAT, CONUS_LON,
        {
            "u_pl":  u_pl,
            "v_pl":  v_pl,
            "cape":  cape_arr,
            "cin":   cin_arr,
            "u10":   u10_arr,
            "v10":   v10_arr,
        },
    )
    u_pl = subbed["u_pl"]
    v_pl = subbed["v_pl"]
    cape_arr = subbed["cape"]
    cin_arr  = subbed["cin"]
    u10_arr  = subbed["u10"]
    v10_arr  = subbed["v10"]
    print(f"[CMC] subset to CONUS: {u_pl.shape}")

    # MSC leaves CAPE and CIN missing (GRIB bitmap) wherever no parcel is
    # buoyant; the product never contains an explicit 0 (checked 2026-11-22:
    # 52% of CONUS missing, 0% zero, CAPE/CIN masks identical). Missing CAPE
    # therefore means CAPE = 0. Without this, CMC drops out of the mean and
    # agreement count wherever it's stable, and regridding smears the NaNs.
    # CIN stays NaN there: main.py averages CIN with skipna, so other models'
    # CIN is used where CMC has no parcel.
    cape_arr = np.nan_to_num(cape_arr, nan=0.0)

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
                 "winds using ISA heights. Downloaded directly from MSC "
                 "Datamart (model_gdps/15km).",
    })

    print(f"[CMC] Successfully assembled dataset")
    return ds_out
