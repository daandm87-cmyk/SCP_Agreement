"""
DWD ICON global data fetcher.

Downloads directly from DWD Open Data (verified against live listings on
2026-11-22):

    https://opendata.dwd.de/weather/nwp/icon/grib/{HH}/{var}/
        icon_global_icosahedral_{kind}_{YYYYMMDDHH}_{fff}[_{lev}]_{VAR}.grib2.bz2

    HH    = run hour 00/06/12/18. DWD keeps only the latest run of each hour
            (~24 h of history).
    fff   = forecast hour: hourly 000-078, then 3-hourly; 00/12Z runs reach
            180 h (Day 7.5), 06/18Z runs 120 h.
    kind  = single-level | pressure-level | time-invariant

Fields pulled per valid time (13 files, ~3 MB each, bz2-compressed GRIB2):

    single-level   CAPE_ML, U_10M, V_10M
    pressure-level {lev}_U, {lev}_V, {lev}_FI for lev in 1000/925/850/700/500
    time-invariant CLAT, CLON (cell centres; cached per run)

ICON Open Data has no CIN, so mlcin is returned as NaN (same as ECMWF).

The native grid is icosahedral/unstructured (2,949,120 cells, ~13 km). Cells
inside a padded CONUS box are linearly interpolated (one shared Delaunay
triangulation per run) onto a regular 0.25 deg CONUS grid, then 0-3 km SRH
and 0-6 km shear are derived from pressure-level winds and real geopotential
height exactly as in ecmwf_fetcher.
"""

import bz2
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

import scp_math


ICON_BASE = "https://opendata.dwd.de/weather/nwp/icon/grib"
ICON_AVAILABILITY_LAG_HOURS = 4    # the HEAD probe confirms availability
ICON_MAX_FXX = {0: 180, 12: 180, 6: 120, 18: 120}
HTTP_TIMEOUT_S = 120
USER_AGENT = "SCP_Agreement/2 (python-urllib)"
G0 = 9.80665

PRESSURE_LEVELS = [1000, 925, 850, 700, 500]

# Regular target grid (CONUS box, same extent the other fetchers subset to)
GRID_LATS = np.arange(20.0, 55.0 + 1e-6, 0.25)
GRID_LONS = np.arange(-130.0, -60.0 + 1e-6, 0.25)
_PAD_DEG = 1.0

# key -> (directory, kind, level or None, VAR, expected cfgrib name)
FIELDS = {
    "cape": ("cape_ml", "single-level", None, "CAPE_ML", "CAPE_ML"),
    "u10":  ("u_10m",   "single-level", None, "U_10M",   "u10"),
    "v10":  ("v_10m",   "single-level", None, "V_10M",   "v10"),
}
for _p in PRESSURE_LEVELS:
    FIELDS[f"u{_p}"] = ("u",  "pressure-level", _p, "U",  "u")
    FIELDS[f"v{_p}"] = ("v",  "pressure-level", _p, "V",  "v")
    FIELDS[f"z{_p}"] = ("fi", "pressure-level", _p, "FI", "z")

# run_init -> (cell index array, LinearNDInterpolator template)
_grid_cache = {}


def candidate_icon_runs(target_valid_time: pd.Timestamp):
    """Yield (run_init, fxx) newest-first over the ~24 h DWD keeps online."""
    now = pd.Timestamp.now('UTC').tz_localize(None)
    target = (target_valid_time.tz_localize(None)
              if target_valid_time.tzinfo else target_valid_time)
    newest = now.floor("6h")
    for k in range(0, 5):
        run_init = newest - pd.Timedelta(hours=6 * k)
        if (now - run_init).total_seconds() / 3600.0 < ICON_AVAILABILITY_LAG_HOURS:
            continue
        fxx = int((target - run_init).total_seconds() / 3600.0)
        if 0 <= fxx <= ICON_MAX_FXX[run_init.hour] and (fxx <= 78 or fxx % 3 == 0):
            yield run_init, fxx


def _url(run_init, fxx, directory, kind, level, var):
    run = run_init.strftime("%Y%m%d%H")
    lev = f"{level}_" if level is not None else ""
    step = "" if kind == "time-invariant" else f"{fxx:03d}_"
    return (f"{ICON_BASE}/{run_init.hour:02d}/{directory}/"
            f"icon_global_icosahedral_{kind}_{run}_{step}{lev}{var}.grib2.bz2")


def _request(url, method="GET"):
    req = urllib.request.Request(url, method=method,
                                 headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RuntimeError(f"network error: {getattr(e, 'reason', e)}") from None


def _exists(url):
    try:
        with _request(url, method="HEAD"):
            return True
    except RuntimeError:
        return False


def _download_grib(url, tmp_dir: Path) -> Path:
    local = tmp_dir / url.rsplit("/", 1)[-1].removesuffix(".bz2")
    try:
        with _request(url) as resp:
            local.write_bytes(bz2.decompress(resp.read()))
    except RuntimeError as e:
        raise RuntimeError(f"ICON {url.rsplit('/', 1)[-1]}: {e}") from None
    return local


def _open_values(path: Path, expected_name: str, level=None) -> np.ndarray:
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    if not isinstance(datasets, list):
        datasets = [datasets]
    found = [(ds, vn) for ds in datasets for vn in ds.data_vars]
    if len(found) != 1 or found[0][1] != expected_name:
        raise RuntimeError(f"ICON {path.name}: got {[v for _, v in found]}, "
                           f"expected '{expected_name}'")
    ds, vn = found[0]
    if level is not None:
        got = ds.coords["isobaricInhPa"].item() if "isobaricInhPa" in ds.coords else None
        if got != level:
            raise RuntimeError(f"ICON {path.name}: level {got}, expected {level}")
    return np.asarray(ds[vn].values, dtype=float)


def _grid_for_run(run_init, tmp_dir):
    """Cell selection + triangulation for the CONUS box, cached per run."""
    if run_init in _grid_cache:
        return _grid_cache[run_init]
    clat = _open_values(_download_grib(
        _url(run_init, 0, "clat", "time-invariant", None, "CLAT"), tmp_dir), "tlat")
    clon = _open_values(_download_grib(
        _url(run_init, 0, "clon", "time-invariant", None, "CLON"), tmp_dir), "tlon")
    clon = np.where(clon > 180, clon - 360, clon)
    idx = np.where((clat >= GRID_LATS[0] - _PAD_DEG) & (clat <= GRID_LATS[-1] + _PAD_DEG)
                   & (clon >= GRID_LONS[0] - _PAD_DEG) & (clon <= GRID_LONS[-1] + _PAD_DEG))[0]
    tri = Delaunay(np.column_stack([clon[idx], clat[idx]]))
    _grid_cache.clear()
    _grid_cache[run_init] = (idx, tri)
    return idx, tri


def fetch_icon(target_valid_time: pd.Timestamp,
               surface_elev_m: float = 100.0) -> xr.Dataset:
    """Fetch ICON global fields and return the SCP_Agreement Dataset schema."""
    target = pd.Timestamp(target_valid_time)

    candidates = list(candidate_icon_runs(target))
    if not candidates:
        raise RuntimeError(f"no ICON run online covers {target} "
                           f"(00/12Z reach F180, DWD keeps ~24 h)")
    chosen = None
    for run_init, fxx in candidates:
        d, kind, lev, var, _ = FIELDS["cape"]
        if _exists(_url(run_init, fxx, d, kind, lev, var)):
            chosen = (run_init, fxx)
            break
    if chosen is None:
        tried = ", ".join(f"{r:%d/%HZ}" for r, _ in candidates)
        raise RuntimeError(f"no ICON run had CAPE_ML for {target} (tried {tried})")
    run_init, fxx = chosen
    print(f"[ICON] run={run_init} F{fxx:03d} -> valid {target}")

    gx, gy = np.meshgrid(GRID_LONS, GRID_LATS)
    with tempfile.TemporaryDirectory(prefix="icon_") as td:
        tmp_dir = Path(td)
        idx, tri = _grid_for_run(run_init, tmp_dir)
        out = {}
        for key, (d, kind, lev, var, name) in FIELDS.items():
            path = _download_grib(_url(run_init, fxx, d, kind, lev, var), tmp_dir)
            vals = _open_values(path, name, lev)[idx]
            out[key] = LinearNDInterpolator(tri, vals)(gx, gy)
            path.unlink()
    print(f"[ICON] interpolated {len(idx)} cells -> {gy.shape} grid")

    cape_arr = out["cape"]
    u10_arr, v10_arr = out["u10"], out["v10"]
    u_arr = np.stack([out[f"u{p}"] for p in PRESSURE_LEVELS])
    v_arr = np.stack([out[f"v{p}"] for p in PRESSURE_LEVELS])
    gh_arr = np.stack([out[f"z{p}"] for p in PRESSURE_LEVELS]) / G0

    n_levels, n_lat, n_lon = u_arr.shape

    # Augmented profile: 10 m winds at h = surface_elev + 10 (as ECMWF)
    u_aug = np.empty((n_levels + 1, n_lat, n_lon), dtype=float)
    v_aug = np.empty_like(u_aug)
    gh_aug = np.empty_like(u_aug)
    u_aug[0] = u10_arr
    v_aug[0] = v10_arr
    gh_aug[0] = surface_elev_m + 10.0
    u_aug[1:] = u_arr
    v_aug[1:] = v_arr
    gh_aug[1:] = gh_arr
    pres_aug = np.concatenate([[1013.0], np.asarray(PRESSURE_LEVELS, dtype=float)])

    print(f"[ICON] deriving SRH/shear over {n_lat}x{n_lon} grid...")
    srh_03, shear_06 = scp_math.grid_derive_srh_and_shear(
        u_aug, v_aug, gh_aug, pres_aug, surface_elev_m=surface_elev_m
    )

    ds_out = xr.Dataset(
        data_vars={
            "mlcape":   (("latitude", "longitude"), cape_arr),
            "mlcin":    (("latitude", "longitude"),
                         np.full((n_lat, n_lon), np.nan, dtype=float)),
            "srh_03":   (("latitude", "longitude"), srh_03),
            "shear_06": (("latitude", "longitude"), shear_06),
        },
        coords={"latitude": GRID_LATS, "longitude": GRID_LONS},
    )
    ds_out.attrs.update({
        "model": "icon",
        "run_init": str(run_init),
        "forecast_hour": int(fxx),
        "valid_time": str(target),
        "notes": "CAPE=mixed-layer (CAPE_ML). SRH/shear derived from "
                 "pressure-level winds + geopotential. CIN unavailable. "
                 "Icosahedral grid linearly interpolated to 0.25 deg.",
    })
    print(f"[ICON] Successfully built dataset")
    return ds_out
