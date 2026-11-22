"""
Met Office (UKMO) Global deterministic 10 km fetcher.

Reads from the Met Office atmospheric model data on AWS Open Data (verified
against live listings on 2026-11-22):

    https://met-office-atmospheric-model-data.s3.amazonaws.com/
        global-deterministic-10km/{RUN:%Y%m%dT%H%MZ}/
        {VALID:%Y%m%dT%H%MZ}-PT{fxx:04d}H00M-{field}.nc

    NOTE the filename starts with the VALID time, not the run time, e.g.
    .../20261122T0000Z/20261123T0000Z-PT0024H00M-CAPE_mixed_layer_lowest_500m.nc
    Runs every 6 h; the 00Z run reaches PT0168H (Day 7), hourly to 54 h then
    3/6-hourly.

Fields (NetCDF4, one variable per file, global 2560 x 1920 lon/lat grid,
lat ascending, lon -180..180, chunked 128 x 128 per level):

    CAPE_mixed_layer_lowest_500m              mixed-layer CAPE (J/kg)
    CIN_mixed_layer_lowest_500m               mixed-layer CIN (J/kg, <= 0)
    wind_speed_at_10m, wind_direction_at_10m
    wind_speed_on_pressure_levels             37 levels, Pa
    wind_direction_on_pressure_levels         ("from" direction, degrees)
    geopotential_height_on_pressure_levels    (m)

The pressure-level files are 120-225 MB each, so instead of downloading
them we read only the CONUS box at 1000/925/850/700/500 hPa through HTTP
range requests (fsspec + h5py); about 2 s per file. u/v are rebuilt from
speed and meteorological direction. SRH/shear are derived exactly as in
ecmwf_fetcher (real geopotential height, 10 m wind at the bottom).

Data: Met Office, via AWS Open Data. Check the bucket's licence before
redistributing output beyond private use.
"""

import fsspec
import h5py
import numpy as np
import pandas as pd
import xarray as xr

import scp_math


UKMO_BASE = ("https://met-office-atmospheric-model-data.s3.amazonaws.com/"
             "global-deterministic-10km")
UKMO_AVAILABILITY_LAG_HOURS = 5    # the existence probe confirms availability
UKMO_MAX_FXX = 168

PRESSURE_LEVELS = [1000, 925, 850, 700, 500]
CONUS_LAT = (20.0, 55.0)
CONUS_LON = (-130.0, -60.0)

# key -> (file field name, NetCDF standard_name of the data variable)
FIELDS = {
    "cape": ("CAPE_mixed_layer_lowest_500m",
             "atmosphere_convective_available_potential_energy"),
    "cin":  ("CIN_mixed_layer_lowest_500m", "atmosphere_convective_inhibition"),
    "ws10": ("wind_speed_at_10m", "wind_speed"),
    "wd10": ("wind_direction_at_10m", "wind_from_direction"),
    "ws":   ("wind_speed_on_pressure_levels", "wind_speed"),
    "wd":   ("wind_direction_on_pressure_levels", "wind_from_direction"),
    "gh":   ("geopotential_height_on_pressure_levels", "geopotential_height"),
}

_fs = fsspec.filesystem("http")


def candidate_ukmo_runs(target_valid_time: pd.Timestamp):
    """Yield (run_init, fxx) newest-first, walking back up to 72 h."""
    now = pd.Timestamp.now('UTC').tz_localize(None)
    target = (target_valid_time.tz_localize(None)
              if target_valid_time.tzinfo else target_valid_time)
    newest = now.floor("6h")
    for k in range(0, 13):
        run_init = newest - pd.Timedelta(hours=6 * k)
        if (now - run_init).total_seconds() / 3600.0 < UKMO_AVAILABILITY_LAG_HOURS:
            continue
        fxx = int((target - run_init).total_seconds() / 3600.0)
        if 0 <= fxx <= UKMO_MAX_FXX:
            yield run_init, fxx


def _url(run_init, fxx, field):
    valid = run_init + pd.Timedelta(hours=fxx)
    return (f"{UKMO_BASE}/{run_init:%Y%m%dT%H%MZ}/"
            f"{valid:%Y%m%dT%H%MZ}-PT{fxx:04d}H00M-{field}.nc")


def _exists(url):
    try:
        return _fs.exists(url)
    except Exception:
        return False


def _data_var(h, standard_name, url):
    for name, ds in h.items():
        if (isinstance(ds, h5py.Dataset) and ds.ndim >= 2
                and ds.attrs.get("standard_name", b"").decode() == standard_name):
            return ds
    raise RuntimeError(f"UKMO {url.rsplit('/', 1)[-1]}: no variable with "
                       f"standard_name '{standard_name}'")


def _read_conus(url, standard_name):
    """Read the CONUS box (and the 5 needed levels, if 3D) via range requests."""
    try:
        f = _fs.open(url, "rb", block_size=2**20, cache_type="blockcache")
    except FileNotFoundError:
        raise RuntimeError(f"UKMO {url.rsplit('/', 1)[-1]}: not found") from None
    with f, h5py.File(f, "r") as h:
        lat = h["latitude"][:]
        lon = h["longitude"][:]
        li = np.where((lat >= CONUS_LAT[0]) & (lat <= CONUS_LAT[1]))[0]
        lo = np.where((lon >= CONUS_LON[0]) & (lon <= CONUS_LON[1]))[0]
        ys, xs = slice(li[0], li[-1] + 1), slice(lo[0], lo[-1] + 1)
        var = _data_var(h, standard_name, url)
        if var.ndim == 3:
            pres_hpa = h["pressure"][:] / 100.0
            idx = []
            for p in PRESSURE_LEVELS:
                hit = np.where(np.isclose(pres_hpa, p))[0]
                if len(hit) != 1:
                    raise RuntimeError(f"UKMO {url.rsplit('/', 1)[-1]}: "
                                       f"no {p} hPa level")
                idx.append(int(hit[0]))
            data = np.stack([var[i, ys, xs] for i in idx]).astype(float)
        else:
            data = var[ys, xs].astype(float)
        fill = var.fillvalue
        if fill is not None and np.isfinite(fill) and abs(fill) > 1e30:
            data[data >= 0.99 * fill] = np.nan
        return lat[ys].astype(float), lon[xs].astype(float), data


def _uv(speed, from_dir_deg):
    rad = np.deg2rad(from_dir_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def fetch_ukmo(target_valid_time: pd.Timestamp,
               surface_elev_m: float = 100.0) -> xr.Dataset:
    """Fetch UKMO Global fields and return the SCP_Agreement Dataset schema."""
    target = pd.Timestamp(target_valid_time)

    candidates = list(candidate_ukmo_runs(target))
    if not candidates:
        raise RuntimeError(f"no UKMO run within F000-F{UKMO_MAX_FXX} of {target}")
    chosen = None
    for run_init, fxx in candidates:
        if _exists(_url(run_init, fxx, FIELDS["gh"][0])):   # largest file, written last-ish
            chosen = (run_init, fxx)
            break
    if chosen is None:
        tried = ", ".join(f"{r:%d/%HZ}" for r, _ in candidates)
        raise RuntimeError(f"no UKMO run had data for {target} (tried {tried})")
    run_init, fxx = chosen
    print(f"[UKMO] run={run_init} F{fxx:03d} -> valid {target}")

    arrays = {}
    for key, (field, std_name) in FIELDS.items():
        lat, lon, arrays[key] = _read_conus(_url(run_init, fxx, field), std_name)

    u10_arr, v10_arr = _uv(arrays["ws10"], arrays["wd10"])
    u_arr, v_arr = _uv(arrays["ws"], arrays["wd"])
    gh_arr = arrays["gh"]
    cape_arr, cin_arr = arrays["cape"], arrays["cin"]
    print(f"[UKMO] CONUS slab: {u_arr.shape}  CAPE {np.nanmin(cape_arr):.0f}.."
          f"{np.nanmax(cape_arr):.0f}  CIN {np.nanmin(cin_arr):.0f}.."
          f"{np.nanmax(cin_arr):.0f} J/kg")

    n_levels, n_lat, n_lon = u_arr.shape
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

    print(f"[UKMO] deriving SRH/shear over {n_lat}x{n_lon} grid...")
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
        "model": "ukmo",
        "run_init": str(run_init),
        "forecast_hour": int(fxx),
        "valid_time": str(target),
        "notes": "CAPE/CIN = mixed layer (lowest 500 m). SRH/shear derived "
                 "from pressure-level winds + geopotential height.",
    })
    print(f"[UKMO] Successfully built dataset")
    return ds_out
