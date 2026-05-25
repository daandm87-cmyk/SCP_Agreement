"""
ECMWF Open Data fetcher.

ECMWF Open Data does not output SRH, bulk shear, CIN, or LFC natively.
We pull what IS available:
    - CAPE (entire atmosphere)
    - 10m u/v winds
    - u/v/geopotential height on pressure levels (1000, 925, 850, 700, 500 mb)

Then we DERIVE 0-3 km SRH and 0-6 km bulk shear at every grid point using
scp_math.grid_derive_srh_and_shear().

CIN is not available; this fetcher returns CIN as NaN. The consensus
CIN correction applied to the multi-model mean comes from GFS only.

Uses the same cfgrib.open_datasets() + iterate pattern as the GFS fetcher
to handle the fact that cfgrib splits multi-typeOfLevel GRIBs into a list
of Datasets that can't be merged.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
from ecmwf.opendata import Client

import scp_math


ECMWF_AVAILABILITY_LAG_HOURS = 7
PRESSURE_LEVELS = [1000, 925, 850, 700, 500]


def find_latest_ecmwf_run(target_valid_time: pd.Timestamp):
    """Most recent ECMWF HRES cycle that can forecast the target valid time."""
    now = pd.Timestamp.now('UTC').tz_localize(None)
    target = target_valid_time.tz_localize(None) if target_valid_time.tzinfo else target_valid_time

    for hours_back in range(0, 72, 12):
        candidate_day = (now - pd.Timedelta(hours=hours_back)).floor("12h")
        for run_hour in (12, 0):
            candidate = candidate_day.replace(hour=run_hour, minute=0,
                                              second=0, microsecond=0)
            if candidate > now:
                continue
            age_hours = (now - candidate).total_seconds() / 3600.0
            if age_hours < ECMWF_AVAILABILITY_LAG_HOURS:
                continue
            step = int((target - candidate).total_seconds() / 3600.0)
            if 0 <= step <= 240 and step % 3 == 0:
                return candidate, step

    raise RuntimeError(
        f"No usable ECMWF run found for target valid time {target_valid_time}"
    )


def _open_grib_as_list(grib_path):
    """Open a GRIB with cfgrib, always return a list of Datasets."""
    result = cfgrib.open_datasets(
        str(grib_path),
        backend_kwargs={"indexpath": ""},
    )
    if not isinstance(result, list):
        result = [result]
    return result


def _find_var(datasets, name_options):
    """Find a variable in a list of Datasets by short name."""
    name_set = {n.lower() for n in name_options}
    for ds in datasets:
        for var_name in ds.data_vars:
            if var_name.lower() in name_set:
                return ds[var_name]
    return None


def fetch_ecmwf(target_valid_time: pd.Timestamp,
                surface_elev_m: float = 100.0) -> xr.Dataset:
    """
    Fetch ECMWF fields and derive SRH/shear.

    Returns xr.Dataset with same schema as GFS fetcher:
        mlcape   (J/kg)     -- ECMWF MUCAPE (more correct for SCP than MLCAPE)
        mlcin    (J/kg)     -- all NaN; ECMWF Open Data has no CIN
        srh_03   (m^2/s^2)  -- derived from pressure-level winds
        shear_06 (m/s)      -- derived from 500mb and 10m winds
    """
    target = pd.Timestamp(target_valid_time)
    run_init, step = find_latest_ecmwf_run(target)
    print(f"[ECMWF] run={run_init} step={step}h -> valid {target}")

    client = Client(source="ecmwf")
    date_str = run_init.strftime("%Y%m%d")
    time_int = run_init.hour

    with tempfile.TemporaryDirectory(prefix="ecmwf_") as tmpdir:
        tmp = Path(tmpdir)

        # Download surface fields (MUCAPE + 10m winds)
        # ECMWF Open Data names this 'mucape', not 'cape'.
        sfc_path = tmp / "sfc.grib2"
        client.retrieve(
            target=str(sfc_path),
            date=date_str, time=time_int, step=step,
            type="fc", levtype="sfc",
            param=["mucape", "10u", "10v"],
        )

        # Download pressure-level u, v, gh
        pl_path = tmp / "pl.grib2"
        client.retrieve(
            target=str(pl_path),
            date=date_str, time=time_int, step=step,
            type="fc", levtype="pl",
            levelist=PRESSURE_LEVELS,
            param=["u", "v", "gh"],
        )

        # Parse both files into lists of Datasets
        sfc_datasets = _open_grib_as_list(sfc_path)
        pl_datasets = _open_grib_as_list(pl_path)

        print(f"[ECMWF] surface: {len(sfc_datasets)} hypercubes, "
              f"pressure-level: {len(pl_datasets)} hypercubes")

        # Find surface fields (mucape comes through as 'mucape' from cfgrib)
        cape = _find_var(sfc_datasets, ["mucape", "cape"])
        u10 = _find_var(sfc_datasets, ["u10", "10u"])
        v10 = _find_var(sfc_datasets, ["v10", "10v"])

        if cape is None or u10 is None or v10 is None:
            print("[ECMWF DEBUG] Surface file contents:")
            for i, ds in enumerate(sfc_datasets):
                for vn in ds.data_vars:
                    tol = ds[vn].attrs.get("GRIB_typeOfLevel", "?")
                    print(f"[ECMWF DEBUG]   [{i}] {vn} | typeOfLevel={tol}")
            raise RuntimeError(
                f"Missing ECMWF surface fields. "
                f"cape={cape is not None}, u10={u10 is not None}, "
                f"v10={v10 is not None}"
            )

        # Find pressure-level fields
        u_pl = _find_var(pl_datasets, ["u"])
        v_pl = _find_var(pl_datasets, ["v"])
        gh_pl = _find_var(pl_datasets, ["gh"])

        if u_pl is None or v_pl is None or gh_pl is None:
            print("[ECMWF DEBUG] Pressure-level file contents:")
            for i, ds in enumerate(pl_datasets):
                for vn in ds.data_vars:
                    print(f"[ECMWF DEBUG]   [{i}] {vn}")
            raise RuntimeError(
                f"Missing ECMWF pressure-level fields. "
                f"u={u_pl is not None}, v={v_pl is not None}, "
                f"gh={gh_pl is not None}"
            )

        # Materialize while files still open
        cape_arr = cape.values
        u10_arr = u10.values
        v10_arr = v10.values
        u_arr = u_pl.values
        v_arr = v_pl.values
        gh_arr = gh_pl.values
        lat = u_pl["latitude"].values
        lon = u_pl["longitude"].values
        pres = u_pl["isobaricInhPa"].values

    # Sort pressure levels descending (surface up)
    order = np.argsort(-pres)
    pres_sorted = pres[order]
    u_arr = u_arr[order]
    v_arr = v_arr[order]
    gh_arr = gh_arr[order]

    # --- SUBSET to CONUS region BEFORE the expensive SRH derivation ---
    # The full ECMWF grid is global (~1M points); CONUS is ~4% of that.
    # 25x speedup on the pure-Python loop in grid_derive_srh_and_shear.
    CONUS_LAT = (20.0, 55.0)
    CONUS_LON = (-130.0, -60.0)
    lat, lon, subbed = scp_math.subset_to_region(
        lat, lon, CONUS_LAT, CONUS_LON,
        {
            "u_pl":  u_arr,
            "v_pl":  v_arr,
            "gh_pl": gh_arr,
            "cape":  cape_arr,
            "u10":   u10_arr,
            "v10":   v10_arr,
        },
    )
    u_arr = subbed["u_pl"]
    v_arr = subbed["v_pl"]
    gh_arr = subbed["gh_pl"]
    cape_arr = subbed["cape"]
    u10_arr = subbed["u10"]
    v10_arr = subbed["v10"]
    print(f"[ECMWF] subset to CONUS: {u_arr.shape}")

    n_levels, n_lat, n_lon = u_arr.shape

    # Build augmented profile: prepend 10m winds at h = surface_elev + 10
    u_aug = np.empty((n_levels + 1, n_lat, n_lon), dtype=float)
    v_aug = np.empty_like(u_aug)
    gh_aug = np.empty_like(u_aug)
    u_aug[0] = u10_arr
    v_aug[0] = v10_arr
    gh_aug[0] = surface_elev_m + 10.0
    u_aug[1:] = u_arr
    v_aug[1:] = v_arr
    gh_aug[1:] = gh_arr
    pres_aug = np.concatenate([[1013.0], pres_sorted])

    print(f"[ECMWF] deriving SRH/shear over {n_lat}x{n_lon} grid...")
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
        coords={"latitude": lat, "longitude": lon},
    )
    ds_out.attrs.update({
        "model": "ecmwf",
        "run_init": str(run_init),
        "forecast_hour": int(step),
        "valid_time": str(target),
        "notes": "SRH/shear derived from pressure-level winds. CIN unavailable.",
    })
    print(f"[ECMWF] Successfully built dataset")
    return ds_out
