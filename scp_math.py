"""
SCP_Agreement: core math module.

Computes fixed-layer Supercell Composite Parameter from CAPE/SRH/shear
plus CIN and LFC factor corrections, and provides helpers for deriving
SRH/shear from raw pressure-level winds (needed for ECMWF, which doesn't
output those fields natively).

References:
    Thompson, R. L., R. Edwards, J. A. Hart, K. L. Elmore, and P. Markowski,
        2003: Close proximity soundings within supercell environments.
        Wea. Forecasting, 18, 1243-1261.
    Bunkers, M. J., et al., 2000: Predicting supercell motion using a new
        hodograph technique. Wea. Forecasting, 15, 61-79.
    Davies-Jones, R., et al., 1990: Streamwise vorticity: the origin of
        updraft rotation in supercell storms. 16th Conf. Severe Local Storms.
"""

import numpy as np


# ============================================================================
# Core SCP formula
# ============================================================================

def compute_scp_fixed(cape, srh_03, shear_06):
    """
    Fixed-layer Supercell Composite Parameter.

        SCP = (CAPE / 1000) * (SRH_03 / 50) * shear_term

    where shear_term scales shear/20 linearly between 10 and 20 m/s, is
    set to 0 below 10 m/s, and is capped at 1 above 20 m/s.

    Parameters
    ----------
    cape : array-like
        CAPE in J/kg (MLCAPE preferred).
    srh_03 : array-like
        0-3km storm-relative helicity in m^2/s^2.
    shear_06 : array-like
        0-6km bulk wind shear magnitude in m/s.

    Returns
    -------
    ndarray
        SCP (dimensionless), clipped at >= 0.
    """
    cape = np.asarray(cape, dtype=float)
    srh = np.asarray(srh_03, dtype=float)
    shear = np.asarray(shear_06, dtype=float)

    shear_term = np.clip(shear / 20.0, 0.0, 1.0)
    shear_term = np.where(shear < 10.0, 0.0, shear_term)

    scp = (cape / 1000.0) * (srh / 50.0) * shear_term
    return np.clip(scp, 0.0, None)


# ============================================================================
# CIN and LFC factor corrections (sigmoid penalties, never fully zeroed)
# ============================================================================

def cin_factor(cin):
    """
    Sigmoid penalty for convective inhibition (capping).

    CIN is conventionally negative; more negative = stronger cap.

    Behavior:
        CIN >=   0 J/kg  -> ~1.00 (no penalty)
        CIN =  -50 J/kg  -> 0.86
        CIN =  -75 J/kg  -> 0.65  (inflection)
        CIN = -100 J/kg  -> 0.45
        CIN = -200 J/kg  -> ~0.30 (floor)
    """
    cin = np.asarray(cin, dtype=float)
    return 1.0 - 0.7 / (1.0 + np.exp((cin + 75.0) / 25.0))


def lfc_factor(lfc_agl):
    """
    Sigmoid penalty for high Level of Free Convection.

    Higher LFC = harder initiation and harder maintenance.

    Behavior:
        LFC = 1000 m   -> 0.95
        LFC = 1500 m   -> 0.89
        LFC = 2000 m   -> 0.75
        LFC = 2500 m   -> 0.65  (inflection)
        LFC = 3000 m   -> 0.45
        LFC = 3500+ m  -> ~0.30 (floor)
    """
    lfc = np.asarray(lfc_agl, dtype=float)
    return 1.0 - 0.7 / (1.0 + np.exp(-(lfc - 2500.0) / 500.0))


# ============================================================================
# Storm motion and SRH (used by ECMWF fetcher to derive from raw winds)
# ============================================================================

def bunkers_right_mover(u_levels, v_levels, height_levels):
    """
    Bunkers et al. (2000) right-mover storm motion at a single point.

        c = mean wind (0-6 km) + 7.5 m/s perpendicular-right to 0-6 km shear

    Parameters
    ----------
    u_levels, v_levels : 1D arrays
        Wind components at each level (m/s).
    height_levels : 1D array
        Heights AGL (m), monotonically increasing.

    Returns
    -------
    c_u, c_v : float
        Storm motion components (m/s). NaN if no levels in 0-6 km.
    """
    u = np.asarray(u_levels, dtype=float)
    v = np.asarray(v_levels, dtype=float)
    h = np.asarray(height_levels, dtype=float)

    mask = (h >= 0) & (h <= 6000)
    if not np.any(mask):
        return np.nan, np.nan

    u_mean = np.mean(u[mask])
    v_mean = np.mean(v[mask])

    # Shear vector: closest level to 6 km minus closest level to surface
    idx_low = int(np.argmin(np.abs(h)))
    idx_high = int(np.argmin(np.abs(h - 6000)))

    du = u[idx_high] - u[idx_low]
    dv = v[idx_high] - v[idx_low]
    shear_mag = np.sqrt(du * du + dv * dv)

    if shear_mag < 0.1:
        return float(u_mean), float(v_mean)

    # Right-mover deviation: rotate shear vector 90 deg clockwise
    perp_u = dv / shear_mag
    perp_v = -du / shear_mag

    c_u = u_mean + 7.5 * perp_u
    c_v = v_mean + 7.5 * perp_v
    return float(c_u), float(c_v)


def compute_srh(u_levels, v_levels, height_levels, c_u, c_v, top_m=3000):
    """
    Storm-relative helicity from 0 to top_m at a single point.

    Discrete form (Markowski & Richardson 2010, eq. 2.91; positive SRH
    corresponds to cyclonic veering shear in the SPC convention):
        SRH = sum_k [(v_k - c_v)(u_{k+1} - u_k)
                   - (u_k - c_u)(v_{k+1} - v_k)]

    Returns
    -------
    float
        SRH in m^2/s^2.
    """
    u = np.asarray(u_levels, dtype=float)
    v = np.asarray(v_levels, dtype=float)
    h = np.asarray(height_levels, dtype=float)

    mask = (h >= 0) & (h <= top_m + 200)
    u = u[mask]
    v = v[mask]

    srh = 0.0
    for i in range(len(u) - 1):
        srh += (v[i] - c_v) * (u[i + 1] - u[i]) - (u[i] - c_u) * (v[i + 1] - v[i])
    return float(srh)


def compute_bulk_shear(u_low, v_low, u_high, v_high):
    """
    Magnitude of bulk wind difference between two levels (m/s).
    """
    du = np.asarray(u_high) - np.asarray(u_low)
    dv = np.asarray(v_high) - np.asarray(v_low)
    return np.sqrt(du * du + dv * dv)


# ============================================================================
# Region subsetting (cheap pre-filter before SRH derivation)
# ============================================================================

def subset_to_region(lat, lon, lat_range, lon_range, arrays):
    """
    Crop arrays to a lat/lon bounding box. Massive speedup for the pure-Python
    grid_derive_srh_and_shear() loop when only a regional subset is needed.

    Handles both lat orderings (ascending or descending) and both lon
    conventions ([-180, 180] or [0, 360]) by normalizing for comparison
    but returning the original lon values intact.

    Parameters
    ----------
    lat, lon : 1D arrays
        Latitude and longitude coordinates.
    lat_range : (min_lat, max_lat) in degrees, e.g. (20, 55)
    lon_range : (min_lon, max_lon) in degrees, [-180, 180] convention
    arrays : dict[str, ndarray]
        Arrays to subset. 2D arrays are treated as (lat, lon); 3D arrays
        are treated as (level, lat, lon).

    Returns
    -------
    sub_lat, sub_lon : 1D arrays
    sub_arrays : dict[str, ndarray]
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    # Normalize lon to [-180, 180] for comparison only
    lon_norm = np.where(lon > 180, lon - 360, lon)

    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
    lon_mask = (lon_norm >= lon_range[0]) & (lon_norm <= lon_range[1])

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f"No grid points found in region "
            f"lat={lat_range}, lon={lon_range}"
        )

    # Contiguous slices (works for any sort order as long as the region is
    # contiguous in index space, which it is for CONUS in any convention).
    lat_slc = slice(lat_idx[0], lat_idx[-1] + 1)
    lon_slc = slice(lon_idx[0], lon_idx[-1] + 1)

    sub_arrays = {}
    for name, arr in arrays.items():
        if arr.ndim == 2:
            sub_arrays[name] = arr[lat_slc, lon_slc]
        elif arr.ndim == 3:
            sub_arrays[name] = arr[:, lat_slc, lon_slc]
        else:
            raise ValueError(f"{name}: unsupported ndim {arr.ndim}")

    return lat[lat_slc], lon[lon_slc], sub_arrays


# ============================================================================
# Grid-wise SRH/shear from pressure-level winds
# ============================================================================

def grid_derive_srh_and_shear(u_pl, v_pl, gh_pl, pressure_levels,
                              surface_elev_m=100.0):
    """
    Derive 0-3km SRH and 0-6km bulk shear at every grid point from raw
    pressure-level winds. Used for ECMWF which does not output these fields.

    Loops over grid points and calls scalar functions above. Slower than
    pure-numpy vectorization but clearer and correct.

    Parameters
    ----------
    u_pl, v_pl : 3D arrays of shape (n_levels, n_lat, n_lon)
        Wind components on pressure levels (m/s).
    gh_pl : 3D array of same shape
        Geopotential height on pressure levels (m, above sea level).
    pressure_levels : 1D array
        Pressure levels in mb, matching the first axis. Used only for sorting
        so the level order is bottom-to-top.
    surface_elev_m : float
        Approximate surface elevation (m) to subtract from gh to get AGL.
        Default 100m is a crude CONUS-plains approximation; refine later
        for terrain accuracy.

    Returns
    -------
    srh_03 : 2D array (n_lat, n_lon)
    shear_06 : 2D array (n_lat, n_lon)
    """
    # Sort levels bottom-up (high pressure first)
    order = np.argsort(-np.asarray(pressure_levels))
    u_pl = u_pl[order]
    v_pl = v_pl[order]
    gh_pl = gh_pl[order]

    n_lev, n_lat, n_lon = u_pl.shape
    srh_03 = np.zeros((n_lat, n_lon), dtype=float)
    shear_06 = np.zeros((n_lat, n_lon), dtype=float)

    for j in range(n_lat):
        for i in range(n_lon):
            u = u_pl[:, j, i]
            v = v_pl[:, j, i]
            h_agl = gh_pl[:, j, i] - surface_elev_m

            c_u, c_v = bunkers_right_mover(u, v, h_agl)
            if np.isnan(c_u):
                srh_03[j, i] = 0.0
                shear_06[j, i] = 0.0
                continue

            srh_03[j, i] = compute_srh(u, v, h_agl, c_u, c_v, top_m=3000)

            # 0-6km bulk shear: top level near 6km minus bottom level
            idx_low = int(np.argmin(np.abs(h_agl)))
            idx_high = int(np.argmin(np.abs(h_agl - 6000)))
            shear_06[j, i] = compute_bulk_shear(
                u[idx_low], v[idx_low], u[idx_high], v[idx_high]
            )

    return srh_03, shear_06
