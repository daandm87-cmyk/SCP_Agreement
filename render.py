"""
Rendering module: plot the multi-model mSCP map with agreement transparency.

Output:
    docs/map.png        -- the PNG map
    docs/index.html     -- simple HTML wrapper for GitHub Pages

Design choices:
    - CONUS Lambert Conformal projection (matches the look of typical
      SPC/Gensini maps).
    - mSCP shaded with a discrete colormap.
    - Agreement count drives alpha (transparency): 0-1 models = invisible,
      2 models = 0.25, 3 = 0.50, 4 = 0.75, 5+ = 1.0.
      With only 2 models in v1, agreement is binary: 0/1 -> invisible,
      2 -> fully visible. (Add more models -> transparency layer matters.)
"""

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Color scale for mSCP. Discrete bins mirroring typical severe-weather
# composite displays.
SCP_BOUNDS = [0.5, 1, 2, 4, 6, 8, 10, 14, 18, 24, 30]
SCP_COLORS = [
    "#cfd8e8",  # 0.5-1   very pale
    "#9ab8d8",  # 1-2     light blue
    "#5e8fcf",  # 2-4
    "#2b6cb8",  # 4-6
    "#1c8e3f",  # 6-8     green
    "#85c43d",  # 8-10
    "#f4d300",  # 10-14   yellow
    "#f08a00",  # 14-18   orange
    "#e23a1d",  # 18-24   red
    "#8e1f1f",  # 24-30   deep red
    "#4a0f0f",  # 30+     extend-max color (off the chart)
]


def alpha_from_agreement(agreement_count, n_models_total):
    """
    Map per-grid model-agreement count to an alpha value.

    Rule:
        0 or 1 model agreeing    -> alpha 0
        2 models                 -> 0.25
        3                        -> 0.50
        4                        -> 0.75
        5+ (or all available)    -> 1.00
    """
    a = np.zeros_like(agreement_count, dtype=float)
    a = np.where(agreement_count >= 2, 0.25, a)
    a = np.where(agreement_count >= 3, 0.50, a)
    a = np.where(agreement_count >= 4, 0.75, a)
    a = np.where(agreement_count >= 5, 1.00, a)
    # For v1 with only 2 models, "agreement = 2" is the max and we want it
    # at full opacity rather than 0.25.
    if n_models_total <= 2:
        a = np.where(agreement_count >= n_models_total, 1.0, a)
    return a


def render_map(mscp, agreement_count, n_models_total,
               lats, lons,
               valid_time_str, title_extra="",
               out_dir="docs"):
    """
    Render mSCP map to PNG + write HTML wrapper.

    Parameters
    ----------
    mscp : 2D array (lat, lon)
        Corrected multi-model mean SCP.
    agreement_count : 2D array (lat, lon)
        Number of models with SCP_fixed above the agreement threshold.
    n_models_total : int
        Total number of models in the consensus (for transparency scaling).
    lats, lons : 1D arrays
    valid_time_str : str
        Human-readable valid time for the title.
    title_extra : str
        Optional second line of title (e.g., run sources).
    out_dir : str
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build agreement-driven alpha mask
    alpha = alpha_from_agreement(agreement_count, n_models_total)

    # Set up figure with Lambert Conformal projection
    proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5,
                                 standard_parallels=(33, 45))
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent([-122, -72, 24, 50], crs=ccrs.PlateCarree())

    # Background features
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e9e9e9", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#c8d6e5", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#888", linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#444", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#444", linewidth=0.6, zorder=2)

    # Color setup
    cmap = ListedColormap(SCP_COLORS)
    norm = BoundaryNorm(SCP_BOUNDS, ncolors=len(SCP_COLORS), extend="max")

    # Create a 2D mesh grid for lon/lat if needed
    if lons.ndim == 1 and lats.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    else:
        lon2d, lat2d = lons, lats

    # Plot mSCP shading with per-pixel alpha
    # Approach: pcolormesh with shading, then multiply alpha into the RGBA image.
    # Simplest robust approach: render mscp shading first into RGBA, then
    # combine alpha array and overlay.
    rgba = cmap(norm(mscp))           # (lat, lon, 4)
    rgba[..., 3] = alpha               # apply our alpha map

    ax.pcolormesh(
        lon2d, lat2d, mscp,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto",
        alpha=None,
        rasterized=True,
    )
    # Overlay the alpha-masked image on top so transparency is honored
    # (the pcolormesh above gives the discrete shading; the imshow-like
    # alpha overlay handles transparency from agreement)
    ax.imshow(
        rgba,
        origin="upper",
        extent=[lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max()],
        transform=ccrs.PlateCarree(),
        interpolation="nearest",
        zorder=3,
    )

    # Title
    title = f"Multi-Model SCP Agreement  |  valid {valid_time_str}"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=13, loc="left")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal",
                      pad=0.04, aspect=40, shrink=0.85, extend="max")
    cb.set_label("mSCP (mean SCP × CIN factor)")

    # Footer / agreement note
    fig.text(0.5, 0.02,
             f"Opacity = model agreement count "
             f"(0-1: invisible, 2+: visible). v1: GFS + ECMWF.",
             ha="center", fontsize=9, color="#444")

    png_path = out_path / "map.png"
    plt.savefig(png_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[render] wrote {png_path}")

    # Write a minimal HTML wrapper so GitHub Pages shows the map
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SCP_Agreement &mdash; {valid_time_str}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
            Roboto, Helvetica, Arial, sans-serif; margin: 24px;
            background: #f9f9f9; color: #222; }}
    h1 {{ font-size: 1.2rem; font-weight: 600; }}
    .sub {{ color: #666; font-size: 0.95rem; margin-bottom: 16px; }}
    img {{ max-width: 100%; height: auto; border-radius: 4px;
           box-shadow: 0 2px 6px rgba(0,0,0,0.1); background: white; }}
  </style>
</head>
<body>
  <h1>SCP_Agreement</h1>
  <p class="sub">Valid {valid_time_str} &middot; v1 (GFS + ECMWF)</p>
  <img src="map.png" alt="Multi-model SCP agreement map">
</body>
</html>
"""
    html_path = out_path / "index.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[render] wrote {html_path}")
