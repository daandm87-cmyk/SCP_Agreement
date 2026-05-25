"""
Rendering module (v2).

Two entry points:
    render_day_map() -- render one day's PNG to docs/<filename>
    build_index_html() -- build docs/index.html with a day slider that
                          shows each rendered day's PNG

Design:
    - Same Lambert Conformal projection / colormap as v1.
    - mSCP cells hidden where no model agrees OR mSCP < lowest bound
      (single pcolormesh on a masked array - automatic transparency).
    - HTML uses vanilla JS, no dependencies. Slider reflects forecast day,
      not date - day labels include the valid date for context.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


OUT_DIR = Path("docs")

SCP_BOUNDS = [0.5, 1, 2, 4, 6, 8, 10, 14, 18, 24, 30]
SCP_COLORS = [
    "#cfd8e8", "#9ab8d8", "#5e8fcf", "#2b6cb8",
    "#1c8e3f", "#85c43d", "#f4d300", "#f08a00",
    "#e23a1d", "#8e1f1f", "#4a0f0f",
]


def alpha_from_agreement(agreement_count, n_models_total):
    """
    Map agreement count -> alpha. With 2 models, agreement=2 → fully opaque.
    With 3+ models, gradation kicks in.
    """
    a = np.zeros_like(agreement_count, dtype=float)
    a = np.where(agreement_count >= 2, 0.25, a)
    a = np.where(agreement_count >= 3, 0.50, a)
    a = np.where(agreement_count >= 4, 0.75, a)
    a = np.where(agreement_count >= 5, 1.00, a)
    if n_models_total <= 2:
        a = np.where(agreement_count >= n_models_total, 1.0, a)
    return a


def render_day_map(mscp, agreement_count, n_models_total, lats, lons,
                   valid_time_str, title_extra="", png_filename="map.png"):
    """Render one day's mSCP map to OUT_DIR/<png_filename>."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha = alpha_from_agreement(agreement_count, n_models_total)

    proj = ccrs.LambertConformal(central_longitude=-97.5, central_latitude=38.5,
                                 standard_parallels=(33, 45))
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent([-122, -72, 24, 50], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e9e9e9", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#c8d6e5", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#888", linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#444", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#444", linewidth=0.6, zorder=2)

    cmap = ListedColormap(SCP_COLORS)
    cmap.set_under("none")
    norm = BoundaryNorm(SCP_BOUNDS, ncolors=len(SCP_COLORS), extend="max")

    if lons.ndim == 1 and lats.ndim == 1:
        lon2d, lat2d = np.meshgrid(lons, lats)
    else:
        lon2d, lat2d = lons, lats

    mask = (alpha == 0) | (mscp < SCP_BOUNDS[0])
    mscp_masked = np.ma.masked_array(mscp, mask=mask)

    ax.pcolormesh(
        lon2d, lat2d, mscp_masked,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(),
        shading="auto", rasterized=True, zorder=3,
    )

    title = f"Multi-Model SCP Agreement  |  valid {valid_time_str}"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=12, loc="left")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal",
                      pad=0.04, aspect=40, shrink=0.85, extend="max")
    cb.set_label("mSCP (mean SCP × CIN factor)")

    fig.text(0.5, 0.02,
             f"Opacity = model agreement count.  "
             f"0-1: invisible, 2: 25%, 3: 50%, 4: 75%, 5+: 100% "
             f"(with 2 models: binary visible/invisible).",
             ha="center", fontsize=9, color="#444")

    out_png = OUT_DIR / png_filename
    plt.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[render] wrote {out_png}")


def build_index_html(day_results):
    """
    Build docs/index.html with a day slider.

    day_results: list of dicts from main.py, one per successfully rendered day:
        {"day_num": int, "valid_time": str, "png_filename": str,
         "models": [str], "n_models": int}
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON-serialize the day metadata so the JS can drive the slider
    days_json = json.dumps([
        {
            "day": d["day_num"],
            "valid": d["valid_time"],
            "png": d["png_filename"],
            "models": d["models"],
        }
        for d in day_results
    ])

    n_days = len(day_results)
    first_day = day_results[0]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SCP_Agreement</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                   Roboto, Helvetica, Arial, sans-serif;
      margin: 24px; background: #f9f9f9; color: #222;
      max-width: 1100px; margin-left: auto; margin-right: auto;
    }}
    h1 {{ font-size: 1.4rem; font-weight: 600; margin-bottom: 4px; }}
    .sub {{ color: #666; font-size: 0.9rem; margin-bottom: 18px; }}
    .controls {{
      background: white; border-radius: 6px; padding: 14px 18px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      margin-bottom: 14px;
    }}
    .day-label {{
      display: flex; justify-content: space-between; align-items: baseline;
      margin-bottom: 8px;
    }}
    .day-label .big {{ font-size: 1.1rem; font-weight: 600; }}
    .day-label .small {{ color: #666; font-size: 0.9rem; }}
    input[type=range] {{
      width: 100%; margin: 4px 0 8px 0;
    }}
    .ticks {{
      display: flex; justify-content: space-between;
      color: #888; font-size: 0.75rem;
    }}
    img {{
      width: 100%; height: auto; border-radius: 4px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1); background: white;
      display: block;
    }}
    .footer {{ color: #888; font-size: 0.8rem; margin-top: 16px; text-align: center; }}
  </style>
</head>
<body>
  <h1>SCP_Agreement</h1>
  <p class="sub">Multi-model Supercell Composite Parameter agreement &mdash; 10-day forecast</p>

  <div class="controls">
    <div class="day-label">
      <span class="big" id="dayLabel">Day {first_day['day_num']}</span>
      <span class="small" id="dayMeta">valid {first_day['valid_time']} &middot; {', '.join(first_day['models'])}</span>
    </div>
    <input type="range" id="daySlider" min="1" max="{n_days}" value="1" step="1">
    <div class="ticks">
      <span>Day 1</span>
      <span>Day {n_days}</span>
    </div>
  </div>

  <img id="dayImage" src="{first_day['png_filename']}" alt="SCP Agreement map">

  <p class="footer">v2: GFS + ECMWF + CMC GDPS. Opacity reflects model agreement.</p>

  <script>
    const days = {days_json};
    const slider = document.getElementById('daySlider');
    const img = document.getElementById('dayImage');
    const label = document.getElementById('dayLabel');
    const meta = document.getElementById('dayMeta');

    function update(idx) {{
      const d = days[idx];
      label.textContent = 'Day ' + d.day;
      meta.textContent = 'valid ' + d.valid + ' \\u00b7 ' + d.models.join(', ');
      img.src = d.png;
    }}

    slider.addEventListener('input', e => {{
      update(parseInt(e.target.value) - 1);
    }});
  </script>
</body>
</html>
"""
    out_html = OUT_DIR / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"[render] wrote {out_html}")
