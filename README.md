# SCP_Agreement

Multi-model Supercell Composite Parameter agreement map.

For a target valid time, this product:

1. Pulls severe weather environment fields from **GFS** (native) and
   **ECMWF Open Data** (with SRH and 0-6 km bulk shear derived from raw
   pressure-level winds, since ECMWF doesn't output them).
2. Computes a **fixed-layer SCP** per model:
   `SCP = (CAPE/1000) * (SRH_03/50) * (Shear_06/20)`
3. Takes the multi-model mean SCP and applies a **CIN factor correction**
   (sigmoid penalty for cap) using GFS CIN.
4. Computes a **model-agreement count** at each grid point.
5. Renders a CONUS map: color shading for mSCP magnitude, **opacity driven
   by model agreement** (more models agreeing -> more opaque).

Output is pushed to `docs/index.html` for GitHub Pages.

## v1 scope and known limitations

- **Models:** GFS deterministic + ECMWF deterministic only.
- **Single valid-time snapshot.** No accumulation across forecast window.
- **CIN correction only.** LFC factor is designed in `scp_math.py` but
  not used in v1 since LFC isn't in standard GFS GRIB output. Defer to v2.
- **ECMWF SRH/shear approximations:** derived from coarse pressure-level
  winds (1000, 925, 850, 700, 500 mb) plus 10 m wind. Surface height is
  approximated as 100 m AGL (CONUS-Plains average); terrain effects in the
  Rockies / Appalachians will be inaccurate.
- **Agreement layer with only 2 models is binary** (visible if both agree,
  invisible otherwise). The transparency framework is general; once more
  models are added it shows real gradation.

## File structure

```
SCP_Agreement/
├── .github/
│   └── workflows/
│       └── render.yml          # GitHub Actions workflow
├── docs/                       # output dir (generated)
│   ├── index.html
│   └── map.png
├── main.py                     # pipeline orchestrator
├── scp_math.py                 # SCP formula + CIN/LFC factors + SRH derivation
├── gfs_fetcher.py              # Herbie-based GFS data pull
├── ecmwf_fetcher.py            # ECMWF Open Data pull + SRH/shear derivation
├── render.py                   # matplotlib/cartopy map + HTML wrapper
├── environment.yml             # conda env spec
└── README.md
```

## Setup

### 1. Create the repo on GitHub

- **+** → **New repository**
- Name: `SCP_Agreement`
- Visibility: **Public** (gets you unlimited free Actions minutes)
- Initialize with README: **yes**
- Click **Create repository**

### 2. Clone it locally

```bash
git clone https://github.com/YOUR_USERNAME/SCP_Agreement.git
cd SCP_Agreement
```

### 3. Drop the files in

Copy these into the repo root:
- `main.py`
- `scp_math.py`
- `gfs_fetcher.py`
- `ecmwf_fetcher.py`
- `render.py`
- `environment.yml`
- `README.md`

And create the workflow folder:
```bash
mkdir -p .github/workflows
```
Move `render.yml` into `.github/workflows/`.

### 4. Commit and push

```bash
git add .
git commit -m "Initial commit"
git push
```

### 5. Enable GitHub Pages

On your repo's GitHub page:
- **Settings** → **Pages**
- **Source**: Deploy from a branch
- **Branch**: `main`, folder: `/docs`
- **Save**

GitHub gives you a URL like `https://YOUR_USERNAME.github.io/SCP_Agreement/`.
It'll 404 until the first workflow run creates `docs/index.html`.

### 6. Run the workflow

- **Actions** tab on your repo
- **Generate SCP Agreement map** in the left sidebar
- **Run workflow**
- Leave `valid_time` blank to use the default (next Monday 00Z), or type
  e.g. `2026-05-25T00:00:00` to target a specific time.
- Click **Run workflow**

Takes ~10-15 minutes (mostly conda env setup + GRIB downloads). When it
finishes the workflow has committed `docs/index.html` and `docs/map.png`
back to the repo; GitHub Pages serves them within ~1 minute.

## Test target

For the first test we're aiming at:

**Monday 2026-05-25 00:00 UTC** (Sunday afternoon/evening peak in central US)

Iowa SCP signal expected. If the map shows a meaningful blob there with
visible agreement (both GFS and ECMWF lighting up), the pipeline works
end-to-end.

## Math reference

See module docstrings:

- `scp_math.compute_scp_fixed` — fixed-layer SCP formula
- `scp_math.cin_factor` — sigmoid CIN penalty (floor at ~0.30, inflection
  at -75 J/kg)
- `scp_math.lfc_factor` — sigmoid LFC penalty (floor at ~0.30, inflection
  at 2500 m AGL); designed but unused in v1
- `scp_math.bunkers_right_mover` — Bunkers et al. (2000) right-mover motion
- `scp_math.compute_srh` — Davies-Jones et al. (1990) discrete SRH

Primary references:
- Thompson, R. L., et al. (2003): Close proximity soundings within supercell
  environments. Wea. Forecasting, 18, 1243-1261.
- Bunkers, M. J., et al. (2000): Predicting supercell motion using a new
  hodograph technique. Wea. Forecasting, 15, 61-79.
