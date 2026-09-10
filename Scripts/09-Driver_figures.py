#%% 0. Imports ----------------------------------------------------------------
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

which_season = 'winter'

#%% 1. Configuration ----------------------------------------------------------

# ── Data paths (output from script 08) ──
data_paths = {
    'SAm': f'../Data/Output_data/DriverImpact_SAm_{which_season}.nc',
    'SAf': f'../Data/Output_data/DriverImpact_SAf_{which_season}.nc',
    'Oce': f'../Data/Output_data/DriverImpact_Oce_{which_season}.nc',
}

# ── Manually chosen LLB sector longitudes per region (must match script 08) ──
manual_sectors = {
    'SAm': [-85, -55, -25],
    'SAf': [15, 75],
    'Oce': [85, 115, 165, -155],
}

sector_labels_per_region = {
    'SAm': ['West', 'Over', 'East'],
    'SAf': ['West', 'East'],
    'Oce': ['West', 'Over West', 'Over East', 'East'],
}

# ── Oceania projection & extents ──
oce_central_lon_W = 125
oce_central_lon_E = 170

oce_extent_W = [50, 180, -75, 0]
oce_extent_E = [100, 230, -75, 0]

# ── Region display settings ──
region_cfg = {
    'SAm': {
        'full_name': 'South America', 'label': 'SAm',
        'zoom_extent': [-120, 10, -75, 0],
        'box_color': 'tab:red',
        'central_longitude': 0,
    },
    'SAf': {
        'full_name': 'South Africa', 'label': 'SAf',
        'zoom_extent': [-25, 105, -75, 0],
        'box_color': 'tab:blue',
        'central_longitude': 0,
    },
    'Oce': {
        'full_name': 'Oceania', 'label': 'Oce',
        'zoom_extent': None,
        'box_color': 'tab:green',
        'central_longitude': None,
    },
}

# ── IVT contourf settings (white gap from -10 to +10) ──
ivt_cmap_name = 'PuOr'
ivt_levels = np.array([-100, -80, -60, -40, -20, -10, 10, 20, 40, 60, 80, 100])
_cont_cmap = plt.get_cmap(ivt_cmap_name)
n_bins = len(ivt_levels) - 1
_sample_pts = np.linspace(0.08, 0.92, n_bins)
_colors = [_cont_cmap(s) for s in _sample_pts]
mid_idx = len(_colors) // 2
_colors[mid_idx] = (1.0, 1.0, 1.0, 1.0)
ivt_cmap = mcolors.ListedColormap(_colors)
ivt_cmap.set_under(_cont_cmap(0.0))
ivt_cmap.set_over(_cont_cmap(1.0))
ivt_norm = mcolors.BoundaryNorm(ivt_levels, ivt_cmap.N)
ivt_contour_levels = ivt_levels

# ── Z500 contour settings ──
z500_threshold = 40
z500_levels_pos = [z500_threshold]
z500_levels_neg = [-z500_threshold]
z500_color_pos = 'tab:red'
z500_color_neg = 'tab:blue'
z500_lw = 1.8

# ── OLR hatching settings ──
olr_pos_threshold = 5
olr_neg_threshold = -5

# ── MSLP absolute field settings (contours in steps of 2 hPa) ──
mslp_levels_thin = np.arange(1000, 1042, 2)
mslp_levels_thick = np.arange(1002, 1042, 6)
mslp_color = 'black'
mslp_lw_thin = 0.2
mslp_lw_thick = 0.8

# ── MSLP high-pressure hatching ──
mslp_hp_threshold = 1023
mslp_hp_color = 'tab:red'

# ── Font sizes ──
label_fs = 18
tick_fs = 14
title_fs = 16

# ── Gridline labels ──
draw_gridline_labels = True
gridline_label_fs    = 12      # label font size
gridline_lon_step    = 30     # longitude tick spacing (deg)
gridline_lat_step    = 15     # latitude tick spacing (deg)
gridline_label_pad   = 2      # gap between axis and label (points)
panel_label_sides = {}

# ── Optional: outline the sector detection zone, as in Fig. 3 (ADDED) ──
draw_sector_box  = True
sector_box_lats  = (-55, -30)
sector_box_halfw = 15
sector_box_kw    = dict(color='k', linewidth=1.2, linestyle='--', alpha=0.8)

# ── Group border settings ──
border_lw = 4.0       # line width of the region outline (points)
border_pad = 0.008    # extra padding around each panel group (figure coords)


#%% 2. Helper: fix antimeridian gap ------------------------------------------
def fix_antimeridian(ds, varname_or_array, lon_coord='lon'):
    lon = ds[lon_coord].values
    if isinstance(varname_or_array, str):
        data = ds[varname_or_array].values
    else:
        data = np.array(varname_or_array)

    if 180.0 not in lon and -180.0 in lon:
        lon_new = np.append(lon, 180.0)
        idx_minus180 = np.argmin(np.abs(lon - (-180.0)))
        data_new = np.concatenate([data, data[:, idx_minus180:idx_minus180+1]], axis=1)
        return lon_new, data_new
    return lon, data


#%% 2b. Helper: labelled gridlines and sector box (ADDED — Reviewer 1) -------
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker


def wrap_lon180(x):
    """Wrap longitudes onto [-180, 180)."""
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def add_labelled_gridlines(ax, real_extent, left=True, bottom=True,
                           top=False, right=False,
                           lon_step=None, lat_step=None, fs=None, pad=None):
    """Gridlines with small lat/lon labels on the requested sides.

    `real_extent` is [lon_min, lon_max, lat_min, lat_max] in TRUE longitudes,
    i.e. before any shift applied for a non-zero central_longitude.
    """
    lon_step = gridline_lon_step  if lon_step is None else lon_step
    lat_step = gridline_lat_step  if lat_step is None else lat_step
    fs       = gridline_label_fs  if fs       is None else fs
    pad      = gridline_label_pad if pad      is None else pad

    lon_min, lon_max, lat_min, lat_max = real_extent
    lon_ticks = np.arange(np.ceil(lon_min / lon_step) * lon_step,
                          lon_max + 0.1, lon_step)
    lat_ticks = np.arange(np.ceil(lat_min / lat_step) * lat_step,
                          lat_max + 0.1, lat_step)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=draw_gridline_labels,
                      linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)
    gl.xlocator   = mticker.FixedLocator(np.unique(wrap_lon180(lon_ticks)))
    gl.ylocator   = mticker.FixedLocator(lat_ticks)
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    gl.top_labels,  gl.bottom_labels = top,  bottom
    gl.left_labels, gl.right_labels  = left, right
    gl.xlabel_style = {'size': fs}
    gl.ylabel_style = {'size': fs}
    gl.xpadding = pad
    gl.ypadding = pad
    gl.rotate_labels = False      # ignored by older cartopy, harmless
    return gl


def add_sector_box(ax, sector_lon):
    """Outline the LLB detection zone for this sector (30-55S, +-15 deg)."""
    lo, hi = sector_lon - sector_box_halfw, sector_lon + sector_box_halfw
    lat0, lat1 = sector_box_lats
    edge = np.linspace(lo, hi, 60)
    xs = np.concatenate([edge, edge[::-1], edge[:1]])
    ys = np.concatenate([np.full(edge.size, lat0),
                         np.full(edge.size, lat1), [lat0]])
    ax.plot(xs, ys, transform=ccrs.PlateCarree(), zorder=6, **sector_box_kw)


#%% 3. Load data and resolve sector indices -----------------------------------
datasets = {name: xr.open_dataset(path) for name, path in data_paths.items()}

sector_picks = {}
for name, chosen_lons in manual_sectors.items():
    ds = datasets[name]
    reg_vals = ds.reg.values

    indices = []
    display_lons = []
    for lon in chosen_lons:
        idx = int(np.argmin(np.abs(reg_vals - lon)))
        indices.append(idx)
        dl = reg_vals[idx]
        display_lons.append(dl - 360 if dl > 180 else dl)

    sector_picks[name] = {
        'orig_idx': indices,
        'lons': display_lons,
        'labels': sector_labels_per_region[name],
    }


#%% 3b. Mean LLB centroid per selected sector ------------
import pandas as pd
import os

compute_sector_centroids = True
recompute_centroids      = False    # True to rebuild the cache from the masks
draw_mean_centroid       = False    # True to also mark the centroid on Fig. 5

centroid_cache  = f'../Data/Output_data/SectorCentroids_{which_season}.csv'
catalogue_path  = '../Data/Blocks_data/03-Blocking_daily_catalogue_1970_2025_SH.csv'
masks_path      = '../Data/Blocks_data/03-CatalogueMasks_1970_2025_SH.nc'

types_llb              = ['Ridge', 'Omega block', 'Rex block (hybrid)']
season_months          = {'winter': [5, 6, 7, 8, 9],
                          'summer': [11, 12, 1, 2, 3]}[which_season]
sector_lat_band        = (-55, -30)
sector_half_width      = 15
min_area_in_sector_km2 = 5e5
min_fraction_in_sector = 0.5

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


def lon_signed_diff(a, b):
    """Signed longitude difference a - b, wrapped onto (-180, 180]."""
    return (np.asarray(a, dtype=float) - b + 180.0) % 360.0 - 180.0


def circular_mean_lon(lon_deg, weights=None):
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=float))
    w = np.ones_like(lon_rad) if weights is None else np.asarray(weights, dtype=float)
    ang = np.rad2deg(np.arctan2(np.sum(w * np.sin(lon_rad)) / np.sum(w),
                                np.sum(w * np.cos(lon_rad)) / np.sum(w)))
    return (ang + 180.0) % 360.0 - 180.0


def build_sector_centroids():
    """Loop over the catalogue masks and collect per-event centroids."""
    cat = pd.read_csv(catalogue_path)
    cat = cat[cat.TYPE.isin(types_llb)]
    llb_keys = set(zip(cat.YEAR.astype(int), cat.MONTH.astype(int),
                       cat.DAY.astype(int), cat.SID.astype(int)))

    mk   = xr.open_dataset(masks_path).sel(time=slice('1970-01-01', '2025-12-31'))
    var  = list(mk.data_vars)[0]
    latv = mk.lat.values
    lonv = mk.lon.values
    dlat = abs(np.median(np.diff(latv)))
    dlon = abs(np.median(np.diff(lonv)))

    R = 6371.0                                   # km
    cell_area = ((np.deg2rad(dlat) * R) *
                 (np.deg2rad(dlon) * R * np.cos(np.deg2rad(latv))))[:, None]
    cell_area = np.broadcast_to(cell_area, (latv.size, lonv.size))
    lon2d, lat2d = np.meshgrid(lonv, latv)

    times = pd.DatetimeIndex(mk.time.values)
    day_idx = np.where(times.month.isin(season_months))[0]

    # (region, sector_index) -> sector centre longitude
    targets = {(name, k): picks['lons'][k]
               for name, picks in sector_picks.items()
               for k in range(len(picks['lons']))}

    rows = []
    for i in tqdm(day_idx, desc='centroids'):
        t = times[i]
        m = mk[var].isel(time=i).values
        ids = np.unique(m)
        ids = ids[ids != 0]
        if ids.size == 0:
            continue
        for sid in ids:
            if (int(t.year), int(t.month), int(t.day), int(sid)) not in llb_keys:
                continue
            cell = (m == sid)
            ap   = cell_area[cell]
            latp = lat2d[cell]
            lonp = lon2d[cell]
            a_tot = ap.sum()
            if a_tot <= 0:
                continue
            in_band = (latp >= sector_lat_band[0]) & (latp <= sector_lat_band[1])

            for (name, k), slon in targets.items():
                in_sec = in_band & (np.abs(lon_signed_diff(lonp, slon)) <= sector_half_width)
                a_sec = ap[in_sec].sum()
                if a_sec < min_area_in_sector_km2 or a_sec / a_tot < min_fraction_in_sector:
                    continue
                rows.append({
                    'region': name, 'sector_idx': k, 'sector_lon': slon,
                    'date': t.strftime('%Y-%m-%d'), 'SID': int(sid),
                    'clat_full':  np.average(latp, weights=ap),
                    'clon_full':  circular_mean_lon(lonp, ap),
                    'clat_insec': np.average(latp[in_sec], weights=ap[in_sec]),
                    'clon_insec': circular_mean_lon(lonp[in_sec], ap[in_sec]),
                    'area_km2':   a_tot,
                })
    return pd.DataFrame(rows)


def summarise_centroids(ev):
    """Aggregate per-event centroids into one row per panel."""
    out = []
    for (name, k), g in ev.groupby(['region', 'sector_idx']):
        slon = g.sector_lon.iloc[0]
        off_full  = lon_signed_diff(g.clon_full.values,  slon)
        off_insec = lon_signed_diff(g.clon_insec.values, slon)
        out.append({
            'region': name, 'sector_idx': k,
            'label': sector_picks[name]['labels'][k],
            'sector_lon': slon, 'n_days': len(g),
            'clat_full_mean':   g.clat_full.mean(),
            'clat_full_median': g.clat_full.median(),
            'clat_full_p25':    g.clat_full.quantile(0.25),
            'clat_full_p75':    g.clat_full.quantile(0.75),
            'dlon_full_mean':   off_full.mean(),
            'dlon_full_iqr':    np.percentile(off_full, 75) - np.percentile(off_full, 25),
            'clat_insec_mean':  g.clat_insec.mean(),
            'dlon_insec_mean':  off_insec.mean(),
            'clat_diff':        g.clat_full.mean() - g.clat_insec.mean(),
        })
    return pd.DataFrame(out).sort_values(['region', 'sector_idx'])


sector_centroids = {}
if compute_sector_centroids:
    if os.path.exists(centroid_cache) and not recompute_centroids:
        cent_summary = pd.read_csv(centroid_cache)
        print(f'Loaded cached centroids from {centroid_cache}')
    else:
        events_df = build_sector_centroids()
        cent_summary = summarise_centroids(events_df)
        os.makedirs(os.path.dirname(centroid_cache), exist_ok=True)
        cent_summary.to_csv(centroid_cache, index=False)
        # events_df.to_csv(centroid_cache.replace('.csv', '_events.csv'), index=False)

    pd.set_option('display.width', 200)
    print(f'\n=== Mean LLB centroid per Fig. 5 panel ({which_season}) ===')
    print(cent_summary[['region', 'label', 'sector_lon', 'n_days',
                        'clat_full_mean', 'clat_full_p25', 'clat_full_p75',
                        'dlon_full_mean', 'dlon_full_iqr',
                        'clat_insec_mean', 'clat_diff']].round(1).to_string(index=False))
    print('\nclat_* are centroid latitudes; dlon_* are centroid longitudes RELATIVE '
          'to the sector centre. "full" uses the whole structure, "insec" only the '
          'part inside the sector box; clat_diff is the disagreement between them.')
    print(f'Spread of mean centroid latitude across panels: '
          f'{cent_summary.clat_full_mean.min():.1f} to {cent_summary.clat_full_mean.max():.1f} deg')
    print(f'Largest full-vs-insector centroid disagreement: '
          f'{cent_summary.clat_diff.abs().max():.1f} deg latitude')

    sector_centroids = {(r.region, int(r.sector_idx)):
                        (r.sector_lon + r.dlon_full_mean, r.clat_full_mean)
                        for r in cent_summary.itertuples()}


#%% 4. Custom panel placement table
panel_table = [
    # col, row, region, sector_idx
    (0, 0, 'SAm', 0),   # a
    (0, 1, 'SAm', 1),   # b
    (0, 2, 'SAm', 2),   # c
    (1, 0, 'SAf', 0),   # d
    (1, 1, 'SAf', 1),   # e
    (2, 0, 'Oce', 0),   # f
    (2, 1, 'Oce', 1),   # g
    (1, 2, 'Oce', 2),   # h  ← moved left
    (2, 2, 'Oce', 3),   # i  ← moved up
]
panel_letters = 'abcdefghi'

max_rows = 4
n_cols = 3

proj_data = ccrs.PlateCarree()

fig = plt.figure(figsize=(18, 22))

# ── Panel geometry ──
left_margin   = 0.06
right_margin  = 0.02
bottom_margin = 0.10
top_margin    = 0.04
hspace        = -0.23
wspace        = 0.025

total_w = 1.0 - left_margin - right_margin
total_h = 1.0 - bottom_margin - top_margin
pw = (total_w - (n_cols - 1) * wspace) / n_cols
ph = (total_h - (max_rows - 1) * hspace) / max_rows

def panel_position(col_i, row_j):
    """Return (x0, y0) lower-left corner in figure coords."""
    x0 = left_margin + col_i * (pw + wspace)
    y0 = 1.0 - top_margin - (row_j + 1) * ph - row_j * hspace
    return x0, y0

def get_oce_proj_and_extent(sector_idx):
    """Return (projection, extent) for an Oceania sector."""
    if sector_idx < 2:
        clon     = oce_central_lon_W
        real_ext = oce_extent_W
    else:
        clon     = oce_central_lon_E
        real_ext = oce_extent_E
    proj = ccrs.PlateCarree(central_longitude=clon)
    shifted_ext = [real_ext[0] - clon, real_ext[1] - clon,
                   real_ext[2], real_ext[3]]
    return proj, shifted_ext, real_ext


#%% 5. Create axes and draw panels 
region_axes = {name: [] for name in region_cfg}   
panel_axes  = {}                                  

cf_ref = None

for letter, (col_i, row_j, name, sec_i) in zip(panel_letters, panel_table):
    cfg  = region_cfg[name]
    ds   = datasets[name]
    picks = sector_picks[name]

    # ── Projection and extent ──
    if name == 'Oce':
        proj_map, ext, real_ext = get_oce_proj_and_extent(sec_i)
    else:
        proj_map = ccrs.PlateCarree(central_longitude=cfg['central_longitude'])
        ext = cfg['zoom_extent']
        real_ext = cfg['zoom_extent']       # central_longitude = 0 here

    x0, y0 = panel_position(col_i, row_j)
    ax = fig.add_axes([x0, y0, pw, ph], projection=proj_map)
    region_axes[name].append(ax)
    panel_axes[letter] = ax

    ax.set_extent(ext, crs=proj_map)

    # Background
    ax.add_feature(cfeature.LAND.with_scale('50m'),      facecolor='#f7f7f7', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),     facecolor='white',   zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8,       zorder=4)
    # ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=0.5,       zorder=4)
    sides = dict(left=(col_i == 0), bottom=True, top=False, right=False)
    sides.update(panel_label_sides.get(letter, {}))
    add_labelled_gridlines(ax, real_ext, **sides)

    oi           = int(picks['orig_idx'][sec_i])
    sector_lon   = picks['lons'][sec_i]
    sector_label = picks['labels'][sec_i]

    if draw_sector_box:
        add_sector_box(ax, sector_lon)

    if draw_mean_centroid and (name, sec_i) in sector_centroids:
        c_lon, c_lat = sector_centroids[(name, sec_i)]
        ax.plot(c_lon, c_lat, marker='*', markersize=16, color='k',
                markerfacecolor='gold', markeredgewidth=1.0,
                transform=ccrs.PlateCarree(), zorder=8)

    # ── IVT anomaly ──
    ivt_anom = (ds['ivt_llb'].isel(reg=oi) - ds['ivt_clima'].isel(reg=oi)).values
    lon_ivt, ivt_anom_fixed = fix_antimeridian(ds, ivt_anom)
    cf = ax.contourf(lon_ivt, ds['lat'].values, ivt_anom_fixed,
                     levels=ivt_levels, cmap=ivt_cmap, norm=ivt_norm,
                     extend='both', transform=proj_data, zorder=2)
    cf_ref = cf
    ax.contour(lon_ivt, ds['lat'].values, ivt_anom_fixed,
               levels=ivt_contour_levels, colors='grey',
               linewidths=0.4, linestyles='solid', transform=proj_data, zorder=3)

    # ── Z500 anomaly ──
    z500_anom = (ds['z500_llb'].isel(reg=oi) - ds['z500_clima'].isel(reg=oi)).values
    lon_z, z500_anom_fixed = fix_antimeridian(ds, z500_anom)
    ax.contour(lon_z, ds['lat'].values, z500_anom_fixed,
               levels=z500_levels_pos, colors=z500_color_pos,
               linewidths=z500_lw, linestyles='solid', transform=proj_data, zorder=5)
    ax.contour(lon_z, ds['lat'].values, z500_anom_fixed,
               levels=z500_levels_neg, colors=z500_color_neg,
               linewidths=z500_lw, linestyles='solid', transform=proj_data, zorder=5)

    # ── OLR anomaly ──
    olr_anom = (ds['olr_llb'].isel(reg=oi) - ds['olr_clima'].isel(reg=oi)).values
    lon_olr, olr_anom_fixed = fix_antimeridian(ds, olr_anom)
    olr_pos = (olr_anom_fixed >= olr_pos_threshold).astype(int)
    if np.any(olr_pos):
        ax.contourf(lon_olr, ds['lat'].values, olr_pos,
                    levels=[0.5, 1.5], hatches=['+'], colors='none',
                    transform=proj_data, zorder=3)
    olr_neg = (olr_anom_fixed <= olr_neg_threshold).astype(int)
    if np.any(olr_neg):
        ax.contourf(lon_olr, ds['lat'].values, olr_neg,
                    levels=[0.5, 1.5], hatches=['//'], colors='none',
                    transform=proj_data, zorder=3)

    # ── MSLP ──
    mslp_vals = ds['mslp_llb'].isel(reg=oi).values.copy()
    if np.nanmean(mslp_vals) > 50000:
        mslp_vals = mslp_vals / 100.0
    lon_mslp, mslp_fixed = fix_antimeridian(ds, mslp_vals)
    ax.contour(lon_mslp, ds['lat'].values, mslp_fixed,
               levels=mslp_levels_thin, colors=mslp_color,
               linewidths=mslp_lw_thin, linestyles='-', transform=proj_data, zorder=5)
    cs_thick = ax.contour(lon_mslp, ds['lat'].values, mslp_fixed,
               levels=mslp_levels_thick, colors=mslp_color,
               linewidths=mslp_lw_thick, linestyles='-', transform=proj_data, zorder=5)
    ax.clabel(cs_thick, inline=True, fontsize=8, fmt='%d')
    hp_mask = (mslp_fixed >= mslp_hp_threshold).astype(int)
    if np.any(hp_mask):
        hp_cf = ax.contourf(lon_mslp, ds['lat'].values, hp_mask,
                            levels=[0.5, 1.5], hatches=['xxx'], colors='none',
                            transform=proj_data, zorder=3)
        for collection in hp_cf.collections:
            collection.set_edgecolor(mslp_hp_color)
            collection.set_linewidth(0.0)

    # ── Panel label ──
    if sector_lon < 0:
        label_text = f'{letter}) {sector_label}  ({abs(sector_lon):.0f}°W)'
    else:
        label_text = f'{letter}) {sector_label}  ({abs(sector_lon):.0f}°E)'
    ax.text(0.03, 0.05, label_text, transform=ax.transAxes, fontsize=title_fs,
            va='bottom', ha='left',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.9),
            zorder=10)

    # ── Remove per-panel spine colouring (replaced by group border below) ──
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')
        spine.set_linewidth(0.8)

    # ── Region title on first panel of each region ──
    if letter in ('a', 'd', 'f'):
        ax.set_title(cfg['full_name'], fontsize=label_fs, fontweight='bold', pad=10)


#%% 6. Draw grouped region borders
fig.canvas.draw()

pad = border_pad

def get_ax_bbox(ax):
    """Return (x0, y0, x1, y1) in figure coordinates for a rendered axes."""
    bb = ax.get_position()
    return bb.x0, bb.y0, bb.x1, bb.y1

def draw_polygon_border(fig, vertices, color, lw):
    """Draw a closed polygon border through (x,y) figure-coord vertices."""
    xs = [v[0] for v in vertices] + [vertices[0][0]]
    ys = [v[1] for v in vertices] + [vertices[0][1]]
    line = mlines.Line2D(xs, ys, transform=fig.transFigure,
                         color=color, linewidth=lw,
                         solid_capstyle='round', solid_joinstyle='round',
                         zorder=20, clip_on=False)
    fig.add_artist(line)


# ── SAm: simple rectangle (col 0, rows 0–2) ──
sam_bboxes = [get_ax_bbox(ax) for ax in region_axes['SAm']]
xmin = min(b[0] for b in sam_bboxes)
ymin = min(b[1] for b in sam_bboxes)
xmax = max(b[2] for b in sam_bboxes)
ymax = max(b[3] for b in sam_bboxes)
draw_polygon_border(fig, [
    (xmin - pad - 0.02, ymin - pad - 0.005),
    (xmax + pad, ymin - pad - 0.005),
    (xmax + pad, ymax + pad + 0.015),
    (xmin - pad - 0.02, ymax + pad + 0.015),
], region_cfg['SAm']['box_color'], border_lw)

# ── SAf: simple rectangle (col 1, rows 0–1) ──
saf_bboxes = [get_ax_bbox(ax) for ax in region_axes['SAf']]
xmin = min(b[0] for b in saf_bboxes)
ymin = min(b[1] for b in saf_bboxes)
xmax = max(b[2] for b in saf_bboxes)
ymax = max(b[3] for b in saf_bboxes)
draw_polygon_border(fig, [
    (xmin - pad, ymin - pad - 0.003),
    (xmax + pad, ymin - pad - 0.003),
    (xmax + pad, ymax + pad + 0.015),
    (xmin - pad, ymax + pad + 0.015),
], region_cfg['SAf']['box_color'], border_lw)

# Actual rendedered boxes
fb = get_ax_bbox(panel_axes['f'])   # (x0, y0, x1, y1)
gb = get_ax_bbox(panel_axes['g'])
hb = get_ax_bbox(panel_axes['h'])
ib = get_ax_bbox(panel_axes['i'])

oce_poly = [
    (fb[0] - pad, fb[3] + pad + 0.015),   # P1  top-left of f
    (fb[2] + pad, fb[3] + pad + 0.015),   # P2  top-right of f
    (ib[2] + pad, ib[1] - pad - 0.005),   # P3  bottom-right of i  (right edge continuous)
    (hb[2] + pad, hb[1] - pad - 0.005),   # P4  step inward: bottom-right → connects to h row
    (hb[0] - pad, hb[1] - pad - 0.005),   # P5  bottom-left of h
    (hb[0] - pad, hb[3] + pad - 0.003),   # P6  top-left of h
    (fb[0] - pad, hb[3] + pad - 0.003),   # P7  jog right to col 2 left edge (= gb bottom)
    # closes back to P1
]

draw_polygon_border(fig, oce_poly, region_cfg['Oce']['box_color'], border_lw)


#%% 8. Shared colorbar --------------------------------------------------------
cbar_ax = fig.add_axes([0.1, 0.3, 0.84, 0.017])
cb = fig.colorbar(cf_ref, cax=cbar_ax, orientation='horizontal', extend='both',
                  ticks=ivt_levels)
cb.set_label('IVT anomaly (kg m⁻¹ s⁻¹)', fontsize=label_fs+1.5)
cb.ax.tick_params(labelsize=tick_fs+1.5)


#%% 9. Legend -----------------------------------------------------------------
legend_handles = [
    mlines.Line2D([], [], color=z500_color_pos, linewidth=z500_lw, linestyle='solid',
                  label=f'Z500 anom. ≥ +{z500_threshold} m'),
    mlines.Line2D([], [], color=z500_color_neg, linewidth=z500_lw, linestyle='solid',
                  label=f'Z500 anom. ≤ −{z500_threshold} m'),
    mpatches.Patch(facecolor='none', edgecolor='black', hatch='+',
                   label=f'OLR anom. ≥ +{olr_pos_threshold} W m⁻²'),
    mpatches.Patch(facecolor='none', edgecolor='black', hatch='//',
                   label=f'OLR anom. ≤ {olr_neg_threshold} W m⁻²'),
    mlines.Line2D([], [], color=mslp_color, linewidth=mslp_lw_thick, linestyle='-',
                  label='MSLP (2 hPa steps)'),
    mpatches.Patch(facecolor='none', edgecolor=mslp_hp_color, hatch='xxx',
                   label=f'MSLP ≥ {mslp_hp_threshold} hPa'),
]

fig.legend(handles=legend_handles, ncol=3,
           fontsize=tick_fs + 2.5, frameon=True, fancybox=True,
           bbox_to_anchor=(0.815, 0.366))


#%% 10. Save ------------------------------------------------------------------
outpath = '../Figures/Fig5_HemisphericDrivers.png'
plt.savefig(outpath, dpi=600, bbox_inches='tight', facecolor='white')