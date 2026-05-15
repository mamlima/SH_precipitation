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
    return proj, shifted_ext


#%% 5. Create axes and draw panels --------------------------------------------
# Store axes objects per region AND per panel letter for border drawing
region_axes = {name: [] for name in region_cfg}   # list of ax objects
panel_axes  = {}                                  # letter → ax

cf_ref = None

for letter, (col_i, row_j, name, sec_i) in zip(panel_letters, panel_table):
    cfg  = region_cfg[name]
    ds   = datasets[name]
    picks = sector_picks[name]

    # ── Projection and extent ──
    if name == 'Oce':
        proj_map, ext = get_oce_proj_and_extent(sec_i)
    else:
        proj_map = ccrs.PlateCarree(central_longitude=cfg['central_longitude'])
        ext = cfg['zoom_extent']

    x0, y0 = panel_position(col_i, row_j)
    ax = fig.add_axes([x0, y0, pw, ph], projection=proj_map)
    region_axes[name].append(ax)
    panel_axes[letter] = ax

    ax.set_extent(ext, crs=proj_map)

    # Background
    ax.add_feature(cfeature.LAND.with_scale('50m'),      facecolor='#f7f7f7', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),     facecolor='white',   zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8,       zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),   linewidth=0.5,       zorder=4)
    ax.gridlines(draw_labels=False, linewidth=0.5, linestyle='--', alpha=0.5, zorder=1)

    oi           = int(picks['orig_idx'][sec_i])
    sector_lon   = picks['lons'][sec_i]
    sector_label = picks['labels'][sec_i]

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


#%% 6. Draw grouped region borders -------------------------------------------
# We need the *rendered* axes positions (after Cartopy adjusts aspect ratios).
# Force a draw so that get_position() returns the final bounding boxes.
fig.canvas.draw()

pad = border_pad   # shorthand (avoid name 'p' which clashes with loop vars)

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
    (xmin - pad, ymin - pad),
    (xmax + pad, ymin - pad),
    (xmax + pad, ymax + pad + 0.015),
    (xmin - pad, ymax + pad + 0.015),
], region_cfg['SAm']['box_color'], border_lw)

# ── SAf: simple rectangle (col 1, rows 0–1) ──
saf_bboxes = [get_ax_bbox(ax) for ax in region_axes['SAf']]
xmin = min(b[0] for b in saf_bboxes)
ymin = min(b[1] for b in saf_bboxes)
xmax = max(b[2] for b in saf_bboxes)
ymax = max(b[3] for b in saf_bboxes)
draw_polygon_border(fig, [
    (xmin - pad, ymin - pad),
    (xmax + pad, ymin - pad),
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
    (ib[2] + pad, ib[1] - pad),   # P3  bottom-right of i  (right edge continuous)
    (hb[2] + pad, hb[1] - pad),   # P4  step inward: bottom-right → connects to h row
    (hb[0] - pad, hb[1] - pad),   # P5  bottom-left of h
    (hb[0] - pad, hb[3] + pad),   # P6  top-left of h
    (fb[0] - pad, hb[3] + pad),   # P7  jog right to col 2 left edge (= gb bottom)
    # closes back to P1
]

draw_polygon_border(fig, oce_poly, region_cfg['Oce']['box_color'], border_lw)


#%% 7. Region titles for panels that don't have them set yet ------------------
# SAf title is already on the first SAf panel (d).
# Oce title: f is the first Oce panel and already has its title.
# (All handled inside the loop above.)


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
outpath = '../Figures/Fig5_LLB_hemispheric_drivers.png'
plt.savefig(outpath, dpi=600, bbox_inches='tight', facecolor='white')