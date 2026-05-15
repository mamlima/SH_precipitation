#%% 0. Init
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from shapely.geometry import box
import cartopy.io.shapereader as shpreader


#%% 1. Configuration
alpha = 0.025
res   = 0.1
relevance_percentile = 82

# ── Season-dependent validity thresholds ─────────────────────────────────────
thresholds = {
    'winter': {'clim': 0.5, 'ext': 3.0},
    'summer': {'clim': 3.0, 'ext': 9.0},
}

label_size = 14
tick_size  = 12
title_size = 14

# ── Colours ──────────────────────────────────────────────────────────────────
# Winter: blue family
color_win_clim = 'tab:blue'          # dark blue  - climatology bars + dashed line
color_win_ext  = 'cornflowerblue'    # light blue - extreme bars
color_win_line = 'mediumblue'        # line colour

# Summer: red family
color_sum_clim = 'tab:red'           # dark red   - climatology bars + dashed line
color_sum_ext  = 'lightcoral'        # light red  - extreme bars
color_sum_line = 'firebrick'         # line colour

bar_alpha_clim = 0.75
bar_alpha_ext  = 0.55
bar_width      = 4.0            # width of each individual bar (degrees)
bar_gap        = 0.5            # gap between winter and summer bar groups

panel_labels = ['a)', 'b)', 'c)']
region_names = ['South America', 'Southern Africa', 'Oceania and maritime continent']


#%% 2. Load data — both seasons
seasons = ['winter', 'summer']
regions = ['SAm', 'SAf', 'Oce']

raw = {}
for season in seasons:
    raw[season] = {}
    for reg in regions:
        raw[season][reg] = xr.open_dataset(
            f'../Data/Output_data/RidgePrecipImpact_{reg}_{season}.nc')


#%% 3. Compute metrics for both seasons
sector_centers = np.arange(-165, 195, 10)

# Sort sector centres to run −180 → +180
sector_centers_plot = np.where(sector_centers > 180,
                               sector_centers - 360, sector_centers)
sort_idx = np.argsort(sector_centers_plot)
x = sector_centers_plot[sort_idx]

results = {}
for reg in regions:
    results[reg] = {}
    for season in seasons:
        ds = raw[season][reg]

        lat_rad   = np.deg2rad(ds['lat'])
        R         = 6371.0
        cell_area = (R**2) * np.abs(np.cos(lat_rad)) * np.deg2rad(res)**2

        land_mask  = ds['precip_clima'].isel(reg=0).notnull()
        thr_clim   = thresholds[season]['clim']
        thr_ext    = thresholds[season]['ext']
        valid_clim = ds['precip_clima'].isel(reg=0) > thr_clim
        valid_ext  = ds['precip_ext'].isel(reg=0)   > thr_ext
        total_land = land_mask.sum(dim=['lat', 'lon']).values

        sig_clim = ((ds['pval_llb'] < alpha) |
                    (ds['pval_llb'] > (1 - alpha))) & land_mask & valid_clim
        sig_ext  = ((ds['pval_llb_ext'] < alpha) |
                    (ds['pval_llb_ext'] > (1 - alpha))) & land_mask & valid_ext

        pct_clim = (sig_clim.sum(dim=['lat', 'lon']) / total_land * 100).values
        pct_ext  = (sig_ext.sum(dim=['lat', 'lon'])  / total_land * 100).values

        anom_clim = (np.abs(ds['precip_llb'] - ds['precip_clima']) * sig_clim).sum(
            dim=['lat', 'lon']).values
        anom_ext  = (np.abs(ds['precip_llb_ext'] - ds['precip_ext']) * sig_ext).sum(
            dim=['lat', 'lon']).values

        results[reg][season] = {
            'pct_clim':  pct_clim[sort_idx],
            'pct_ext':   pct_ext[sort_idx],
            'anom_clim': anom_clim[sort_idx],
            'anom_ext':  anom_ext[sort_idx],
            'bounds':    [float(ds['lat'].min()), float(ds['lat'].max()),
                          float(ds['lon'].min()), float(ds['lon'].max())],
        }


#%% 4. Compute combined relevance score (winter + summer average)
for reg in regions:
    combined_score = np.zeros(len(x))
    for season in seasons:
        r = results[reg][season]
        def norm(arr):
            m = np.max(arr)
            return arr / m if m > 0 else arr
        combined_score += (norm(r['pct_clim'])  +
                           norm(r['pct_ext'])   +
                           norm(r['anom_clim']) +
                           norm(r['anom_ext'])) / 4.0
    results[reg]['combined_score'] = combined_score / len(seasons)


#%% 5. Plot
fig, axes = plt.subplots(3, 1, figsize=(18, 12),
                          subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(hspace=0.1, left=0.08)

for i, (ax, reg) in enumerate(zip(axes, regions)):
    rw = results[reg]['winter']
    rs = results[reg]['summer']
    score = results[reg]['combined_score']
    threshold = np.percentile(score, relevance_percentile)

    # ── Background map ───────────────────────────────────────────────────────
    ax.set_extent([-180, 180, -85, 0], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', linewidth=0.9, color='dimgrey', zorder=1)
    ax.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'land', '50m', facecolor='gainsboro', edgecolor='none'), zorder=0)
    ax.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_0_boundary_lines_land', '50m',
        edgecolor='#999999', facecolor='none', linewidth=0.7), zorder=1)
    ax.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'rivers_lake_centerlines', '50m',
        edgecolor='#a6cee3', facecolor='none', linewidth=0.4), zorder=1)

    gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='grey',
                       alpha=0.3, linestyle='--', zorder=1)
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 10))
    gl.ylocator = plt.FixedLocator(np.arange(-80, 1, 10))

    # Shade analysis domain
    bounds = rw['bounds']   # [lat_min, lat_max, lon_min, lon_max]
    region_box = box(bounds[2], bounds[0], bounds[3], bounds[1])
    land_shp = shpreader.natural_earth(resolution='50m', category='physical', name='land')
    reader = shpreader.Reader(land_shp)
    for geom in reader.geometries():
        intersection = geom.intersection(region_box)
        if not intersection.is_empty:
            ax.add_geometries([intersection], ccrs.PlateCarree(),
                               facecolor='black', alpha=0.5, edgecolor='none', zorder=2)

    # ── Overlay axes ─────────────────────────────────────────────────────────
    # ax_line : left y-axis  — fraction (%)
    # ax_anom : right y-axis — accumulated anomaly (m)
    ax_line = fig.add_axes(ax.get_position(), frameon=False)
    ax_anom = ax_line.twinx()
    
    ax_line.set_zorder(10)
    ax_anom.set_zorder(1)

    # ── Green relevance shading ───────────────────────────────────────────────
    for j, s in enumerate(x):
        if score[j] >= threshold:
            ax_line.axvspan(s - 4.8, s + 4.8, color='green', alpha=0.1, zorder=0)

    # ── Bars: accumulated anomaly (side-by-side, on right axis) ─────────────
    x_win = x - bar_width + 2
    x_sum = x + 2 

    ax_anom.bar(x_win, rw['anom_clim'] / 1000, bar_width,
                color=color_win_clim, alpha=bar_alpha_clim, zorder=-1,
                label='Accum. |anom| clim. (win.)')
    ax_anom.bar(x_win, rw['anom_ext'] / 1000, bar_width,
                color=color_win_ext, alpha=bar_alpha_ext, zorder=-1,
                label='Accum. |anom| extr. (win.)')

    ax_anom.bar(x_sum, rs['anom_clim'] / 1000, bar_width,
                color=color_sum_clim, alpha=bar_alpha_clim, zorder=-1,
                label='Accum. |anom| clim. (sum.)')
    ax_anom.bar(x_sum, rs['anom_ext'] / 1000, bar_width,
                color=color_sum_ext, alpha=bar_alpha_ext, zorder=-1,
                label='Accum. |anom| extr. (sum.)')

    # ── Lines: fraction (on left axis) ───────────────────────────────────────
    # Winter: dashed; Summer: solid. Blue = winter, Red = summer
    ax_line.plot(x, rw['pct_clim'], color=color_win_clim, lw=2.5, marker='o', ms=7,
                 ls='--', zorder=100.5, label='Fraction clim. (win.)',
                 markeredgecolor = 'black', markeredgewidth = .6)
    ax_line.plot(x, rw['pct_ext'],  color=color_win_ext,  lw=2.5, marker='o', ms=7,
                 ls='-',  zorder=100.5, label='Fraction extr. (win.)',
                 markeredgecolor = 'black', markeredgewidth = .6)
    ax_line.plot(x, rs['pct_clim'], color=color_sum_clim, lw=2.5, marker='o', ms=7,
                 ls='--', zorder=100.5, label='Fraction clim. (sum.)',
                 markeredgecolor = 'black', markeredgewidth = .6)
    ax_line.plot(x, rs['pct_ext'],  color=color_sum_ext,  lw=2.5, marker='o', ms=7,
                 ls='-',  zorder=100.5, label='Fraction extr. (sum.)',
                 markeredgecolor = 'black', markeredgewidth = 0.6)

    # ── Axis limits and ticks ─────────────────────────────────────────────────
    ax_line.set_xlim(-180, 180)
    ax_anom.set_xlim(-180, 180)
    ax_line.set_ylim(bottom=0)
    ax_anom.set_ylim(bottom=0)

    ax_line.set_xticks(np.arange(-180, 181, 30))
    ax_line.xaxis.set_minor_locator(plt.MultipleLocator(10))
    ax_line.tick_params(axis='x', which='minor', length=3)

    ax_line.set_ylabel('Significant pixels (%)', fontsize=label_size)
    ax_anom.set_ylabel('Accum. |anomaly| (m)',   fontsize=label_size)

    ax_line.tick_params(axis='y', labelsize=tick_size)
    ax_anom.tick_params(axis='y', labelsize=tick_size)
    ax_line.tick_params(axis='x', labelsize=tick_size)

    if i < 2:
        ax_line.set_xticklabels([])
        ax_line.tick_params(axis='x', which='major', length=5)
    else:
        ax_line.set_xticklabels(
            [f'{int(s)}°' for s in np.arange(-180, 181, 30)],
            fontsize=tick_size)
        ax_line.set_xlabel('LLB sector longitude', fontsize=label_size)

    # ── Panel label ───────────────────────────────────────────────────────────
    ax_line.text(0.02, 0.94, f'{panel_labels[i]} {region_names[i]}',
                  transform=ax_line.transAxes, fontsize=title_size, va='top',
                  bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                            edgecolor='grey', alpha=0.8))

    # ── Legend (first panel only) ─────────────────────────────────────────────
    if i == 0:
        legend_handles = [
            # ── Fraction lines ──
            mlines.Line2D([], [], color=color_win_clim, lw=1.8, ls='--',
                          marker='o', ms=4, label='% Sign. Clim. Winter',
                          markeredgecolor='black', markeredgewidth=0.6),
            mlines.Line2D([], [], color=color_sum_clim, lw=1.8, ls='--',
                          marker='o', ms=4, label='% Sign. Clim. Summer',
                          markeredgecolor='black', markeredgewidth=0.6),
            mlines.Line2D([], [], color=color_win_ext,  lw=1.8, ls='-',
                          marker='o', ms=4, label='% Sign. Extr. Winter',
                          markeredgecolor='black', markeredgewidth=0.6),
            mlines.Line2D([], [], color=color_sum_ext,  lw=1.8, ls='-',
                          marker='o', ms=4, label='% Sign. Extr. Summer',
                          markeredgecolor='black', markeredgewidth=0.6),
            # ── Anomaly bars ──
            mpatches.Patch(facecolor=color_win_clim, alpha=bar_alpha_clim,
                           label=r'$\Sigma$|anom| Clim. Winter'),
            mpatches.Patch(facecolor=color_sum_clim, alpha=bar_alpha_clim,
                           label=r'$\Sigma$|anom| Clim. Summer'),
            mpatches.Patch(facecolor=color_win_ext,  alpha=bar_alpha_ext,
                           label=r'$\Sigma$|anom| Extr. Winter'),
            mpatches.Patch(facecolor=color_sum_ext,  alpha=bar_alpha_ext,
                           label=r'$\Sigma$|anom| Extr. Summer'),
        ]
        ax_line.legend(handles=legend_handles, fontsize=11,
                        loc='upper right', ncol=2,
                        frameon=True, framealpha=0.95)

plt.savefig('../Figures/Fig2_Distance_vs_significance1.png',
            dpi=600, bbox_inches='tight')