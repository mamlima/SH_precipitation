# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

# ── Shared configuration ──────────────────────────────────────────────────────
alpha = 0.025

sector_centers = np.arange(-165, 195, 10)   # 36 sector centre longitudes

region_extents = {
    'SAm': [-84, -34, -58,  0],
    'SAf': [  8,  52, -36,  0],
    'Oce': [ 95, 180, -50,  0],
}

region_colors = {
    'SAm': 'tab:red',
    'SAf': 'tab:blue',
    'Oce': 'tab:green',
}

cmap = plt.get_cmap('BrBG')

levels_clim = np.append(np.arange(-3., -0.3, 0.3),
                         -np.arange(-3., -0.3, 0.3)[::-1])
levels_ext  = levels_clim * 3

valid_thresh = {
    'winter': {'clim': 0.5, 'ext': 3.0},
    'summer': {'clim': 3.0, 'ext': 9.0},
}

proj_pc      = ccrs.PlateCarree()
regions_order = ['SAm', 'SAf', 'Oce']


# ── Helper: draw one region sub-axis ─────────────────────────────────────────

def draw_region_subax(ax, ds, reg_idx, mode, region, levels, valid_min):
    ax.set_extent(region_extents[region], crs=proj_pc)

    ax.add_feature(cfeature.LAND.with_scale('50m'),
                   facecolor='#f0efe9', zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                   facecolor='#e8f0f8', zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                   linewidth=0.4, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                   linewidth=0.2, zorder=4)

    if mode == 'clim':
        clim  = ds['precip_clima'].isel(reg=reg_idx)
        event = ds['precip_llb'].isel(reg=reg_idx)
        pval  = ds['pval_llb'].isel(reg=reg_idx)
    else:
        clim  = ds['precip_ext'].isel(reg=reg_idx)
        event = ds['precip_llb_ext'].isel(reg=reg_idx)
        pval  = ds['pval_llb_ext'].isel(reg=reg_idx)

    anomaly     = (event - clim).values
    valid_mask  = ~np.isnan(clim.values) & (clim.values >= valid_min)
    sig_any     = (((pval.values >= (1 - alpha)) | (pval.values <= alpha))
                   & valid_mask)
    anom_masked = np.where(sig_any, anomaly, np.nan)

    cf = None
    if np.any(np.isfinite(anom_masked)):
        cf = ax.contourf(ds['lon'], ds['lat'], anom_masked,
                         levels=levels, cmap=cmap, extend='both',
                         transform=proj_pc, zorder=2)

    # Coloured spine per region
    for spine in ax.spines.values():
        spine.set_edgecolor(region_colors[region])
        spine.set_linewidth(0.9)

    ax.set_xticks([])
    ax.set_yticks([])

    return cf


def sector_lon_label(lon_val):
    if lon_val > 180:
        lon_val -= 360
    if lon_val < 0:
        return f'{abs(lon_val):.0f}°W'
    elif lon_val > 0:
        return f'{lon_val:.0f}°E'
    return '0°'


# ── Main loop ─────────────────────────────────────────────────────────────────

configs = [
    ('winter', 'clim', 'Winter – Climatology'),
    ('winter', 'ext',  'Winter – Extremes'),
    ('summer', 'clim', 'Summer – Climatology'),
    ('summer', 'ext',  'Summer – Extremes'),
]

for season, mode, fig_title in configs:

    datasets = {
        r: xr.open_dataset(
            f'../Data/Output_data/RidgePrecipImpact_{r}_{season}.nc')
        for r in regions_order
    }

    levels = levels_clim if mode == 'clim' else levels_ext
    vmin   = valid_thresh[season][mode]

    n_panel_cols = 3    # panels per row
    n_panel_rows = 12   # rows  →  3×12 = 36 panels total
    n_sub        = 3    # sub-axes per panel (SAm | SAf | Oce)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 28))

    # Outer GridSpec: 12 rows × 3 cols of panels
    outer = gridspec.GridSpec(
        n_panel_rows, n_panel_cols,
        figure=fig,
        top=0.952, bottom=0.055,
        left=0.03, right=0.99,
        hspace=0.42, wspace=0.07,
    )

    cf_ref = None

    for panel_idx in tqdm(range(36), desc=fig_title):
        row_i = panel_idx // n_panel_cols
        col_i = panel_idx  % n_panel_cols

        # Inner GridSpec: 1 × 3 (one sub-axis per region)
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_sub,
            subplot_spec=outer[row_i, col_i],
            wspace=0.04,
        )

        for sub_j, region in enumerate(regions_order):
            ax = fig.add_subplot(inner[0, sub_j], projection=proj_pc)

            cf = draw_region_subax(ax, datasets[region], panel_idx,
                                   mode, region, levels, vmin)
            if cf is not None:
                cf_ref = cf

            # Region abbreviation label above sub-axis (first row only)
            if row_i == 0:
                ax.set_title(region, fontsize=6.5,
                             color=region_colors[region],
                             pad=2, fontweight='bold')

        # ── Panel label: sector index + longitude, centred above the trio ──
        lon_c     = sector_centers[panel_idx]
        lon_str   = sector_lon_label(lon_c)
        panel_lbl = f'R{panel_idx+1:02d}  {lon_str}'

        bbox = outer[row_i, col_i].get_position(fig)
        fig.text(
            bbox.x0 + bbox.width / 2,
            bbox.y1 + 0.003,
            panel_lbl,
            ha='center', va='bottom',
            fontsize=12, color='#333333',
        )

    # ── Shared colorbar ───────────────────────────────────────────────────────
    if cf_ref is not None:
        cbar_ax = fig.add_axes([0.10, 0.022, 0.80, 0.014])
        cb = fig.colorbar(cf_ref, cax=cbar_ax,
                          orientation='horizontal', extend='both')
        unit_lbl = ('Extreme precip. anomaly (mm/day)'
                    if mode == 'ext' else 'Precipitation anomaly (mm/day)')
        cb.set_label(unit_lbl, fontsize=16)
        if mode == 'clim':
            cb.set_ticks([-3, -2.4, -1.8, -1.2, -0.6, 0,
                           0.6, 1.2, 1.8, 2.4, 3])
        else:
            cb.set_ticks(
                np.array([-3, -2.4, -1.8, -1.2, -0.6, 0,
                           0.6, 1.2, 1.8, 2.4, 3]) * 3)
        cb.ax.tick_params(labelsize=12)

    # ── Significance note ─────────────────────────────────────────────────────
    fig.text(0.5, 0.042,
             (f'Only significant anomalies shown  '
              f'(two-tailed α = {alpha*2:.3f};  '
              f'clim. mask ≥ {vmin} mm/day)'),
             ha='center', va='bottom', fontsize=12,
             color='#666666', style='italic')

    # ── Save ──────────────────────────────────────────────────────────────────
    outpath = (f'../Figures/99-LLB_supplementary_{season}_{mode}.png')
    plt.savefig(outpath, dpi=600, bbox_inches='tight', facecolor='white')
    print(f'   Saved → {outpath}')
    plt.close(fig)

    for ds in datasets.values():
        ds.close()