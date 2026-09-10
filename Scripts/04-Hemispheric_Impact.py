#%% 0. Imports ----------------------------------------------------------------
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np

which_season = 'winter'

#%% 1. Configuration ----------------------------------------------------------
data_paths = {
    'SAm': f'../Data/Output_data/RidgePrecipImpact_SAm_{which_season}.nc',
    'SAf': f'../Data/Output_data/RidgePrecipImpact_SAf_{which_season}.nc',
    'Oce': f'../Data/Output_data/RidgePrecipImpact_Oce_{which_season}.nc',
}

alpha = 0.025

USE_FDR = True
fdr_paths = {
    'SAm': f'../Data/Output_data/RidgePrecipFDR_SAm_{which_season}.nc',
    'SAf': f'../Data/Output_data/RidgePrecipFDR_SAf_{which_season}.nc',
    'Oce': f'../Data/Output_data/RidgePrecipFDR_Oce_{which_season}.nc',
}

# ── Sector centres as defined in the data (original grid) ──
sector_centers = np.arange(-165, 195, 10)

# ── Manually chosen LLB sector longitudes per region ──
manual_sectors = {
    'SAm': [-25, -55, -85],     # Pacific, continent, Atlantic
    'SAf': [15, 75],            # 2 sectors only
    'Oce': [85, 115, 165, -155],  # 4 sectors
}
sector_labels_3 = ['East', 'Over', 'West']
sector_labels_2 = ['West', 'East']
sector_labels_4 = ['West', 'Over West', 'Over East', 'East']

sector_labels_per_region = {
    'SAm': sector_labels_3,
    'SAf': sector_labels_2,
    'Oce': sector_labels_4,
}

# ── Colormap and contour levels ──
cmap = plt.get_cmap('BrBG')
levels_regular = np.append(np.arange(-3., -0.3, 0.3),
                           -np.arange(-3, -0.3, 0.3)[::-1])

# ── Region display settings ──
region_cfg = {
    'SAm': {
        'full_name': 'South America', 'label': 'SAm',
        'zoom_extent': [-84, -34, -58, 0],
        'box_color': 'tab:red', 'centre_lon': -60,
    },
    'SAf': {
        'full_name': 'South Africa', 'label': 'SAf',
        'zoom_extent': [8, 52, -36, 0],
        'box_color': 'tab:blue', 'centre_lon': 20,
    },
    'Oce': {
        'full_name': 'Oceania', 'label': 'Oce',
        'zoom_extent': [95, 180, -50, 0],
        'box_color': 'tab:green', 'centre_lon': 145,
    },
}

label_fs = 24
tick_fs = 16
title_fs = 40
annot_fs = 11


#%% 2. Load data and resolve sector indices -----------------------------------
datasets = {name: xr.open_dataset(path) for name, path in data_paths.items()}
fdr_sets = ({name: xr.open_dataset(p) for name, p in fdr_paths.items()}
            if USE_FDR else {})      # ADDED

def sector_lon_to_index(sector_lon, sector_centers):
    idx = np.argmin(np.abs(sector_centers - sector_lon))
    return idx

sector_picks = {}
for name, chosen_lons in manual_sectors.items():
    orig_indices = [sector_lon_to_index(lon, sector_centers)
                    for lon in chosen_lons]
    actual_lons = [sector_centers[i] for i in orig_indices]
    display_lons = [l - 360 if l > 180 else l for l in actual_lons]
    labels = sector_labels_per_region[name]

    sector_picks[name] = {
        'orig_idx': orig_indices,
        'lons': display_lons,
        'labels': labels,
    }


#%% 3. Figure layout ----------------------------------------------------------
fig = plt.figure(figsize=(20, 15))

proj_stereo = ccrs.SouthPolarStereo()
proj_pc = ccrs.PlateCarree()

# ── Globe (centre) ──
globe_size = 0.45
globe_pos = [0.545 - globe_size / 2, 0.2, globe_size, globe_size * (20.0 / 15.0)]

# ── SAm panels (LEFT, vertical stack) ──
sam_pw, sam_ph = 0.27, 0.35
sam_gap = 0.025
sam_x = 0.00
sam_y0 = 0.78
sam_positions = [
    [sam_x, sam_y0 - i * (sam_ph + sam_gap), sam_pw, sam_ph]
    for i in range(3)
]

# ── SAf panels (TOP, horizontal row, 2 panels) ──
saf_pw, saf_ph = 0.32, 0.28
saf_gap = -0.05
saf_y = 0.92
saf_x0 = 0.25
saf_positions = [
    [saf_x0 + i * (saf_pw + saf_gap), saf_y, saf_pw, saf_ph]
    for i in range(2)
]

# ── Oce panels (RIGHT, vertical stack, 4 panels) ──
oce_pw, oce_ph = 0.32, 0.23
oce_gap = 0.025
oce_x = 0.83
oce_y0 = 0.84
oce_positions = [
    [oce_x, oce_y0 - i * (oce_ph + oce_gap), oce_pw, oce_ph]
    for i in range(4)
]

layout = {
    'SAm': sam_positions,
    'SAf': saf_positions,
    'Oce': oce_positions,
}

# ── Connection-line anchors ──
globe_cx = globe_pos[0] + globe_pos[2] / 2
globe_cy = globe_pos[1] + globe_pos[3] / 2
globe_rx = globe_pos[2] / 2
globe_ry = globe_pos[3] / 2

globe_anchors = {
    'SAm': (globe_cx - globe_rx + 0.01, globe_cy + globe_ry * 0.4),
    'SAf': (globe_cx - globe_rx * 0.3, globe_cy + globe_ry - 0.01),
    'Oce': (globe_cx + globe_rx - 0.01, globe_cy - globe_ry * 0.1),
}
panel_anchors = {
    'SAm': (sam_positions[1][0] + sam_pw,
            sam_positions[1][1] + sam_ph / 2),
    'SAf': (saf_positions[0][0] + saf_pw / 2,
            saf_positions[0][1]),
    'Oce': (oce_positions[1][0],
            oce_positions[1][1] + oce_ph / 2),
}


#%% 4. Draw the globe ---------------------------------------------------------
ax_globe = fig.add_axes(globe_pos, projection=proj_stereo)
ax_globe.set_extent([-180, 180, -90, 0], crs=proj_pc)

# Circular boundary
theta = np.linspace(0, 2 * np.pi, 100)
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * 0.5 + [0.5, 0.5])
ax_globe.set_boundary(circle, transform=ax_globe.transAxes)

ax_globe.add_feature(cfeature.NaturalEarthFeature(
    'physical', 'ocean', '50m', facecolor='#e8f0f8', edgecolor='none'), zorder=0)
ax_globe.add_feature(cfeature.NaturalEarthFeature(
    'physical', 'land', '50m', facecolor='#f0efe9', edgecolor='none'), zorder=1)
ax_globe.coastlines(resolution='50m', linewidth=0.9, color='#555555', zorder=3)
ax_globe.add_feature(cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '50m',
    edgecolor='#999999', facecolor='none', linewidth=0.7), zorder=3)

gl = ax_globe.gridlines(draw_labels=False, linewidth=0.3, color='grey',
                         alpha=0.4, linestyle='--')
gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 30))
gl.ylocator = plt.FixedLocator(np.arange(-80, 10, 10))


def draw_smooth_rect(ax, lon_min, lon_max, lat_min, lat_max, npts=1000, **kwargs):
    top = np.linspace(lon_min, lon_max, npts)
    bot = np.linspace(lon_max, lon_min, npts)
    left = np.linspace(lat_max, lat_min, npts)
    right = np.linspace(lat_min, lat_max, npts)
    lons = np.concatenate([top, np.full(npts, lon_max),
                           bot, np.full(npts, lon_min)])
    lats = np.concatenate([np.full(npts, lat_max), right,
                           np.full(npts, lat_min), left])
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), **kwargs)


# Region rectangles (smooth)
for name, cfg in region_cfg.items():
    ext = cfg['zoom_extent']
    draw_smooth_rect(ax_globe, ext[0], ext[1], ext[2], ext[3],
                     color=cfg['box_color'], linewidth=2.5, zorder=5)

ax_globe.text(0.05, 0.95, 'a)', transform=ax_globe.transAxes,
              fontsize=title_fs-5, va='top')


#%% 5. Draw zoom panels -----------------------------------------------------
fig.canvas.draw()

panel_letters = 'dcbefghijklmno'
letter_idx = 0
region_order = ['SAm', 'SAf', 'Oce']
cf_ref = None

for name in region_order:
    cfg = region_cfg[name]
    ds = datasets[name]
    picks = sector_picks[name]
    ext = cfg['zoom_extent']
    positions = layout[name]

    n_sectors = len(picks['orig_idx'])
    for col_j in range(n_sectors):
        pos = positions[col_j]
        ax = fig.add_axes(pos, projection=proj_pc)
        ax.set_extent(ext, crs=proj_pc)

        # Background (matching 01 script)
        ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#f7f7f7', zorder=1)
        ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='white', zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8, zorder=4)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.5, zorder=4)
        ax.add_feature(cfeature.RIVERS.with_scale('50m'), linewidth=0.4, alpha=0.5, zorder=4)

        gl = ax.gridlines(draw_labels=False, linewidth=0.5, linestyle='--',
                          alpha=0.5, zorder=1)

        # ── Anomaly ──
        oi = int(picks['orig_idx'][col_j])
        sector_lon = picks['lons'][col_j]
        sector_label = picks['labels'][col_j]

        precip_clim = ds['precip_clima'].isel(reg=oi)
        precip_event = ds['precip_llb'].isel(reg=oi)
        pval = ds['pval_llb'].isel(reg=oi)
        anomaly = precip_event - precip_clim

        cf = ax.contourf(ds['lon'], ds['lat'], anomaly,
                         levels=levels_regular, cmap=cmap, extend='both',
                         transform=proj_pc, zorder=2)
        cf_ref = cf

        # ── Significance hatching ──
        thr_map = {'winter': 0.5, 'summer': 1.5}
        valid_mask = (~np.isnan(precip_clim.values) &
                      (precip_clim.values >= thr_map[which_season]))

        if USE_FDR:                                     # ADDED (R2.2)
            sig_fld = fdr_sets[name]['sig_clim'].isel(reg=oi).values
            sig_high = (sig_fld == 1) & valid_mask
            sig_low  = (sig_fld == -1) & valid_mask
        else:
            sig_high = (pval.values >= (1 - alpha)) & valid_mask
            sig_low  = (pval.values <= alpha) & valid_mask
        if np.any(sig_high):
            ax.contourf(ds['lon'], ds['lat'], sig_high.astype(int),
                        levels=[0.5, 1.5], hatches=['//'], colors='none',
                        transform=proj_pc, zorder=3)

        if np.any(sig_low):
            ax.contourf(ds['lon'], ds['lat'], sig_low.astype(int),
                        levels=[0.5, 1.5], hatches=['.'], colors='none',
                        transform=proj_pc, zorder=3)

        # Panel label & title (inside plot with white box)
        letter = panel_letters[letter_idx]
        letter_idx += 1
        if sector_lon < 0:
            label_text = (f'{letter}) {sector_label}  '
                          f'({abs(sector_lon):.0f}°W)')
        elif sector_lon > 0:
            label_text = (f'{letter}) {sector_label}  '
                          f'({abs(sector_lon):.0f}°E)')

        if name == 'Oce':
            # Bottom-left for Oceania
            ax.text(0.03, 0.05, label_text,
                    transform=ax.transAxes, fontsize=title_fs-19,
                    va='bottom', ha='left',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                              edgecolor='grey', alpha=0.9),
                    zorder=10)
        else:
            # Bottom-right for SAm and SAf
            ax.text(0.96, 0.05, label_text,
                    transform=ax.transAxes, fontsize=title_fs-19,
                    va='bottom', ha='right',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                              edgecolor='grey', alpha=0.9),
                    zorder=10)

        # Border colour
        for spine in ax.spines.values():
            spine.set_edgecolor(cfg['box_color'])
            spine.set_linewidth(2.2)

        # ── LLB location box on the globe ──
        draw_smooth_rect(ax_globe,
                         sector_lon - 15, sector_lon + 15, -55, -30,
                         npts=500, color=cfg['box_color'], linewidth=2.5,
                         linestyle='--', alpha=0.7, zorder=6)

        # ── Thin line from LLB box centre (on globe) to panel corner ──
        # Use ConnectionPatch which handles cross-axes coordinate transforms
        from matplotlib.patches import ConnectionPatch

        # LLB box centre in the globe's DATA coordinates (already projected)
        llb_xy = proj_stereo.transform_point(sector_lon, -42.5, proj_pc)

        # Determine which corner of the panel to connect to
        p_left, p_bot, p_w, p_h = pos
        if name == 'SAm':
            # Panels on LEFT → connect to bottom-right corner
            panel_corner = (1.0, 0.5)  # in axes fraction
        elif name == 'SAf':
            # Panels on TOP → connect to bottom-centre
            panel_corner = (0.5, 0.0)
        else:  # Oce
            # Panels on RIGHT → connect to bottom-left corner
            panel_corner = (0.0, 0.5)

        con = ConnectionPatch(
            xyA=llb_xy, coordsA=ax_globe.transData,
            xyB=panel_corner, coordsB=ax.transAxes,
            color=cfg['box_color'], linewidth=3.5, linestyle='-',
            alpha=0.4, zorder=10)
        fig.add_artist(con)
        

#%% 6. Shared colorbar --------------------------------------------------------
cbar_ax = fig.add_axes([0.282, 0.08, 0.525, 0.02])
cb = fig.colorbar(cf_ref, cax=cbar_ax, orientation='horizontal', extend='both')
cb.set_label('Precipitation anomaly (mm/day)', fontsize=label_fs)
cb.set_ticks([-3, -2.4, -1.8, -1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3])
cb.ax.tick_params(labelsize=tick_fs+3)


#%% 7. Hatching legend --------------------------------------------------------
hatch_handles = [
    mpatches.Patch(facecolor='none', edgecolor='black', hatch='//',
                   label='Sig. positive anomaly'),
    mpatches.Patch(facecolor='none', edgecolor='black', hatch='.',
                   label='Sig. negative anomaly'),
]
legend = fig.legend(handles=hatch_handles, loc='lower right', ncol=2,
                    fontsize=label_fs-2, frameon=True, fancybox=True,
                    bbox_to_anchor=(0.79, 0.12))
frame = legend.get_frame()
frame.set_facecolor('white')
legend.set_zorder(20)


#%% 9. Save -------------------------------------------------------------------
outpath = '../Figures/FigS2_ClimaImpact_FDR.png'
plt.savefig(outpath, dpi=600, bbox_inches='tight', facecolor='white')