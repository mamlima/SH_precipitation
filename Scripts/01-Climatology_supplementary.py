#%% 0. Start
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from tqdm import tqdm
import cartopy.feature as cf
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
from matplotlib.lines import Line2D


#%% 1. Functions
def clean_nans(arr):
    return arr[~np.isnan(arr)]

def circular_mean_lon_180(lon_deg, weights=None):
    lon_deg = np.asarray(lon_deg, dtype=float)
    lon_rad = np.deg2rad(lon_deg)
    if weights is None:
        weights = np.ones_like(lon_deg, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
    mean_sin = np.sum(weights * np.sin(lon_rad)) / np.sum(weights)
    mean_cos = np.sum(weights * np.cos(lon_rad)) / np.sum(weights)
    mean_lon = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    if mean_lon > 180:
        mean_lon -= 360
    elif mean_lon <= -180:
        mean_lon += 360
    return mean_lon

def circular_percentiles(lon_deg, weights=None):
    lon = np.asarray(lon_deg, dtype=float)
    lon_rad = np.deg2rad(lon)
    if weights is None:
        weights = np.ones_like(lon)
    else:
        weights = np.asarray(weights)
    mean_sin = np.sum(weights * np.sin(lon_rad)) / np.sum(weights)
    mean_cos = np.sum(weights * np.cos(lon_rad)) / np.sum(weights)
    mean_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))
    shifted = (lon - mean_angle + 180) % 360 - 180
    p25 = np.percentile(shifted, 25)
    p50 = np.percentile(shifted, 50)
    p75 = np.percentile(shifted, 75)
    p25 = (p25 + mean_angle + 180) % 360 - 180
    p50 = (p50 + mean_angle + 180) % 360 - 180
    p75 = (p75 + mean_angle + 180) % 360 - 180
    return p25, p50, p75

def get_area_weighted_center(row, masks, mask_var):
    sid = row['SID']
    date = pd.Timestamp(
        year=int(row['YEAR']),
        month=int(row['MONTH']),
        day=int(row['DAY'])
    )
    mask_day = masks[mask_var].sel(time=date)
    structure_mask = (mask_day == sid)
    yy, xx = np.where(structure_mask.values)
    if len(yy) == 0:
        return pd.Series({'CLON': np.nan, 'CLAT': np.nan})
    lat_pts = mask_day.lat.values[yy]
    lon_pts = mask_day.lon.values[xx]
    weights = np.cos(np.deg2rad(lat_pts))
    clat = np.average(lat_pts, weights=weights)
    clon = circular_mean_lon_180(lon_pts, weights=weights)
    return pd.Series({'CLON': clon, 'CLAT': clat})

def give_climatology(months_to_eval, masks, masks_data, time, struct_types):
    Ridge_days = np.zeros(np.shape(lons))
    indexes = np.where(time.month.isin(months_to_eval))
    dates_to_an = time.astype(str).str[:10].values[indexes]
    n_days = 0
    for data_string0 in tqdm(dates_to_an):
        n_days += 1
        day_n = np.where(time == data_string0)[0][0]
        if day_n < 15:
            continue
        else:
            mask_day = masks[day_n]
            for strct_id in np.unique(mask_day):
                if strct_id != 0:
                    strct_type = masks_data[(masks_data.SID == strct_id) &
                                            (masks_data.YEAR == int(data_string0[:4])) &
                                            (masks_data.MONTH == int(data_string0[5:7])) &
                                            (masks_data.DAY == int(data_string0[8:]))].TYPE.values[0]
                    daily_struct_array = np.zeros(np.shape(lons))
                    daily_struct_array[mask_day == strct_id] = 1
                    if strct_type in struct_types:
                        Ridge_days[daily_struct_array == 1] += 1
    return Ridge_days, n_days

def make_map_on_ax(fig, pos_rect, lon_min, lon_max, lat_min, lat_max):
    """Create a cartopy axes at a specific position rectangle [x0, y0, w, h]."""
    ax = fig.add_axes(pos_rect, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, ls='--', colors='grey', alpha=0.6)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}

    land = cf.NaturalEarthFeature(category='physical', scale='50m',
                                  facecolor='gainsboro', name='land')
    ax.add_feature(land, edgecolor='k', linewidth=0.7, alpha=0.9, zorder=0)
    borders = cf.NaturalEarthFeature(category='cultural',
                                     name='admin_0_boundary_lines_land',
                                     scale='50m', facecolor='none', edgecolor='black')
    ax.add_feature(borders, edgecolor='k', linewidth=0.6, alpha=0.9, zorder=0)

    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = cticker.LongitudeFormatter()
    gl.ylocator = mticker.FixedLocator([-80, -60, -40, -20, 0])
    gl.yformatter = cticker.LatitudeFormatter()

    return ax, gl


#%% 2. Open data
daily_data = pd.read_csv('../Data/Blocks_data/03-Blocking_daily_catalogue_1970_2025_SH.csv')
daily_masks = xr.open_dataset('../Data/Blocks_data/03-CatalogueMasks_1970_2025_SH.nc')

data_full = daily_data[daily_data.YEAR.isin([i for i in range(1970, 2025+1)])]
masks_full = daily_masks.sel(time=slice('1970-01-01', '2025-12-31'))

lat = daily_masks.lat.values
lon = daily_masks.lon.values
lons, lats = np.meshgrid(lon, lat)
time_full = pd.DatetimeIndex(masks_full.time)

mask_var = list(masks_full.data_vars)[0]
types_llb = ['Ridge', 'Omega block', 'Rex block (hybrid)']
types_hlb = ['Rex block', 'Rex block (polar)']


#%% 3. Compute climatologies for both seasons
print("Computing winter LLB climatology...")
winter_LLB, ndays_winter1 = give_climatology([5,6,7,8,9], masks_full.Structs.values, data_full, time_full, types_llb)
print("Computing winter HLB climatology...")
winter_HLB, ndays_winter2 = give_climatology([5,6,7,8,9], masks_full.Structs.values, data_full, time_full, types_hlb)

print("Computing summer LLB climatology...")
summer_LLB, ndays_summer1 = give_climatology([11,12,1,2,3], masks_full.Structs.values, data_full, time_full, types_llb)
print("Computing summer HLB climatology...")
summer_HLB, ndays_summer2 = give_climatology([11,12,1,2,3], masks_full.Structs.values, data_full, time_full, types_hlb)


#%% 4. Compute first/last occurrences for both seasons
def compute_first_last(season_months):
    first_all = data_full.groupby('SID', as_index=False).first()
    last_all = data_full.groupby('SID', as_index=False).last()

    first_occ = first_all[first_all.MONTH.isin(season_months)].copy()
    last_occ = last_all[first_all.MONTH.isin(season_months)].copy()

    first_occ[['CLON', 'CLAT']] = first_occ.apply(
        get_area_weighted_center, axis=1, args=(masks_full, mask_var))
    last_occ[['CLON', 'CLAT']] = last_occ.apply(
        get_area_weighted_center, axis=1, args=(masks_full, mask_var))

    first_occ = first_occ.rename(columns={
        'YEAR': 'YEAR_START', 'MONTH': 'MONTH_START', 'DAY': 'DAY_START',
        'CLON': 'CLON_START', 'CLAT': 'CLAT_START'})
    last_occ = last_occ.rename(columns={
        'YEAR': 'YEAR_END', 'MONTH': 'MONTH_END', 'DAY': 'DAY_END',
        'CLON': 'CLON_END', 'CLAT': 'CLAT_END'})

    return first_occ, last_occ

print("Computing winter first/last...")
first_winter, last_winter = compute_first_last([5,6,7,8,9])
print("Computing summer first/last...")
first_summer, last_summer = compute_first_last([11,12,1,2,3])


#%% 5. Helper to draw one panel (map + histogram)
def draw_season_panel(fig, map_pos, hist_pos, LLB_clim, HLB_clim, ndays_LLB, ndays_HLB,
                      first_occ, last_occ, panel_label, season_title):
    """
    Draw a single season panel: map at map_pos, histogram at hist_pos.
    """
    cmap_LLB = 'OrRd'
    cmap_HLB = 'PuRd'
    vmin_LLB=0; vmax_LLB=15; step_LLB=1.5
    vmin_HLB=0; vmax_HLB=30; step_HLB=3

    # --- Map ---
    ax, gl = make_map_on_ax(fig, map_pos, -179.9, 179, 0, -85)
    gl.right_labels = False

    to_plot1 = LLB_clim / ndays_LLB * 100
    to_plot1[to_plot1 < 1] = np.nan
    to_plot2 = HLB_clim / ndays_HLB * 100
    to_plot2[to_plot2 < 1] = np.nan

    im1 = ax.contourf(lon, lat, to_plot1, levels=np.arange(vmin_LLB, vmax_LLB+step_LLB, step_LLB),
                       cmap=cmap_LLB, alpha=0.8, transform=ccrs.PlateCarree(), zorder=5, extend='max')
    ax.contour(lon, lat, to_plot1, levels=np.arange(1.5, 15+1.5, 1.5),
               alpha=0.8, transform=ccrs.PlateCarree(), zorder=7, colors='k', linewidths=0.7)
    im2 = ax.contourf(lon, lat, to_plot2, levels=np.arange(vmin_HLB, vmax_HLB+step_HLB, step_HLB),
                       cmap=cmap_HLB, alpha=0.8, transform=ccrs.PlateCarree(), zorder=1, extend='max')
    ax.contour(lon, lat, to_plot2, levels=np.arange(3, 30+3, 3),
               alpha=0.8, transform=ccrs.PlateCarree(), zorder=2, colors='k', linewidths=0.7)

    # Panel label
    ax.text(0.02, 0.8, f'{panel_label}) {season_title}', transform=ax.transAxes,
            fontsize=14, va='top', ha='left',
            bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.9), zorder=10)

    # --- Transition markers ---
    # LLB → HLB
    first_lon_LLB = first_occ[first_occ.TYPE.isin(types_llb) & ~last_occ.TYPE.isin(types_llb)]['CLON_START'].values
    first_lat_LLB = first_occ[first_occ.TYPE.isin(types_llb) & ~last_occ.TYPE.isin(types_llb)]['CLAT_START'].values
    last_lon_HLB  = last_occ[first_occ.TYPE.isin(types_llb) & ~last_occ.TYPE.isin(types_llb)]['CLON_END'].values
    last_lat_HLB  = last_occ[first_occ.TYPE.isin(types_llb) & ~last_occ.TYPE.isin(types_llb)]['CLAT_END'].values
    ax.scatter(first_lon_LLB, first_lat_LLB, transform=ccrs.PlateCarree(), s=15, marker='o', zorder=100, color='black', alpha=0.6)
    ax.scatter(last_lon_HLB, last_lat_HLB, transform=ccrs.PlateCarree(), s=30, marker='+', zorder=100, color='black', alpha=0.7)

    # HLB → LLB
    first_lon_HLB = first_occ[~first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLON_START'].values
    first_lat_HLB = first_occ[~first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLAT_START'].values
    last_lon_LLB  = last_occ[~first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLON_END'].values
    last_lat_LLB  = last_occ[~first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLAT_END'].values
    ax.scatter(first_lon_HLB, first_lat_HLB, transform=ccrs.PlateCarree(), s=15, marker='^', zorder=100, color='black', alpha=0.6)
    ax.scatter(last_lon_LLB, last_lat_LLB, transform=ccrs.PlateCarree(), s=15, marker='v', zorder=100, color='black', alpha=0.7)

    # --- Event statistics ---
    total_events = len(first_occ)
    n_llb_llb = np.sum(first_occ.TYPE.isin(types_llb).values & last_occ.TYPE.isin(types_llb).values)
    n_hlb_hlb = np.sum(~first_occ.TYPE.isin(types_llb).values & ~last_occ.TYPE.isin(types_llb).values)
    n_llb_hlb = np.sum(first_occ.TYPE.isin(types_llb).values & ~last_occ.TYPE.isin(types_llb).values)
    n_hlb_llb = np.sum(~first_occ.TYPE.isin(types_llb).values & last_occ.TYPE.isin(types_llb).values)
    pct_llb_llb = n_llb_llb / total_events * 100
    pct_hlb_hlb = n_hlb_hlb / total_events * 100
    pct_llb_hlb = n_llb_hlb / total_events * 100
    pct_hlb_llb = n_hlb_llb / total_events * 100

    # --- Legend ---
    legend_handles = [
        Line2D([0], [0], marker='s', color='none', markerfacecolor='tomato',
               markeredgecolor='k', markersize=10, label=f' LLB → LLB ({pct_llb_llb:.1f}%)'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='mediumorchid',
               markeredgecolor='k', markersize=10, label=f' HLB → HLB ({pct_hlb_hlb:.1f}%)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
               markersize=0, label=f'(● → +) LLB → HLB ({pct_llb_hlb:.1f}%)'),
        Line2D([0], [0], marker='^', color='none', markerfacecolor='black',
               markersize=0, label=f'(▲ → ▼) HLB → LLB ({pct_hlb_llb:.1f}%)'),
    ]
    ax.legend(handles=legend_handles, ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, 1.0), fontsize=11, frameon=True,
              facecolor='white', edgecolor='grey', framealpha=0.95,
              handletextpad=0.3, columnspacing=0.8)

    # --- Histogram ---
    lon_first = first_occ[first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLON_START'].values
    lon_last  = last_occ[first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLON_END'].values
    lon_first = lon_first[~np.isnan(lon_first)]
    lon_last  = lon_last[~np.isnan(lon_last)]

    ax_hist = fig.add_axes(hist_pos)
    bins = np.arange(-180, 180+15, 15)

    ax_hist.hist(lon_first, bins=bins, histtype='step', linewidth=2.0,
                 color='tab:blue', label='First obs.')
    ax_hist.hist(lon_last, bins=bins, histtype='step', linewidth=2.0,
                 color='tab:red', label='Last obs.')

    p25, median, p75 = circular_percentiles(lon_first)
    fc_blue = mcolors.to_rgba('tab:blue', alpha=0.2)
    ec_black = mcolors.to_rgba('black', alpha=1)
    ax_hist.axvline(median, ymin=0, ymax=0.3, color='tab:blue', linestyle='--')
    ax_hist.axvspan(p25, 180, ymin=0, ymax=0.3, fc=fc_blue, ec=ec_black)
    ax_hist.axvspan(-180, p75, ymin=0, ymax=0.3, fc=fc_blue, ec=ec_black)

    p25, median, p75 = circular_percentiles(lon_last)
    fc_red = mcolors.to_rgba('tab:red', alpha=0.2)
    ax_hist.axvline(median, ymin=0.3, ymax=0.6, color='tab:red', linestyle='--')
    ax_hist.axvspan(p25, 180, ymin=0.3, ymax=0.6, fc=fc_red, ec=ec_black)
    ax_hist.axvspan(-180, p75, ymin=0.3, ymax=0.6, fc=fc_red, ec=ec_black)

    ax_hist.set_xlim(-180, 180)
    ax_hist.set_ylabel('Count', fontsize=10)
    ax_hist.set_xticks([])
    ax_hist.tick_params(axis='y', labelsize=9)
    ax_hist.set_yticks([0, 50, 100])
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.legend(fontsize=10, loc='center left', bbox_to_anchor=(0.52, 0.7), frameon=True)

    return ax, im1, im2


#%% 6. Draw figure
plt.close('all')
fig = plt.figure(figsize=(14, 14))

# Panel positions: [x0, y0, width, height]
# Top panel (winter): map + histogram
winter_map_pos  = [0.08, 0.42, 0.85, 0.32]
winter_hist_pos = [0.08, 0.68, 0.85, 0.08]

# Bottom panel (summer): map + histogram
summer_map_pos  = [0.08, 0.10, 0.85, 0.32]
summer_hist_pos = [0.08, 0.361, 0.85, 0.08]

# Draw winter panel
ax_w, im1_w, im2_w = draw_season_panel(
    fig, winter_map_pos, winter_hist_pos,
    winter_LLB, winter_HLB, ndays_winter1, ndays_winter2,
    first_winter, last_winter,
    panel_label='a', season_title='Extended winter (MJJAS)')

# Draw summer panel
ax_s, im1_s, im2_s = draw_season_panel(
    fig, summer_map_pos, summer_hist_pos,
    summer_LLB, summer_HLB, ndays_summer1, ndays_summer2,
    first_summer, last_summer,
    panel_label='b', season_title='Extended summer (NDJFM)')

# --- Shared colorbars at the very bottom ---
cax1 = fig.add_axes([0.08, 0.12, 0.40, 0.015])
cbar1 = fig.colorbar(im1_s, cax=cax1, ticks=np.arange(0, 15+1.5, 1.5),
                      orientation='horizontal', extend='max')
cbar1.set_label(r'% of days in season ($\mathbf{LLB}$)', fontsize=11, labelpad=5)
cbar1.ax.tick_params(labelsize=10)

cax2 = fig.add_axes([0.53, 0.12, 0.40, 0.015])
cbar2 = fig.colorbar(im2_s, cax=cax2, ticks=np.arange(0, 30+3, 3),
                      orientation='horizontal', extend='max')
cbar2.set_label(r'% of days in season ($\mathbf{HLB}$)', fontsize=11, labelpad=5)
cbar2.ax.tick_params(labelsize=10)


#%% 7. Save
plt.savefig('../Figures/FigS1_Climatology_seasonal.jpg', dpi=600, bbox_inches='tight')