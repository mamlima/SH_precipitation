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
#%%% 1.1. Clean nan values
def clean_nans(arr):
    return arr[~np.isnan(arr)]


#%% 2. Open data
#%%% 2.1. Open catalogue
daily_data = pd.read_csv('../Data/Blocks_data/03-Blocking_daily_catalogue_1970_2025_SH.csv')
daily_masks = xr.open_dataset('../Data/Blocks_data/03-CatalogueMasks_1970_2025_SH.nc')


#%% 3. Filter data for same time period
#%%% 3.1. Catalogues
data_full = daily_data[daily_data.YEAR.isin([i for i in range(1970, 2025+1)])]
masks_full = daily_masks.sel(time=slice('1970-01-01', '2025-12-31'))

#%%% 3.2. Lat, Lon, time and box definition
lat = daily_masks.lat.values
lon = daily_masks.lon.values
lons, lats = np.meshgrid(lon,lat)

time_full = pd.DatetimeIndex(masks_full.time)


#%% 4. Get CLAT and CLON of first and last occurrences
#%%% 4.1. Get first and last occurrences
extended_season = [1,2,3,4,5,6,7,8,9,10,11,12]

first_all = data_full.groupby('SID', as_index=False).first()
last_all = data_full.groupby('SID', as_index=False).last()

first_occ = first_all[first_all.MONTH.isin(extended_season)]
last_occ = last_all[first_all.MONTH.isin(extended_season)]

#%%% 4.2. Identify the mask variable name
mask_var = list(masks_full.data_vars)[0]

#%%% 4.3. Helper functions
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

    # Ensure output is in [-180, 180]
    if mean_lon > 180:
        mean_lon -= 360
    elif mean_lon <= -180:
        mean_lon += 360

    return mean_lon

def circular_percentiles(lon_deg, weights=None):

    lon = np.asarray(lon_deg, dtype=float)

    # --- Step 1: circular mean (your existing function)
    lon_rad = np.deg2rad(lon)

    if weights is None:
        weights = np.ones_like(lon)
    else:
        weights = np.asarray(weights)

    mean_sin = np.sum(weights * np.sin(lon_rad)) / np.sum(weights)
    mean_cos = np.sum(weights * np.cos(lon_rad)) / np.sum(weights)

    mean_angle = np.rad2deg(np.arctan2(mean_sin, mean_cos))

    # --- Step 2: shift data around mean → unwrap
    shifted = (lon - mean_angle + 180) % 360 - 180

    # --- Step 3: compute percentiles in linear space
    p25 = np.percentile(shifted, 25)
    p50 = np.percentile(shifted, 50)  # median
    p75 = np.percentile(shifted, 75)

    # --- Step 4: shift back
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

    # Select the mask on the given day
    mask_day = masks[mask_var].sel(time=date)

    # Boolean mask for this structure
    structure_mask = (mask_day == sid)

    # Get indices of grid cells belonging to this SID
    yy, xx = np.where(structure_mask.values)

    # If nothing found, return NaN
    if len(yy) == 0:
        return pd.Series({'CLON': np.nan, 'CLAT': np.nan})

    # Get lat/lon of those grid cells
    lat_pts = mask_day.lat.values[yy]
    lon_pts = mask_day.lon.values[xx]

    # Area weights for regular lat-lon grid
    weights = np.cos(np.deg2rad(lat_pts))

    # Area-weighted latitude
    clat = np.average(lat_pts, weights=weights)

    # Area-weighted circular longitude
    clon = circular_mean_lon_180(lon_pts, weights=weights)

    return pd.Series({'CLON': clon, 'CLAT': clat})

#%%% 4.4. Compute area-weighted centers for first and last occurrence
first_occ[['CLON', 'CLAT']] = first_occ.apply(
    get_area_weighted_center,
    axis=1,
    args=(masks_full, mask_var)
)

last_occ[['CLON', 'CLAT']] = last_occ.apply(
    get_area_weighted_center,
    axis=1,
    args=(masks_full, mask_var)
)

#%%% 4.5. Optional: rename columns to avoid confusion
first_occ = first_occ.rename(columns={
    'YEAR': 'YEAR_START',
    'MONTH': 'MONTH_START',
    'DAY': 'DAY_START',
    'CLON': 'CLON_START',
    'CLAT': 'CLAT_START'
})

last_occ = last_occ.rename(columns={
    'YEAR': 'YEAR_END',
    'MONTH': 'MONTH_END',
    'DAY': 'DAY_END',
    'CLON': 'CLON_END',
    'CLAT': 'CLAT_END'
})


#%% 5. Get Ridge climatology
#%%% 5.1. Function for climatology
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

#%%% 5.2. Compute climatology
winter_LLB_full, ndays_winter_full1 = give_climatology(extended_season, masks_full.Structs.values, data_full, time_full, ['Ridge', 'Omega block', 'Rex block (hybrid)'])
winter_HLB_full, ndays_winter_full2 = give_climatology(extended_season, masks_full.Structs.values, data_full, time_full, ['Rex block', 'Rex block (polar)'])

#%%% 5.3. Function to draw map
def make_map(line, cols, which, lon_min, lon_max, lat_min, lat_max):
    ax = fig.add_subplot(line,cols,which,projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, ls='--', colors='grey', alpha=0.6)
    gl.xlabel_style = {'size': 9}  # Longitude label size
    gl.ylabel_style = {'size': 9}  # Latitude label size
    
    land = cf.NaturalEarthFeature(category='physical',
                                        scale='50m',
                                        facecolor='gainsboro', name='land')
    ax.add_feature(land, edgecolor='k', linewidth=0.7, alpha=0.9, zorder=0)
    borders = cf.NaturalEarthFeature(category='cultural',
                                           name='admin_0_boundary_lines_land',
                                           scale='50m',
                                           facecolor='none', edgecolor='black')
    ax.add_feature(borders, edgecolor='k', linewidth=0.6, alpha=0.9, zorder=0)
    
    gl.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
    gl.xformatter = cticker.LongitudeFormatter()

    gl.ylocator = mticker.FixedLocator([-80, -60, -40, -20, 0])
    gl.yformatter = cticker.LatitudeFormatter()
    
    return ax, gl

#% 5.4. Draw climatologies full
plt.close('all')
fig = plt.figure(figsize=(14, 8))

cmap_LLB = 'OrRd'
cmap_HLB = 'PuRd'

##### Draw Winter Ridges
ax1, gl1 = make_map(1, 1, 1, -179.9, 179, 0, -85)
gl1.right_labels = False

vmin_LLB=0; vmax_LLB=15; step_LLB=1.5
vmin_HLB=0; vmax_HLB=30; step_HLB=3
to_plot1 = winter_LLB_full/ndays_winter_full1*100
to_plot1[to_plot1 < 1] = np.nan
to_plot2 = winter_HLB_full/ndays_winter_full2*100
to_plot2[to_plot2 < 1] = np.nan
im1 = ax1.contourf(lon, lat, to_plot1, levels = np.arange(vmin_LLB, vmax_LLB+step_LLB, step_LLB),
                  cmap = cmap_LLB, alpha = 0.8, transform=ccrs.PlateCarree(), zorder=5, extend = 'max')
ax1.contour(lon, lat, to_plot1, levels = np.arange(1.5,15+1.5,1.5),
            alpha = 0.8, transform=ccrs.PlateCarree(), zorder=7, colors='k', linewidths = 0.7)
im2 = ax1.contourf(lon, lat, to_plot2, levels = np.arange(vmin_HLB, vmax_HLB+step_HLB, step_HLB),
                  cmap = cmap_HLB, alpha = 0.8, transform=ccrs.PlateCarree(), zorder=1, extend = 'max')
ax1.contour(lon, lat, to_plot2, levels = np.arange(3,30+3,3),
            alpha = 0.8, transform=ccrs.PlateCarree(), zorder=2, colors='k', linewidths = 0.7)

# Add longitude histogram (normalized + step) above ax1
# --- Select same subset as in ax1 ---
types_llb = ['Ridge', 'Omega block', 'Rex block (hybrid)']

# --- Count events by transition type ---
total_events = len(first_occ)

# LLB → LLB: started as LLB and ended as LLB
n_llb_llb = np.sum(first_occ.TYPE.isin(types_llb).values & last_occ.TYPE.isin(types_llb).values)

# HLB → HLB: started as HLB and ended as HLB
n_hlb_hlb = np.sum(~first_occ.TYPE.isin(types_llb).values & ~last_occ.TYPE.isin(types_llb).values)

# LLB → HLB
n_llb_hlb = np.sum(first_occ.TYPE.isin(types_llb).values & ~last_occ.TYPE.isin(types_llb).values)

# HLB → LLB
n_hlb_llb = np.sum(~first_occ.TYPE.isin(types_llb).values & last_occ.TYPE.isin(types_llb).values)

pct_llb_llb = n_llb_llb / total_events * 100
pct_hlb_hlb = n_hlb_hlb / total_events * 100
pct_llb_hlb = n_llb_hlb / total_events * 100
pct_hlb_llb = n_hlb_llb / total_events * 100

lon_first = first_occ[first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLON_START'].values
lon_last  = last_occ[first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLON_END'].values

# Remove NaNs
lon_first = lon_first[~np.isnan(lon_first)]
lon_last  = lon_last[~np.isnan(lon_last)]

# --- Create axis above ax1 ---
pos = ax1.get_position()
ax_histx = fig.add_axes([pos.x0, pos.y1, pos.width, 0.12])

# --- Define bins (18 bins across [-180, 180]) ---
bins = np.arange(-180, 180+15, 15)

# --- Plot normalized step histograms ---
ax_histx.hist(lon_first, bins=bins,
              histtype='step', linewidth=2.0,
              color='tab:blue', label='First obs.')

ax_histx.hist(lon_last, bins=bins,
              histtype='step', linewidth=2.0,
              color='tab:red', label='Last obs.')

p25, median, p75 = circular_percentiles(lon_first)
facecolor = mcolors.to_rgba('tab:blue', alpha=0.2)
edgecolor = mcolors.to_rgba('black', alpha=1)
ax_histx.axvline(median, ymin=0, ymax=0.3, color='tab:blue', linestyle='--')
ax_histx.axvspan(p25, 180, ymin=0, ymax=0.3, fc=facecolor, ec = edgecolor)
ax_histx.axvspan(-180, p75, ymin=0, ymax=0.3, fc=facecolor, ec = edgecolor)

p25, median, p75 = circular_percentiles(lon_last)
facecolor = mcolors.to_rgba('tab:red', alpha=0.2)
edgecolor = mcolors.to_rgba('black', alpha=1)
ax_histx.axvline(median, ymin=0.3, ymax=0.6, color='tab:red', linestyle='--')
ax_histx.axvspan(p25, 180, ymin=0.3, ymax=0.6, fc=facecolor, ec = edgecolor)
ax_histx.axvspan(-180, p75, ymin=0.3, ymax=0.6, fc=facecolor, ec = edgecolor)

# --- Formatting (adjusted y-axis for full year counts) ---
ax_histx.set_xlim(-180, 180)
ax_histx.set_ylabel('Count', fontsize=12)
ax_histx.set_xticks([])
ax_histx.tick_params(axis='y', labelsize=10)
ax_histx.set_yticks([0, 100, 200])

ax_histx.spines['top'].set_visible(False)
ax_histx.spines['right'].set_visible(False)

ax_histx.legend(
    fontsize=12,
    loc='center left',
    bbox_to_anchor=(0.52, 0.7),
    frameon=True
)

#### LLB to HLB markers
first_lon_LLB = first_occ[first_occ.TYPE.isin(types_llb) & ~last_occ.TYPE.isin(types_llb)]['CLON_START'].values
first_lat_LLB = first_occ[first_occ.TYPE.isin(types_llb) & ~last_occ.TYPE.isin(types_llb)]['CLAT_START'].values

last_lon_HLB  = last_occ[first_occ.TYPE.isin(types_llb) & ~last_occ.TYPE.isin(types_llb)]['CLON_END'].values
last_lat_HLB  = last_occ[first_occ.TYPE.isin(types_llb) & ~last_occ.TYPE.isin(types_llb)]['CLAT_END'].values

ax1.scatter(first_lon_LLB, first_lat_LLB, transform=ccrs.PlateCarree(), s = 15, marker = 'o', zorder=100, color='black', alpha = 0.6)
ax1.scatter(last_lon_HLB, last_lat_HLB, transform=ccrs.PlateCarree(), s = 30, marker = '+', zorder=100, color='black', alpha = 0.7)

#### HLB to LLB markers
first_lon_HLB = first_occ[~first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLON_START'].values
first_lat_HLB = first_occ[~first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLAT_START'].values

last_lon_LLB  = last_occ[~first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLON_END'].values
last_lat_LLB  = last_occ[~first_occ.TYPE.isin(types_llb) & last_occ.TYPE.isin(types_llb)]['CLAT_END'].values

ax1.scatter(first_lon_HLB, first_lat_HLB, transform=ccrs.PlateCarree(), s = 15, marker = '^', zorder=100, color='black', alpha = 0.6)
ax1.scatter(last_lon_LLB, last_lat_LLB, transform=ccrs.PlateCarree(), s = 15, marker = 'v', zorder=100, color='black', alpha = 0.7)

################ Colorbar LLB
cax = fig.add_axes([0.125, 0.27, 0.37, 0.023])
cbar = fig.colorbar(im1, cax = cax, ticks = np.arange(0, 15+1.5, 1.5),
                    orientation='horizontal', extend = 'max', pad = 0.45, aspect = 25)
cbar.set_label(r'% of days in year ($\mathbf{LLB}$)', fontsize=12, labelpad=7)
cbar.ax.tick_params(labelsize=11)

################ Colorbar HLB
cax = fig.add_axes([0.53, 0.27, 0.37, 0.023])
cbar = fig.colorbar(im2, cax = cax, ticks = np.arange(0, 30+3, 3),
                    orientation='horizontal', extend = 'max', pad = 0.45, aspect = 25)
cbar.set_label('% of days in year ($\mathbf{HLB}$)', fontsize=12, labelpad=7)
cbar.ax.tick_params(labelsize=11)

#### Adjust plots
plt.subplots_adjust(hspace=-0.15)

# --- Custom legend with percentages ---
legend_handles = [
    Line2D([0], [0], marker='s', color='none', markerfacecolor='tomato',
           markeredgecolor='k', markersize=12, label=f' LLB → LLB ({pct_llb_llb:.1f}%)'),
    Line2D([0], [0], marker='s', color='none', markerfacecolor='mediumorchid',
           markeredgecolor='k', markersize=12, label=f' HLB → HLB ({pct_hlb_hlb:.1f}%)'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor='black',
           markersize=0, label=f'(● → +) LLB → HLB ({pct_llb_hlb:.1f}%)'),
    Line2D([0], [0], marker='^', color='none', markerfacecolor='black',
           markersize=0, label=f'(▲ → ▼) HLB → LLB ({pct_hlb_llb:.1f}%)'),
]

ax1.legend(
    handles=legend_handles,
    ncol=4,
    loc='upper center',      # anchors at upper center of the axes
    bbox_to_anchor=(0.5, 1.0),  # places it near the top (equator side, since SH is flipped)
    fontsize=11,
    frameon=True,
    facecolor='white',
    edgecolor='grey',
    framealpha=0.95,
    handletextpad=0.3,
    columnspacing=1.0,
)

#### Adjust and save
plt.savefig('../Figures/Fig1_Climatology.jpg', dpi=600, bbox_inches = 'tight')