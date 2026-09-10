#%% 0. Start
import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker


#%% 1. Configuration
DATA_DIR  = '../Data/Input_data'
CACHE_DIR = '../Data/Output_data'
FIG_DIR   = '../Figures'

FILE_PATTERN = 'Precip_{region}_1970_2025_ERA5Land.nc'
REGIONS = {'SAm': 'South America',
           'SAf': 'Southern Africa',
           'Oce': 'Oceania'}

SEASONS = {'MJJAS': [5, 6, 7, 8, 9],
           'NDJFM': [11, 12, 1, 2, 3]}

YEARS = (1970, 2025)

# Manuscript mask thresholds (mm/day)
THRESH = {'mean': {'MJJAS': 0.5, 'NDJFM': 1.5},
          'p95':  {'MJJAS': 3.0, 'NDJFM': 9.0}}

TOP_ROW_FIELD = 'mean'

RECOMPUTE = False          
UNITS_SCALE = None         

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


#%% 2. Helpers
def _coord_names(ds):
    lat = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon = 'longitude' if 'longitude' in ds.coords else 'lon'
    return lat, lon


def _precip_var(ds):
    for cand in ['tp', 'precip', 'precipitation', 'total_precipitation', 'pr']:
        if cand in ds.data_vars:
            return cand
    return list(ds.data_vars)[0]


def _to_mm_per_day(da):
    """ERA5-Land accumulations are in metres; daily fields in mm are O(1-100)."""
    if UNITS_SCALE is not None:
        return da * UNITS_SCALE
    units = str(da.attrs.get('units', '')).lower()
    if units in ('m', 'metres', 'meter', 'meters'):
        print('   units attribute says metres -> scaling by 1000')
        return da * 1000.0
    sample = float(np.nanmax(da.isel(time=slice(0, 200)).values))
    if sample < 1.0:
        print(f'   max of first 200 days = {sample:.4f} -> assuming metres, scaling by 1000')
        return da * 1000.0
    print(f'   max of first 200 days = {sample:.2f} -> assuming mm/day, no scaling')
    return da


def seasonal_climatology(region):
    """Return a Dataset with mean, p50 and p95 per season for one region."""
    cache = os.path.join(CACHE_DIR, f'PrecipClim_{region}_{YEARS[0]}_{YEARS[1]}.nc')
    if os.path.exists(cache) and not RECOMPUTE:
        print(f'{region}: loading cached climatology')
        return xr.open_dataset(cache)

    path = os.path.join(DATA_DIR, FILE_PATTERN.format(region=region))
    print(f'{region}: reading {path}')
    # chunk along latitude, keep the whole time axis in each chunk (needed for quantiles)
    ds = xr.open_dataset(path, chunks={'time': -1, 'latitude': 40, 'lat': 40})
    latn, lonn = _coord_names(ds)
    da = ds[_precip_var(ds)]
    da = da.sel(time=slice(f'{YEARS[0]}-01-01', f'{YEARS[1]}-12-31'))
    da = _to_mm_per_day(da)

    out = {}
    for sname, months in SEASONS.items():
        sub = da.sel(time=da['time.month'].isin(months))
        print(f'   {sname}: {sub.sizes["time"]} days -> mean, p50, p95')
        out[f'mean_{sname}'] = sub.mean('time', skipna=True)
        q = sub.quantile([0.50, 0.95], dim='time', skipna=True)
        out[f'p50_{sname}'] = q.sel(quantile=0.50, drop=True)
        out[f'p95_{sname}'] = q.sel(quantile=0.95, drop=True)

    clim = xr.Dataset(out).compute()
    clim = clim.rename({latn: 'lat', lonn: 'lon'}) if latn != 'lat' else clim
    clim.to_netcdf(cache)
    print(f'{region}: cached to {cache}')
    return clim


clims = {r: seasonal_climatology(r) for r in REGIONS}


#%% 3. Diagnostics for the reply
rows = []
for r, name in REGIONS.items():
    c = clims[r]
    for field, key in [('mean', TOP_ROW_FIELD if TOP_ROW_FIELD != 'p50' else 'p50'),
                       ('p95', 'p95')]:
        for s in SEASONS:
            # the field the threshold is applied to
            var = f'{"mean" if field == "mean" else "p95"}_{s}'
            arr = c[var].values.ravel()
            arr = arr[np.isfinite(arr)]
            thr = THRESH[field][s]
            if arr.size == 0:
                continue
            pct_of_dist = (arr < thr).mean() * 100          # where the threshold sits
            n_total = arr.size
            n_kept = int((arr > thr).sum())
            rows.append({'region': name, 'field': field, 'season': s,
                         'threshold_mm_day': thr,
                         'pctile_of_land_dist': pct_of_dist,
                         'n_land_points': n_total,
                         'n_kept': n_kept,
                         'pct_kept': n_kept / n_total * 100})

diag = pd.DataFrame(rows)
pd.set_option('display.width', 160)
print(diag.round(1).to_string(index=False))

for r, name in REGIONS.items():
    for field in ['mean', 'p95']:
        d = diag[(diag.region == name) & (diag.field == field)]
        if len(d) == 2:
            w = d[d.season == 'MJJAS'].pctile_of_land_dist.iloc[0]
            s = d[d.season == 'NDJFM'].pctile_of_land_dist.iloc[0]
            print(f'{name:16s} {field:4s}: threshold excludes {w:5.1f}% of land in MJJAS '
                  f'vs {s:5.1f}% in NDJFM   (difference {abs(w - s):4.1f} pp)')
diag.to_csv(os.path.join(CACHE_DIR, 'R2_3_mask_diagnostics.csv'), index=False)


#%% 4. Figure 
plt.close('all')
fig = plt.figure(figsize=(15, 6))

col_spec = [
    (TOP_ROW_FIELD, 'mean', np.arange(0, 12.1, 1.0), 'YlGnBu',
     f'{"Mean" if TOP_ROW_FIELD == "mean" else "median"} precipitation'),
    ('p95', 'p95', np.arange(0, 40.1, 2.5), 'YlGnBu',
     '95th percentile'),
]

panel_letters = ['a', 'b', 'c', 'd']
axes = []
col_images = {}
k = 0
for i, s_name in enumerate(SEASONS):                       # rows = seasons
    for j, (field_key, thr_key, levels, cmap, col_title) in enumerate(col_spec):   # cols = fields
        ax = fig.add_subplot(2, 2, k + 1, projection=ccrs.PlateCarree())
        ax.set_extent([-100, 180, -60, 0], ccrs.PlateCarree())
        ax.add_feature(cf.LAND, facecolor='whitesmoke', zorder=0)
        ax.coastlines(linewidth=0.5, color='grey', zorder=4)

        gl = ax.gridlines(draw_labels=True, ls='--', color='grey', alpha=0.4, zorder=1)
        gl.top_labels = gl.right_labels = False
        gl.left_labels = (j == 0)
        gl.bottom_labels = (i == 1)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
        gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0])
        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()
        gl.xlabel_style = gl.ylabel_style = {'size': 8}

        thr = THRESH[thr_key][s_name]
        kept_txt = []
        for r in REGIONS:
            da = clims[r][f'{field_key}_{s_name}']
            im = ax.contourf(da.lon, da.lat, da.values, levels=levels, cmap=cmap,
                             extend='max', transform=ccrs.PlateCarree(), zorder=2)

            md = clims[r][f'{"mean" if thr_key == "mean" else "p95"}_{s_name}']

            excluded = np.where(np.isfinite(md.values) & (md.values <= thr), 1.0, np.nan)
            hs = ax.contourf(md.lon, md.lat, excluded, levels=[0.5, 1.5],
                             colors='none', hatches=['///'],
                             transform=ccrs.PlateCarree(), zorder=3)
            try:                       # matplotlib >= 3.8
                hs.set_edgecolor('0.35')
                hs.set_linewidth(0.0)
            except AttributeError:     # older matplotlib
                for coll in hs.collections:
                    coll.set_edgecolor('0.35')
                    coll.set_linewidth(0.0)

            # mask boundary
            ax.contour(md.lon, md.lat, md.values, levels=[thr], colors='crimson',
                       linewidths=1.4, transform=ccrs.PlateCarree(), zorder=5)

            d = diag[(diag.region == REGIONS[r]) & (diag.field == thr_key) &
                     (diag.season == s_name)]
            if len(d):
                kept_txt.append(f'{r} {d.pct_kept.iloc[0]:.0f}%')
        col_images[j] = im

        ax.text(0.637, 0.06, 'retained: ' + '  |  '.join(kept_txt),
                transform=ax.transAxes, fontsize=8, va='bottom', ha='left',
                bbox=dict(boxstyle='square,pad=0.25', facecolor='white',
                          edgecolor='grey', alpha=0.85), zorder=6)

        ax.set_title(f'{panel_letters[k]}) {col_title} - {s_name}   '
                     f'(mask: {thr} mm day$^{{-1}}$)', fontsize=11, loc='left')
        axes.append(ax)
        k += 1

fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.16,
                    hspace=-0.2, wspace=0.04)

# one horizontal colourbar per column, underneath it
for j in range(2):
    x0 = 0.08 + j * 0.48
    cax = fig.add_axes([x0, 0.14, 0.38, 0.032])
    cb = fig.colorbar(col_images[j], cax=cax, orientation='horizontal', extend='max')
    cb.set_label(f'{col_spec[j][4]} (mm day$^{{-1}}$)', fontsize=12)
    cb.ax.tick_params(labelsize=10)

fig.savefig(os.path.join(FIG_DIR, 'FigS7_AridMasks.jpg'),
            dpi=600, bbox_inches='tight', facecolor='white')