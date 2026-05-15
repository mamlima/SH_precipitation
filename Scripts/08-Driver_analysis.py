#%% 0. Init
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm

which_season = 'winter'

#%% 1. Functions
#%%% 1.1. Clean nan values
def clean_nans(arr):
    return arr[~np.isnan(arr)]

#%%% 1.2. Geographical distance function
def dist(lat1, lat2, lon1, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

#%%% 1.3. Compute area matrix
def area_matrix(lon, lat, res):
    lat_2d = np.array([lat for i in lon]).T
    lon_2d = np.array([lon for i in lat])

    a = dist(lat_2d+res/2, lat_2d+res/2, lon_2d-res/2, lon_2d+res/2)
    b = dist(lat_2d-res/2, lat_2d-res/2, lon_2d-res/2, lon_2d+res/2)
    h = dist(lat_2d+res/2, lat_2d-res/2, lon_2d, lon_2d)

    return np.around((a+b)*h/2, 1)


#%% 2. Open data
#%%% 2.1. Driver datasets (ERA5 at 1° resolution, full SH)
driver_files = {
    'z500': ('../Data/Input_data/Z500_1940_2025_SH_ERA5.nc',   'z'),
    'mslp': ('../Data/Input_data/MSLP_1940_2025_SH_ERA5.nc',   'msl'),
    'ivt':  ('../Data/Input_data/IVT_1970_2025_SH_ERA5.nc',    'ivt'),
    'olr':  ('../Data/Input_data/OLR_1970_2025_SH_ERA5.nc',    'olr'),
}

driver_datasets = {}
driver_arrays = {}

for key, (fpath, varname) in driver_files.items():
    ds = xr.open_dataset(fpath)
    ds = ds.sel(time=slice('1970-01-01', '2025-12-31'))
    driver_datasets[key] = ds
    driver_arrays[key] = ds[varname].values

# Use coordinates from the first dataset (all share the same grid)
ref_ds = list(driver_datasets.values())[0]
lat_drv = ref_ds.latitude.values
lon_drv = ref_ds.longitude.values
time_drv = ref_ds.time.values

#%%% 2.2. Blocking data
daily_data = pd.read_csv('../Data/Blocks_data/03-Blocking_daily_catalogue_1970_2025_SH.csv')
daily_masks = xr.open_dataset('../Data/Blocks_data/03-CatalogueMasks_1970_2025_SH.nc')

data_full = daily_data[daily_data.YEAR.isin([i for i in range(1970, 2025+1)])]
masks_full = daily_masks.sel(time=slice('1970-01-01', '2025-12-31'))

#%%% 2.3. Get blocking coordinates and area
lat_llb = daily_masks.lat.values
lon_llb = daily_masks.lon.values

time_llb = pd.DatetimeIndex(masks_full.time)
if which_season == 'winter':
    times_to_sample = np.where(masks_full.time.dt.month.isin([5, 6, 7, 8, 9]))[0]
elif which_season == 'summer':
    times_to_sample = np.where(masks_full.time.dt.month.isin([11, 12, 1, 2, 3]))[0]


area = area_matrix(lon_llb, lat_llb, 1)


#%% 3. Define sectors per region (matching script 07)
latmin_r, latmax_r = -55, -30

manual_sectors = {
    'SAm': [-25, -55, -85],
    'SAf': [15, 75],
    'Oce': [85, 115, 165, -155],
}

sector_labels_per_region = {
    'SAm': ['East', 'Over', 'West'],
    'SAf': ['West', 'East'],
    'Oce': ['West', 'Over West', 'Over East', 'East'],
}

# Convert sector centre longitudes to lon bounds (30° wide sectors)
def sector_bounds(centre_lon):
    """Return (lonmin, lonmax) for a 30°-wide sector centred at centre_lon."""
    lonmin = centre_lon - 15
    lonmax = centre_lon + 15
    return lonmin, lonmax


#%% 4. Ridge day identification function (from script 05)
def get_day_type(months_to_eval, masks, masks_data, lonmin, lonmax, latmin, latmax, time_arr, allowed_types):
    data_filt = masks_data[masks_data.TYPE.isin(allowed_types) & masks_data.MONTH.isin(months_to_eval)]
    dates_to_an = pd.to_datetime(data_filt[['YEAR', 'MONTH', 'DAY']]).values
    strct_days = []

    for time_to_an in np.unique(dates_to_an):
        flag = False
        mask_day = masks.sel(time=time_to_an)

        if lonmax >= 180:
            mask_day = mask_day.assign_coords(
                lon=(mask_day.lon % 360)
            )
            mask_day = mask_day.sortby('lon')

        mask_filt = mask_day.sel(lon=slice(lonmin, lonmax), lat=slice(latmax, latmin)).Structs
        SIDs_in_filt = np.unique(mask_filt)[np.unique(mask_filt) != 0]

        if len(SIDs_in_filt) != 0:
            for strct_id in SIDs_in_filt:

                strct_mask_area = np.sum((mask_day.Structs == strct_id) * area)
                filt_mask_area = np.sum(((mask_day.Structs == strct_id) * area).sel(lon=slice(lonmin, lonmax), lat=slice(latmax, latmin)))

                perc_filt = filt_mask_area / strct_mask_area

                if perc_filt > 0.5 and filt_mask_area >= 500000:
                    flag = True
                else:
                    continue

        if flag:
            strct_days.append(np.where(time_arr == time_to_an)[0][0])

    return np.array(strct_days)


#%% 5. Identify ridge days for selected sectors only
ridge_results = {}

for region, centres in tqdm(manual_sectors.items(), desc='Identifying ridge days'):
    for j, centre_lon in enumerate(centres):
        lonmin, lonmax = sector_bounds(centre_lon)
        key = f'{region}_sec{j}'
        ridge_results[key] = get_day_type(
            [5, 6, 7, 8, 9], masks_full, data_full,
            lonmin, lonmax,
            latmin_r, latmax_r,
            time_llb, ['Ridge']
        )
        

#%% 6. Compute MJJAS climatology indices (shared across regions)
if which_season == 'winter':
    mjjas_mask = pd.DatetimeIndex(time_drv).month.isin([5, 6, 7, 8, 9])
    mjjas_idx = np.where(mjjas_mask)[0]
elif which_season == 'summer':
    mjjas_mask = pd.DatetimeIndex(time_drv).month.isin([11,12,1,2,3])
    mjjas_idx = np.where(mjjas_mask)[0]


#%% 7. Compute composites and save per region
for region, centres in tqdm(manual_sectors.items(), desc='Computing composites'):
    n_sectors = len(centres)
    labels = sector_labels_per_region[region]

    # Prepare output arrays: (n_sectors, lat, lon)
    shape_out = (n_sectors,) + driver_arrays['z500'].shape[1:]

    output_vars = {}
    for dkey in driver_files.keys():
        output_vars[f'{dkey}_clima'] = np.full(shape_out, np.nan)
        output_vars[f'{dkey}_llb']   = np.full(shape_out, np.nan)

    for j, centre_lon in enumerate(centres):
        key = f'{region}_sec{j}'
        ridge_idx = ridge_results[key]

        if len(ridge_idx) == 0:
            continue

        for dkey in driver_files.keys():
            drv = driver_arrays[dkey]

            # MJJAS climatology
            output_vars[f'{dkey}_clima'][j] = np.mean(drv[mjjas_idx], axis=0)

            # LLB ridge-day composite
            output_vars[f'{dkey}_llb'][j] = np.mean(drv[ridge_idx], axis=0)

    # ── Build and save NetCDF ──
    sector_centres = np.array(centres, dtype=float)

    data_vars = {}
    for dkey in driver_files.keys():
        data_vars[f'{dkey}_clima'] = (['reg', 'lat', 'lon'], output_vars[f'{dkey}_clima'])
        data_vars[f'{dkey}_llb']   = (['reg', 'lat', 'lon'], output_vars[f'{dkey}_llb'])

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            reg=sector_centres,
            lat=lat_drv,
            lon=lon_drv,
        ),
        attrs=dict(
            description=f'Driver composites (Z500, MSLP, IVT, OLR) for LLB ridge days — {region}',
            sector_labels=str(labels),
            sector_centres=str(centres),
        )
    )

    outpath = f'../Data/Output_data/DriverImpact_{region}_{which_season}.nc'
    ds_out.to_netcdf(outpath)