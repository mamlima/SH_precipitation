#%% 0. Init
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm


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

    return np.around((a+b)*h/2,1)

#%% 2. Open data
#%%% 2.1. Precipitation data
precip_data = xr.open_dataset('../../Data/Input_data/Precip_Oce_1970_2025_ERA5Land.nc')
precip_data = precip_data.sel(time=slice('1970-01-01', '2025-12-31'))
prec = precip_data.tp.values * 1000

#%%% 2.2. Blocking data
daily_data = pd.read_csv('../../Data/Output_data/03-Blocking_daily_catalogue_1940_2025_SH.csv')
daily_masks = xr.open_dataset('../../Data/Output_data/03-CatalogueMasks_1940_2025_SH.nc')

data_full = daily_data[daily_data.YEAR.isin([i for i in range(1970, 2025+1)])]
masks_full = daily_masks.sel(time=slice('1970-01-01', '2025-12-31'))

#%%% 2.3. Get coordinates and area
lat_prec = precip_data.latitude.values
lon_prec = precip_data.longitude.values
time_prec = precip_data.time.values

lat_llb = daily_masks.lat.values
lon_llb = daily_masks.lon.values

time_llb = pd.DatetimeIndex(masks_full.time)
times_to_sample = np.where(masks_full.time.dt.month.isin([5,6,7,8,9]))[0]

area = area_matrix(lon_llb, lat_llb, 1)


#%% 3. Choose interest boxes
#### For subtropical ridges
latmin_r, latmax_r = -55, -30
lons_r = np.arange(-180,180,10)
lonbounds_r =  {}
for i in range(len(lons_r)):
    lonbounds_r[f'lonmin_{i+1}'] = lons_r[i]
    lonbounds_r[f'lonmax_{i+1}'] = lons_r[i]+30


#%% 5. Get ridge, blocked, and normal days
#%%% 5.1. Function
def get_day_type(months_to_eval, masks, masks_data, lonmin, lonmax, latmin, latmax, time_arr, allowed_types):
    data_filt = masks_data[masks_data.TYPE.isin(allowed_types) & data_full.MONTH.isin(months_to_eval)]
    dates_to_an = pd.to_datetime(data_filt[['YEAR', 'MONTH', 'DAY']]).values
    strct_days = []

    for time_to_an in np.unique(dates_to_an):
        flag = False
        mask_day = masks.sel(time=time_to_an)
        
        if lonmax >= 180:
            # Convert longitude from [-180,180] → [0,360]
            mask_day = mask_day.assign_coords(
                lon=(mask_day.lon % 360)
            )
            # Important: sort so slicing works correctly
            mask_day = mask_day.sortby('lon')

        mask_filt = mask_day.sel(lon=slice(lonmin,lonmax), lat=slice(latmax,latmin)).Structs
        SIDs_in_filt = np.unique(mask_filt)[np.unique(mask_filt) != 0]
        
        if len(SIDs_in_filt) != 0:
            for strct_id in SIDs_in_filt:
                
                strct_mask_area = np.sum((mask_day.Structs == strct_id) * area)
                filt_mask_area = np.sum(((mask_day.Structs == strct_id) * area).sel(lon=slice(lonmin,lonmax), lat=slice(latmax,latmin)))
                
                perc_filt = filt_mask_area/strct_mask_area
                
                if perc_filt > 0.5 and filt_mask_area >= 500000:
                    flag = True
                else:
                    continue
        
        if flag:
            strct_days.append(np.where(time_arr == time_to_an)[0][0])
    
    return np.array(strct_days)

#%%% 5.2. Compute days of Ridges
which_regions = range(len(lons_r))  # All regions
ridge_results = {}

for i in tqdm(which_regions):
    ridge_results[f'strcts_R{i+1}'] = get_day_type([5,6,7,8,9], masks_full, data_full,
                                                   lonbounds_r[f'lonmin_{i+1}'], lonbounds_r[f'lonmax_{i+1}'], latmin_r, latmax_r,
                                                   time_llb, ['Ridge'])

#%% 6. Analysis
results_prec_llb = {}
precclima = []
precllb = []
precpval = []
precext = []
precllbext = []
precpvalext = []

for reg in tqdm(which_regions):
    rng = np.random.default_rng(seed=42)
    prec_llb = np.zeros(np.shape(prec)[1:])
    pvals_llb = np.zeros(np.shape(prec)[1:])
    prec_llb_ext = np.zeros(np.shape(prec)[1:])
    pvals_llb_ext = np.zeros(np.shape(prec)[1:])
    
    rand_maps = []
    rand_maps_ext = []
    
    for k in tqdm(range(1000)):
        random_i = rng.choice(times_to_sample, size=len(ridge_results[f'strcts_R{reg+1}']), replace=True)
        rand_maps.append(np.mean(prec[random_i],axis=0))
        rand_maps_ext.append(np.percentile(prec[random_i],q=95,axis=0))
    
    rand_maps = np.array(rand_maps)
    boot_clim = np.mean(rand_maps,axis=0)
    
    rand_maps_ext = np.array(rand_maps_ext)
    boot_ext = np.median(rand_maps_ext,axis=0)
    
    for i in range(np.shape(prec)[1]):
        for j in range(np.shape(prec)[2]):
            temp_point = prec[:,i,j]
            temp_val = np.mean(temp_point[ridge_results[f'strcts_R{reg+1}']])
            temp_val_ext = np.percentile(temp_point[ridge_results[f'strcts_R{reg+1}']], q=95)
            
            #### Get bootstrap results for mean vals
            rand_vals = rand_maps[:,i,j]
            prec_llb[i,j] = temp_val
            pvals_llb[i,j] = 1-np.mean(rand_vals <= temp_val)
            
            #### Get bootstrap results for extremes
            rand_vals_ext = rand_maps_ext[:,i,j]
            prec_llb_ext[i,j] = temp_val_ext
            pvals_llb_ext[i,j] = 1-np.mean(rand_vals_ext <= temp_val_ext)
    
    precclima.append( boot_clim )
    precllb.append( prec_llb )
    precpval.append( pvals_llb )
    
    precext.append( boot_ext )
    precllbext.append( prec_llb_ext )
    precpvalext.append( pvals_llb_ext )


#%% 7. Save data
data = xr.Dataset(
        data_vars = dict(precip_clima=(['reg', 'lat', 'lon'], np.array(precclima)),
                         precip_llb=(['reg', 'lat', 'lon'], np.array(precllb)),
                         pval_llb=(['reg', 'lat', 'lon'], np.array(precpval)),
                         precip_ext=(['reg', 'lat', 'lon'], np.array(precext)),
                         precip_llb_ext=(['reg', 'lat', 'lon'], np.array(precllbext)),
                         pval_llb_ext=(['reg', 'lat', 'lon'], np.array(precpvalext)),
                         ),
        coords = dict(reg=lons_r, lat=lat_prec, lon=lon_prec),
        attrs = dict(description='Bootstrapped composites of precip per analysis box for Oceania')
        )

data.to_netcdf('../../Data/Output_data/RidgePrecipImpact_Oce.nc')
