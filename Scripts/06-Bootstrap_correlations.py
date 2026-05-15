#%% 0. Start
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
import pickle

#### Select parameters
regions_to_consider = 'all'
which_season = 'year'

var_list = ['Z500', 'MSLP', 'IVT', 'IWV', 'UV', 'OLR']
short_list = ['z', 'msl', 'ivt', 'tcwv', '-', 'olr']

for which_var, short in zip(var_list, short_list):

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
    #%%% 2.1. Open catalogue
    daily_data = pd.read_csv('../Data/Blocks_data/03-Blocking_daily_catalogue_1970_2025_SH.csv')
    daily_masks = xr.open_dataset('../Data/Blocks_data/03-CatalogueMasks_1970_2025_SH.nc')
    
    #%%% 2.2. Open original data
    if which_var in ['Z500', 'MSLP']:
        var_data = xr.open_dataset(f'../Data/Input_data/{which_var}_1940_2025_SH_ERA5.nc')
    if which_var in ['IVT', 'IWV', 'Precip', 'PV', 'OLR']:
        var_data = xr.open_dataset(f'../Data/Input_data/{which_var}_1970_2025_SH_ERA5.nc')
    if which_var == 'UV':
        var_data1 = xr.open_dataset('../Data/Input_data/IVT_1970_2025_SH_ERA5.nc')
        var_data2 = xr.open_dataset('../Data/Input_data/IWV_1970_2025_SH_ERA5.nc')
        
    #%%% 2.3. Open composite data
    composite_data = xr.open_dataset(f'../Data/Composites/Bootstrap_composites_{which_var}_{which_season}.nc')
    
    
    #%% 3. Filter data for same time period
    #%%% 3.1. Catalogues
    data_full = daily_data[daily_data.YEAR.isin([i for i in range(1970, 2025+1)])]
    masks_full = daily_masks.sel(time=slice('1970-01-01', '2025-12-31'))
    
    #%%% 3.2. Z500 data
    if which_var != 'UV':
        var_full = var_data.sel(time=slice('1970-01-01', '2025-12-31'))
        var_array_full = var_full[short].values
    elif which_var == 'UV':
        var_full1 = var_data1.sel(time=slice('1970-01-01', '2025-12-31'))
        var_array_full1 = var_full1['ivt'].values
        
        var_full2 = var_data2.sel(time=slice('1970-01-01', '2025-12-31'))
        var_array_full2 = var_full2['tcwv'].values
        
        var_array_full = var_array_full1/var_array_full2
    
    #%%% 3.3. Lat, Lon, time and box definition
    lat = daily_masks.lat.values
    lon = daily_masks.lon.values
    lons, lats = np.meshgrid(lon,lat)
    
    time_full = pd.DatetimeIndex(masks_full.time)
    
    area = area_matrix(lon, lat, 1)
    
    
    #%% 4. Choose interest boxes
    #### For subtropical ridges
    latmin_r, latmax_r = -55, -30
    step_r = 30; lons_r = np.arange(-180,180+step_r,step_r)
    lonbounds_r =  {}
    for i in range(len(lons_r)-1):
        lonbounds_r[f'lonmin_{i+1}'] = lons_r[i]
        lonbounds_r[f'lonmax_{i+1}'] = lons_r[i+1]
    
    
    #%% 5. Get ridge, blocked, and normal days
    #%%% 5.1. Function
    def get_day_type(months_to_eval, masks, masks_data, lonmin, lonmax, latmin, latmax, time, allowed_types):
        lonmin_i = np.argmin(abs(lon-lonmin))
        lonmax_i = np.argmin(abs(lon-lonmax))
        latmin_i = np.argmin(abs(lat-latmin))
        latmax_i = np.argmin(abs(lat-latmax))
                
        strct_days = []; other_days = []; normal_days = []
        accepted_structs = 0; invalidated_structs = 0
        an_sids = []
        
        indexes = np.where(time.month.isin(months_to_eval))
        dates_to_an = time.astype(str).str[:10].values[indexes]
        
        for data_string0 in dates_to_an:
            day_n = np.where(time == data_string0)[0][0]
            
            mask_day = masks[day_n]
            
            mask_day_filt = mask_day[latmax_i:latmin_i+1, lonmin_i:lonmax_i+1]
            
            unique_structs = np.array(list(filter(lambda x : x != 0, np.unique(mask_day_filt))))
            
            if len(unique_structs) == 0:
                normal_days.append(day_n)
            else:
                types_in_day = []
                temp_sids = []
                for strct_id in unique_structs:
                    strct_data_temp = masks_data[(masks_data.SID == strct_id) &
                                                 (masks_data.YEAR == int(data_string0[:4])) &
                                                 (masks_data.MONTH == int(data_string0[5:7])) &
                                                 (masks_data.DAY == int(data_string0[8:]))]
                    strct_index   = strct_data_temp.index.values[0]
                    strct_type = strct_data_temp.TYPE.values[0]
                    strct_area = np.sum((np.where(mask_day == strct_id, 1, 0)*area))
                    strct_area_filt = np.sum((np.where(mask_day == strct_id, 1, 0)*area)[latmax_i:latmin_i+1, lonmin_i:lonmax_i])
                    perc_filt = strct_area_filt/strct_area
                    if perc_filt > 0.5 and strct_area_filt>=500000:
                        types_in_day.append(strct_type)
                        temp_sids.append(strct_index)
                    
                types_in_day = np.unique(types_in_day)
                
                if len(types_in_day) != 0:
                    flag = True
                    for i in types_in_day:
                        if i not in allowed_types:
                            flag = False
                else:
                    flag = False
                    
                if flag == True:
                    strct_days.append(day_n)
                    accepted_structs += len(temp_sids)
                    for k in temp_sids:
                        an_sids.append(k)
                elif flag == False:
                    other_days.append(day_n)
                    invalidated_structs += len(unique_structs)
        
        return np.array(strct_days), np.array(other_days), np.array(normal_days)
    
    #%%% 5.2. Compute days of Ridges
    ridge_results = {}
    section_results = {}
    absolut_vals = []
    diff_results_section = []
    sign_results_section = []
    
    correlation_results = {}
    
    if regions_to_consider == 'all':
        which_regions = [0,1,2,3,4,5,6,7,8,9,10,11]  # All regions
    if regions_to_consider == 'continents':
        which_regions = [3,4,6,7,9,10,11]            # Continents
    if regions_to_consider == 'oceans':
        which_regions = [0,1,2,5,8]                  # Ocean
    
    for i in tqdm(which_regions):
        if which_season == 'winter':
            months_to_consider = [5,6,7,8,9]
        elif which_season == 'summer':
            months_to_consider = [11,12,1,2,3]
        elif which_season == 'year':
            months_to_consider = [1,2,3,4,5,6,7,8,9,10,11,12]
        results = get_day_type(months_to_consider, masks_full.Structs.values, data_full,
                               lonbounds_r[f'lonmin_{i+1}'], lonbounds_r[f'lonmax_{i+1}'], latmin_r, latmax_r,
                               time_full, ['Ridge'])
        ridge_results[f'strcts_R{i+1}'], ridge_results[f'other_R{i+1}'], ridge_results[f'normal_R{i+1}'] = results
    
    
        #%% 6. Make composites
        strct_arr = var_array_full[ridge_results[f'strcts_R{i+1}']]
        normal_arr = var_array_full[ridge_results[f'normal_R{i+1}']]
        
        
        #%% 8. Normalize coordinates
        central_lon = np.argmin(abs(lon-(lonbounds_r[f'lonmin_{i+1}']+lonbounds_r[f'lonmax_{i+1}'])/2))
        
        strct_arr_norm1 = strct_arr[:,:,central_lon-len(lon)//2:]
        if central_lon < len(lon)//2:
            strct_arr_norm2 = strct_arr[:,:,:central_lon+len(lon)//2]
        else:
            strct_arr_norm2 = strct_arr[:,:,:central_lon+len(lon)//2-len(lon)]
        section_results[f'section_{i+1}_strct'] = np.concatenate([strct_arr_norm1,strct_arr_norm2],axis=2)
        
        normal_arr_norm1 = normal_arr[:,:,central_lon-len(lon)//2:]
        if central_lon < len(lon)//2:
            normal_arr_norm2 = normal_arr[:,:,:central_lon+len(lon)//2]
        else:
            normal_arr_norm2 = normal_arr[:,:,:central_lon+len(lon)//2-len(lon)]
        section_results[f'section_{i+1}_normal'] = np.concatenate([normal_arr_norm1,normal_arr_norm2],axis=2)
    
    
        #%% 9. Get ridge absolute data for region
        ridge_data = section_results[f'section_{i+1}_strct']  # Shape: (N_ridge, lat, lon)
        
        #%% 10. Make daily comparison
        correlation_results[f'section_{i+1}'] = []
        for daily_ridge_data in ridge_data:
            a = daily_ridge_data.reshape(-1)[composite_data.c_sign.values.reshape(-1)]
            b = composite_data.c_abs.values.reshape(-1)[composite_data.c_sign.values.reshape(-1)]
            
            correlation_results[f'section_{i+1}'].append(np.corrcoef(a, b)[0, 1]**2)
    
    
    #%% 11. Save data into pkl file
    with open(f'../Data/Composites/Correlation_{which_var}_{which_season}.pickle', 'wb') as handle:
        pickle.dump(correlation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)