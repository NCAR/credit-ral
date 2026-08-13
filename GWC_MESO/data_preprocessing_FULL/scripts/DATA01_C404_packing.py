
import os
import sys
import time
import dask
import zarr
import xesmf as xe
import numpy as np
import xarray as xr
from glob import glob

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

def hourly_datetimes(year: int) -> np.ndarray:
    start = np.datetime64(f"{year}-01-01T00:00:00", "ns")
    stop  = np.datetime64(f"{year+1}-01-01T00:00:00", "ns")  # exclusive
    hours = np.arange(start, stop, np.timedelta64(1, "h"))
    return hours  # dtype: datetime64[ns]

year = int(args['year'])
dt_list = hourly_datetimes(year)
flag_soil = False

base_dir = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_FULL/raw_404/'
base_dir_extra = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_FULL/raw_404_new/'

varname_4d = ['WRF_P', 'WRF_Q', 'WRF_T', 'WRF_U', 'WRF_V', 'WRF_Q_tot']

if flag_soil:
    ds_static = xr.open_zarr('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_FULL/static/C404_static.zarr')
    
    land = ds_static["LANDMASK"] # 1 = land, 0 = water
    lake = ds_static['LAKEMASK'] # 1 = lake, 0 = land & ocean
    ocean_mask = (land == 0) & (lake == 0) # 1 = ocean
    
    lat2d = ds_static["XLAT"]
    lon2d = ds_static["XLONG"]
    
    # --- xESMF wants its grid as an xarray Dataset -------------------------------
    src_grid = xr.Dataset(
        {
            "lat":  (("y", "x"), lat2d.values),
            "lon":  (("y", "x"), lon2d.values),
            "mask": (("y", "x"), land.values), # 1 = keep, 0 = ignore
        }
    )
    
    # Destination grid is the *same* geometry but **without** the mask
    dst_grid = xr.Dataset({"lat": (("y", "x"), lat2d.values),
                           "lon": (("y", "x"), lon2d.values)})
    
    regridder = xe.Regridder(
        src_grid, dst_grid, 
        method = "bilinear",
        extrap_method = "nearest_s2d",
    )


start_time = time.time() 
fn_year = sorted(glob(base_dir+f'*{year}*.zarr')+glob(base_dir_extra+f'*{year}*.zarr'))

if len(fn_year) > 0:
    file_collect = []
    
    for i_fn, fn in enumerate(fn_year):
        ds = xr.open_zarr(fn)
        ds['time'] = [dt_list[i_fn],]
        file_collect.append(ds)
        
    ds_year = xr.concat(file_collect, dim='time')
    
    # merge all
    ds_year = ds_year.drop_vars(['WRF_Q',])
    
    ds_year['WRF_precip_025'] = ds_year['WRF_precip']**0.25
    ds_year['WRF_radar_composite_025'] = ds_year['WRF_radar_composite']**0.25
    ds_year['WRF_PWAT_05'] = ds_year['WRF_PWAT']**0.5
    ds_year['WRF_Q_tot_05'] = ds_year['WRF_Q_tot']**0.5

    if flag_soil:
        # =================================================== #
        # SMOIS handling
        da_SMOIS = ds_year["WRF_SMOIS"]
        da_SMOIS_land = da_SMOIS.where(land == 1)
        
        da_SMOIS_filled = regridder(da_SMOIS_land, skipna=True,)
        da_SMOIS_filled = da_SMOIS_filled.rename({'y': 'south_north', 'x': 'west_east'})
        da_SMOIS_filled['south_north'] = da_SMOIS['south_north']
        da_SMOIS_filled['west_east'] = da_SMOIS['west_east']
    
        # land vals in da_SMOIS_filled corrected by da_SMOIS
        da_SMOIS_correct = xr.where(land == 1, da_SMOIS, da_SMOIS_filled)
    
        # ocean vals in da_SMOIS_filled corrected by 0.0
        da_SMOIS_correct = xr.where(ocean_mask == 1, 0.0, da_SMOIS_correct)
        da_SMOIS_correct = da_SMOIS_correct.transpose("time", "south_north", "west_east")
        
        ds_year["WRF_SMOIS"] = da_SMOIS_correct
        
        # =================================================== #
        # TSLB handling
        da_TSLB = ds_year["WRF_TSLB"]
        da_TSLB_land = da_TSLB.where(lake == 0)
        
        da_TSLB_filled = regridder(da_TSLB_land, skipna=True,)
        da_TSLB_filled = da_TSLB_filled.rename({'y': 'south_north', 'x': 'west_east'})
        da_TSLB_filled['south_north'] = da_TSLB['south_north']
        da_TSLB_filled['west_east'] = da_TSLB['west_east']
    
        # non-lake vals in da_TSLB_filled corrected by da_TSLB
        da_TSLB_correct = xr.where(lake == 0, da_TSLB, da_TSLB_filled)
        da_TSLB_correct = da_TSLB_correct.transpose("time", "south_north", "west_east")
    
        ds_year["WRF_TSLB"] = da_TSLB_correct

    # =================================================== #
    # rechunk
    ds_year = ds_year.chunk(
        {
            'time': 16, 
            'bottom_top': 12, 
            'pressure_approx': 12, 
            'south_north': 1015, 
            'west_east': 1367
        }
    )
    
    varnames = list(ds_year.keys())
    # zarr encodings
    dict_encoding = {}
    
    chunk_size_3d = dict(chunks=(16, 1015, 1367))
    chunk_size_4d = dict(chunks=(16, 12, 1015, 1367))
    
    compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)
    
    for i_var, var in enumerate(varnames):
        if var in varname_4d:
            dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
        else:
            dict_encoding[var] = {'compressor': compress, **chunk_size_3d}
    
    save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_FULL/C404/C404_FULL_{year}.zarr'
    ds_year.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
    print(save_name)
    print("--- %s seconds ---" % (time.time() - start_time))
else:
    print(f'Skip year {year}')



