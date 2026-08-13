'''
Yingkai Sha
ksha@ucar.edu
'''

import os
import sys
import yaml
import dask
import zarr
import numpy as np
import xesmf as xe
import xarray as xr
import pandas as pd
from glob import glob
from dask.utils import SerializableLock

import calendar
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

# ==================================================================================== #
# get year from input
year = int(args['year'])
N_months = 12

# ==================================================================================== #
# import variable name and save location form yaml
config_name = os.path.realpath('../data_config.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# ================================================================================================ #
# WRF grid info
static_WRF_name = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_FULL/static/C404_static_8km.zarr'
ds_WRF_static = xr.open_zarr(static_WRF_name)
XLAT = ds_WRF_static['XLAT'].values
XLONG = ds_WRF_static['XLONG'].values
ds_WRF_static = ds_WRF_static.assign_coords(lat=(("south_north", "west_east"), XLAT))
ds_WRF_static = ds_WRF_static.assign_coords(lon=(("south_north", "west_east"), XLONG))

domain_inds_sn = np.arange(507).astype(np.float32)
domain_inds_we = np.arange(683).astype(np.float32)

# 200, 500, 700, 850, 900, 950
ind_pick = [14, 21, 25, 30, 32, 34]

# Finite difference of lon/lat along j (south_north)
dlat = np.gradient(ds_WRF_static["XLAT"].values, axis=0)
dlon = np.gradient(ds_WRF_static["XLONG"].values, axis=0)

# Rotation angle: how much grid-north deviates from true north
alpha = np.arctan2(dlon * np.cos(np.radians(ds_WRF_static["XLAT"].values)), dlat)
cosalpha = np.cos(alpha)
sinalpha = np.sin(alpha)

# increase the file cache size
xr.set_options(file_cache_maxsize=500)
# lock for safe parallel access
netcdf_lock = SerializableLock()

# all days within a year
start_time = datetime(year, 1, 1, 0, 0)
dt_list = [start_time + relativedelta(months=i) for i in range(N_months)]

# var names
varnames = list(conf['RDA']['varname_single'].values())

ds_list = []

for i_mon, dt in enumerate(dt_list):
    # file source info
    base_dir = dt.strftime(conf['RDA']['source']['ansfc_format'])

    first_day = datetime(year, dt.month, 1)
    last_day = datetime(year, dt.month, calendar.monthrange(year, dt.month)[1])
    
    dt_pattern = dt.strftime(conf['RDA']['source']['ansfc_dt_pattern_format'])
    dt_pattern = dt_pattern.format(first_day.day, last_day.day)
    
    # get upper-air vars
    filename_collection = [glob(base_dir + f'*{var}*{dt_pattern}*')[0] for var in varnames]
    
    if len(filename_collection) != len(varnames):
        raise ValueError(f'Year {year}, day {day_idx} has incomplete files')
    
    # Open with a lock to avoid race conditions when accessing files
    ds = xr.open_mfdataset(filename_collection, combine='by_coords', parallel=True, lock=netcdf_lock)

    # drop useless var
    ds = ds.drop_vars('utc_date', errors='ignore')
    ds = ds.chunk({'time': 4, 'latitude': -1, 'longitude': -1})
    
    # ======================================================== #
    # Interpolation block
    ds['longitude'] = (ds['longitude']  + 180) % 360 - 180
    ds = ds.sortby('longitude')
    
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    
    if i_mon == 0:
        regridder = xe.Regridder(ds, ds_WRF_static, method='bilinear')

    ds_ERA5_interp = regridder(ds)
    
    ds_ERA5_interp = ds_ERA5_interp.assign_coords(
        south_north=domain_inds_sn, 
        west_east=domain_inds_we
    )
    
    ds_ERA5_interp = ds_ERA5_interp.drop_vars(['lon', 'lat'])
    
    # convert to grid-relative u, v
    u_grid =  ds_ERA5_interp['VAR_10U'] * cosalpha + ds_ERA5_interp['VAR_10V'] * sinalpha
    v_grid = -ds_ERA5_interp['VAR_10U'] * sinalpha + ds_ERA5_interp['VAR_10V'] * cosalpha
    
    ds_ERA5_interp['VAR_10U'] = u_grid
    ds_ERA5_interp['VAR_10V'] = v_grid
    
    ds_list.append(ds_ERA5_interp)
    
# concatenate
ds_yearly = xr.concat(ds_list, dim='time')

# save to zarr
base_dir = '/glade/campaign/ral/hap/ksha/ERA5_data_full/surf_8km/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# save
save_name = base_dir + conf['RDA']['prefix'] + '_surf_{}.zarr'.format(year)
ds_yearly.to_zarr(save_name, mode='w', consolidated=True, compute=True)

print('...all done...')
