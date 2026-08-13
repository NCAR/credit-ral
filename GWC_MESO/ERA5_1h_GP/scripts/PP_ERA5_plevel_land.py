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

# ==================================================================================== #
# import variable name and save location form yaml
config_name = os.path.realpath('../data_config.yml')

with open(config_name, 'r') as stream:
    conf = yaml.safe_load(stream)

# ==================================================================================== #
static_WRF_name = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static.zarr'
ds_WRF_static = xr.open_zarr(static_WRF_name)
XLAT = ds_WRF_static['XLAT'].values
XLONG = ds_WRF_static['XLONG'].values
ds_WRF_static = ds_WRF_static.assign_coords(lat=(("south_north", "west_east"), XLAT))
ds_WRF_static = ds_WRF_static.assign_coords(lon=(("south_north", "west_east"), XLONG))
domain_inds = np.arange(336).astype(np.float32)
# 1000.,  950.,  850.,  700.,  600.,  500.,  400.,  300.,  200.,  100., 50.
ind_pick = [36, 34, 30, 25, 23, 21, 19, 17, 14, 10, 8]

# ==================================================================================== #
varnames = ['total_precipitation', 'total_column_water']

# save to zarr
base_dir = conf['ARCO']['save_loc'] + 'land/' 
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)


chunk_size_3d = dict(chunks=(4, 336, 336))

dict_encoding = {}

for i_var, var in enumerate(varnames):
    dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

# ==================================================================================== #
ERA5_1h = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    chunks=None,
    storage_options=dict(token='anon'),)[varnames]

time_start = '{}-01-01T00'.format(year)
time_end = '{}-12-31T23'.format(year)

ds = ERA5_1h.sel(time=slice(time_start, time_end))

# ======================================================== #
# Interpolation block
ds['longitude'] = (ds['longitude']  + 180) % 360 - 180
ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

regridder = xe.Regridder(ds, ds_WRF_static, method='bilinear')

ds_ERA5_interp = regridder(ds)

ds_ERA5_interp = ds_ERA5_interp.assign_coords(
    south_north=domain_inds, 
    west_east=domain_inds
)

ds_ERA5_interp = ds_ERA5_interp.drop_vars(['lon', 'lat'])

save_name = base_dir + conf['ARCO']['prefix'] + '_land_{}.zarr'.format(year)
print(save_name)

ds_ERA5_interp.to_zarr(save_name, mode="w", consolidated=True, compute=True, encoding=dict_encoding)

print('...all done...')
