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

fn_1h_upper = '/glade/campaign/ral/hap/ksha/ERA5_data/upper_air/ERA5_plevel_1h_upper_air_{}.zarr'
fn_1h_surf = '/glade/campaign/ral/hap/ksha/ERA5_data/surf/ERA5_plevel_1h_surf_{}.zarr'
fn_1h_land = '/glade/campaign/ral/hap/ksha/ERA5_data/land/ERA5_plevel_1h_land_{}.zarr'

static_WRF_name = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static.zarr'
ds_WRF_static = xr.open_zarr(static_WRF_name)
LANDMASK = ds_WRF_static['LANDMASK'].values

varname_4d = ['U', 'V', 'Z', 'T', 'Q']

# Open input datasets
ds_1h_upper = xr.open_zarr(fn_1h_upper.format(year), chunks={})
ds_1h_surf = xr.open_zarr(fn_1h_surf.format(year), chunks={})
ds_1h_land = xr.open_zarr(fn_1h_land.format(year), chunks={})

# Merge datasets
ds_year = xr.merge([ds_1h_upper, ds_1h_surf, ds_1h_land])

# =================================================== #
# rechunk
ds_year = ds_year.chunk(
    {
        'time': 16, 
        'level': 11, 
        'south_north': 336, 
        'west_east': 336
    }
)

varnames = list(ds_year.keys())
# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(16, 336, 336))
chunk_size_4d = dict(chunks=(16, 15, 336, 336))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4d:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale_BCSD/ERA5_GP_1h_{year}.zarr'

ds_year.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

print('...all done...')
