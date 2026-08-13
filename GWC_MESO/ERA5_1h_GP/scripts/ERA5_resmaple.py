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
fn_1h = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale_1h/ERA5_GP_1h_{}.zarr'
varname_4d = ['U', 'V', 'T', 'Q', 'Z']

ds_1h = xr.open_zarr(fn_1h.format(year))
ds_1h = ds_1h.drop_vars(['soil_temperature_level_1', 'volumetric_soil_water_layer_1'])

ds_3h = ds_1h.isel(time=slice(None, None, 3))
ds_3h = ds_3h.chunk({'time':16, 'level':11, 'west_east':336, 'south_north':336})

ds_6h = ds_1h.isel(time=slice(None, None, 6))
ds_6h = ds_6h.chunk({'time':16, 'level':11, 'west_east':336, 'south_north':336})

varnames = list(ds_3h.keys())
# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(16, 336, 336))
chunk_size_4d = dict(chunks=(16, 11, 336, 336))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4d:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

savename_3h = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale_3h/ERA5_GP_3h_{year}.zarr'
ds_3h.to_zarr(savename_3h, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

savename_6h = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale_6h/ERA5_GP_6h_{year}.zarr'
ds_6h.to_zarr(savename_6h, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

print('...all done...')
