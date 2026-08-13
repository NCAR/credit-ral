import os
import sys
import time
import dask
import zarr
import numpy as np
import xarray as xr
from glob import glob

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

year = int(args['year'])

upper_dir = '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_plevel_base/upper_air/'
surf_dir = '/glade/campaign/cisl/aiml/ksha/CREDIT_data/ERA5_plevel_base/surf/'

level_pick = [8, 10, 14, 17, 19, 21, 23, 25, 30, 34, 36]
varname_surf = ['MSL', 'SP', 'VAR_10U', 'VAR_10V', 'VAR_2T']

ds_upper = xr.open_zarr(upper_dir + f'ERA5_plevel_6h_upper_air_{year}.zarr')
ds_surf = xr.open_zarr(surf_dir + f'ERA5_plevel_6h_surf_{year}.zarr')

ds_upper = ds_upper.isel(
    level=level_pick,
    latitude=slice(180, 268),
    longitude=slice(1001, 1113)
).drop_vars(['W'])

ds_surf = ds_surf[varname_surf].isel(
    latitude=slice(180, 268),
    longitude=slice(1001, 1113)
)

ds_final = xr.merge([ds_upper, ds_surf])
# surface --> top, south --> north | align with WRF data 
ds_final = ds_final.isel(level=slice(None, None, -1), latitude=slice(None, None, -1))
ds_final = ds_final.load()

varnames = list(ds_final.keys())
# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(1, 88, 112))
chunk_size_4d = dict(chunks=(1, 11, 88, 112))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_surf:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/all_in_one/ERA5_GP_{year}.zarr'
ds_final.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
print(save_name)
