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

varname_init = [
    'WRF_P', 'WRF_U', 'WRF_V', 'WRF_T', 'WRF_Q_tot_05', 
    'WRF_SP', 'WRF_MSLP', 'WRF_T2', 'WRF_TD2', 'WRF_U10', 'WRF_V10', 'WRF_PWAT_05', 'WRF_IVT_U', 'WRF_IVT_V'
]

varname_4d = ['WRF_P', 'WRF_U', 'WRF_V', 'WRF_T', 'WRF_Q_tot_05']

WRF_dir = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404/C404_GP_{}.zarr'

ds_WRF = xr.open_zarr(WRF_dir.format(year))
ds_WRF_6H = ds_WRF.isel(time=slice(None, None, 6))
ds_WRF_6H = ds_WRF_6H[varname_init]

# ==================================================================== #
# re-chunk data after slicing
ds_WRF_6H = ds_WRF_6H.chunk({'time': -1}) # un-chunk time 
varnames = list(ds_WRF_6H.keys())
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

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale/C404_dscale_GP_{year}.zarr'
print(save_name)
ds_WRF_6H.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)


