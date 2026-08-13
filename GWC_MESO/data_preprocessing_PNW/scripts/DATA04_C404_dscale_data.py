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
    'WRF_U', 'WRF_V', 'WRF_T', 'WRF_Q', 'WRF_P', 
    'WRF_SP', 'WRF_T2', 'WRF_TD2', 'WRF_U10', 'WRF_V10'
]

WRF_dir = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/all_in_one/C404_GP_{}.zarr'

ds_WRF = xr.open_zarr(WRF_dir.format(year))
ds_WRF_6H = ds_WRF.isel(time=slice(None, None, 6))
ds_WRF_6H = ds_WRF_6H[varname_init]

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/dscale/C404_dscale_GP_{year}.zarr'
print(save_name)
ds_WRF_6H.to_zarr(save_name, mode='w', consolidated=True, compute=True)
