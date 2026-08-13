
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

year = int(args['year'])

ds = xr.open_zarr(f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_FULL/C404/C404_FULL_{year}.zarr')
ds_8km = ds.coarsen(south_north=2, west_east=2, boundary="trim").mean()

varname_4d = ['WRF_P', 'WRF_Q', 'WRF_T', 'WRF_U', 'WRF_V', 'WRF_Q_tot']
# =================================================== #
# rechunk
ds_8km = ds_8km.chunk(
    {
        'time': 16, 
        'bottom_top': 12, 
        'pressure_approx': 12, 
        'south_north': -1, 
        'west_east': -1
    }
)

varnames = list(ds_8km.keys())
# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(16, -1, -1))
chunk_size_4d = dict(chunks=(16, 12, -1, -1))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4d:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}


save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_FULL/C404_8km/C404_8km_{year}.zarr'
ds_8km.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)


