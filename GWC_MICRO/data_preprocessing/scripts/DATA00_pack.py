
import os
import re
import sys
import zarr
import time
import numpy as np
import xarray as xr
from glob import glob
from datetime import datetime, timedelta

#sys.path.insert(0, os.path.realpath('../../libs/'))

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ind', help='ind')
args = vars(parser.parse_args())

ind = int(args['ind'])

dt_strings = ['2018010100', '2018010400', '2018011200', '2018011300', '2018011400', '2018011500', '2018011800', '2018012000', '2018012200', '2018012400', '2018012600', '2018012800', '2018013100']

variable_collection = [
    'u', 'v', 'w', 'theta', 'rho', 'qv', 'TKE_0'
]

# ind_pick = [1, 4, 7, 10, 15, 20, 30, 40, 50]
ind_pick = [1, 4, 20, 40]
start = time.time()

base_str = dt_strings[ind]
base_date = datetime.strptime(base_str, '%Y%m%d%H')

# ========================================================== #
# Fasteddy time coordinates
start_time = base_date.replace(hour=12, minute=15)
end_time = base_date.replace(hour=13, minute=0)

step = timedelta(seconds=30) # <----------------------- 30 second per step
n_steps = int((end_time - start_time) / step) + 1
time_list = [start_time + i * step for i in range(n_steps)]

# ========================================================== #
# output file process

fn_all = glob(f'/glade/derecho/scratch/casali/DigitalTwin/InputData/{base_str}/output/*')
fn_all = sorted(fn_all, key=lambda x: int(x.split('.')[-1]))

ds_collect = []
for i_fn, fn in enumerate(fn_all):
    ds = xr.open_dataset(fn)
    ds = ds[variable_collection].isel(zIndex=ind_pick)
    ds = ds.assign_coords(time=("time", [np.datetime64(time_list[i_fn], 'ns')]))
    ds_collect.append(ds)

ds_all = xr.concat(ds_collect, 'time')
ds_all = ds_all.isel(yIndex=slice(151, -151, 1), xIndex=slice(152, -152, 1)) # 16*37

# ========================================================== #
# variable chunking
varnames = list(ds_all.keys())
# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(12, 592, 592))
chunk_size_4d = dict(chunks=(12, 4, 592, 592))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    dict_encoding[var] = {'compressor': compress, **chunk_size_4d}

save_name = '/glade/derecho/scratch/ksha/FastEddy/FE_04lev_new/experiment_{:02d}.zarr'.format(ind)
print(save_name)
ds_all.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)

end = time.time()
print(f"Elapsed time: {end - start:.3f} seconds")

