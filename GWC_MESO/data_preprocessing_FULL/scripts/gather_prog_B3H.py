import re
import os
import sys
import zarr
import yaml
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ind_start', help='verif_ind_start')
parser.add_argument('ind_end', help='verif_ind_end')
args = vars(parser.parse_args())

ind_start = int(args['ind_start'])
ind_end = int(args['ind_end'])

# fn_target = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404/C404_GP_2020.zarr'
# ds_target = xr.open_zarr(fn_target)
source_dir = '/glade/derecho/scratch/ksha/DWC/RAW_OUTPUT/CONUS_FULL_B6H/'

fn_all = vu.get_nc_files(source_dir)[0]
fn_all = sorted(fn_all, key=lambda x: int(re.search(r'_(\d+)\.nc$', x).group(1)))[:-1]

ds_collect = []

for fn in fn_all[ind_start:ind_end]:
    ds = xr.open_dataset(fn)
    ds_collect.append(ds)

# year = ds_collect[0]['time'].values[0].astype("datetime64[Y]").astype(int) + 1970
ds_final = xr.concat(ds_collect, dim='time')

ds_final = ds_final.rename({'latitude': 'south_north', 'longitude': 'west_east', 'level': 'bottom_top'})
ds_final['west_east'] = np.arange(336).astype(np.float32)
ds_final['south_north'] = np.arange(336).astype(np.float32)
ds_final['bottom_top'] = np.arange(12).astype(np.float32)

# =================================================== #
# combine with diag
ds_final = ds_final.chunk({'time': 12, 'bottom_top': 12, 'south_north': 336, 'west_east': 336})

# =================================================== #
# zarr encodings
dict_encoding = {}
varnames = list(ds_final.keys())
varname_4D = ['WRF_U', 'WRF_V', 'WRF_T', 'WRF_Q_tot_05', 'WRF_P']

chunk_size_3d = dict(chunks=(12, 336, 336))
chunk_size_4d = dict(chunks=(12, 12, 336, 336))
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4D:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

base_dir = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_FULL/opt_init_ERA5/prog_outputs/B3H/'
save_name = base_dir + f'opt_B3H_{ind_start:04d}_{ind_end:04d}_2020-01-01T00Z.zarr'
ds_final.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
