
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

base_dir = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_PNW/raw_404/'
base_dir_extra = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_PNW/raw_404_E_extra/'
varname_4d = ['WRF_P', 'WRF_Q', 'WRF_T', 'WRF_U', 'WRF_V', 'WRF_Q_tot']

start_time = time.time() 
fn_year = sorted(glob(base_dir+f'C404_*{year}*H.zarr'))
fn_year_extra = sorted(glob(base_dir_extra+f'C404_*{year}*extra.zarr'))

if len(fn_year) > 0:
    file_collect = []
    file_collect_extra = []
    
    for fn in fn_year:
        ds = xr.open_zarr(fn)
        ds = ds.isel(bottom_top=slice(0, 15), pressure_approx=slice(0, 15))
        file_collect.append(ds)
        
    for fn in fn_year_extra:
        ds = xr.open_zarr(fn)
        ds = ds.isel(bottom_top=slice(0, 15), pressure_approx=slice(0, 15))
        file_collect_extra.append(ds)
        
    ds_year_base = xr.concat(file_collect, dim='time')
    ds_year_extra = xr.concat(file_collect_extra, dim='time')
    
    # merge all
    ds_year = xr.merge([ds_year_base, ds_year_extra])
    ds_year = ds_year.drop_vars(['WRF_Q', 'WRF_MLCAPE', 'WRF_Q_LC', 'WRF_evapor', 'WRF_PWAT_LC'])
    ds_year = ds_year.rename(
        {
            'SWDOWN': 'WRF_SWDOWN', 
            'GLW': 'WRF_GLW',
            'SBCAPE': 'WRF_SBCAPE'
        }
    )
    
    ds_year['WRF_precip_025'] = ds_year['WRF_precip']**0.25
    ds_year['WRF_radar_composite_025'] = ds_year['WRF_radar_composite']**0.25
    ds_year['WRF_PWAT_05'] = ds_year['WRF_PWAT']**0.5
    ds_year['WRF_Q_tot_05'] = ds_year['WRF_Q_tot']**0.5
    
    # =================================================== #
    # rechunk
    ds_year = ds_year.chunk(
        {
            'time': 16, 
            'bottom_top': 15, 
            'pressure_approx': 15, 
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
    
    save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_PNW/C404/C404_GP_{year}.zarr'
    ds_year.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
    print(save_name)
    print("--- %s seconds ---" % (time.time() - start_time))
else:
    print(f'Skip year {year}')

