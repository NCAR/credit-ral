import os
import sys
import time
import zarr
import numpy as np
import xarray as xr
from glob import glob

sys.path.insert(0, os.path.realpath('../../libs/'))
import solar_utils as su

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('year', help='year')
args = vars(parser.parse_args())

def hourly_range_numpy(year):
    # start at midnight Jan 1 of the year
    start = np.datetime64(f'{year}-01-01T00:00', 'ns')
    # end at midnight Jan 1 of the next year
    end   = np.datetime64(f'{year+1}-01-01T00:00', 'ns')
    # step by one hour
    hours = np.arange(start, end, np.timedelta64(1, 'h'), dtype='datetime64[ns]')
    return hours

year = int(args['year'])
dt = hourly_range_numpy(year)

static_name = '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static.zarr'
ds_static = xr.open_zarr(static_name)
lon_WRF = ds_static['XLONG'].values
lat_WRF = ds_static['XLAT'].values
elev_WRF = ds_static['HGT_M'].values

lon_grid = lon_WRF
lat_grid = lat_WRF

t0 = f"{year}-01-01"
t1 = f"{year}-12-31 23:00"
toa_radiation = su.get_toa_radiation(t0, t1)

TSI = np.zeros((len(dt), 336, 336))
south_north = np.arange(336).astype(np.float32)
west_east = np.arange(336).astype(np.float32)

for i in range(336):
    for j in range(336):
        lon_val = lon_WRF[i, j]
        lat_val = lat_WRF[i, j]
        elev = elev_WRF[i, j]
        out = su.get_solar_radiation_loc(toa_radiation, lon_val, lat_val, elev, t0, t1, step_freq='1h')
        TSI[:, i, j] = out['tsi'].values[:, 0, 0]

ds = xr.Dataset(
    data_vars={"TSI": (("time", "south_north", "west_east"), TSI)},
    coords={
        "time":        dt,
        "south_north": south_north,
        "west_east":   west_east
    }
)

# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(1, 336, 336))
compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)
dict_encoding['TSI'] = {'compressor': compress, **chunk_size_3d}

save_name = f'/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/solar/solar_GP_{year}.zarr'
ds.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
print(save_name)





