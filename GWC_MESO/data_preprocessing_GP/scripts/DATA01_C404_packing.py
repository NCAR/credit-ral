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

def hourly_datetimes(year: int) -> np.ndarray:
    start = np.datetime64(f"{year}-01-01T00:00:00", "ns")
    stop  = np.datetime64(f"{year+1}-01-01T00:00:00", "ns")  # exclusive
    hours = np.arange(start, stop, np.timedelta64(1, "h"))
    return hours  # dtype: datetime64[ns]
    
year = int(args['year'])
dt_list = hourly_datetimes(year)
flag_soil = True

base_dir = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/raw_404/'
base_dir_new = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/raw_404_new/'

varname_4d = ['WRF_P', 'WRF_Q', 'WRF_T', 'WRF_U', 'WRF_V', 'WRF_Q_tot']

if flag_soil:
    ds_static = xr.open_zarr(
        '/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static_LAKE.zarr'
    )

    land = ds_static["LANDMASK"]   # 1 = land, 0 = water
    lake = ds_static["LAKEMASK"]   # 1 = lake, 0 = land & ocean
    ocean_mask = (land == 0) & (lake == 0)  # True = ocean

    lat2d = ds_static["XLAT"]
    lon2d = ds_static["XLONG"]

    # ensure land mask coordinate names are south_north and west_east
    land_hw = land.rename({land.dims[-2]: "south_north", land.dims[-1]: "west_east"}) if land.dims != ("south_north", "west_east") else land
    lake_hw = lake.rename({lake.dims[-2]: "south_north", lake.dims[-1]: "west_east"}) if lake.dims != ("south_north", "west_east") else lake

    # ... ocean mask ...
    ocean_hw = ocean_mask.rename({ocean_mask.dims[-2]: "south_north", ocean_mask.dims[-1]: "west_east"}) if ocean_mask.dims != ("south_north", "west_east") else ocean_mask

    # ensure latd and lon2d xr.dataarrays have dim names of south_north and west_east
    lat_hw  = lat2d.rename({lat2d.dims[-2]: "south_north", lat2d.dims[-1]: "west_east"}) if lat2d.dims != ("south_north", "west_east") else lat2d
    lon_hw  = lon2d.rename({lon2d.dims[-2]: "south_north", lon2d.dims[-1]: "west_east"}) if lon2d.dims != ("south_north", "west_east") else lon2d

    # xESMF grid datasets (dims must match the data horizontal dims)
    src_grid = xr.Dataset(
        {
            "lat":  (("south_north", "west_east"), lat_hw.values),
            "lon":  (("south_north", "west_east"), lon_hw.values),
            "mask": (("south_north", "west_east"), land_hw.values),  # 1=keep, 0=ignore
        }
    )
    dst_grid = xr.Dataset(
        {
            "lat": (("south_north", "west_east"), lat_hw.values),
            "lon": (("south_north", "west_east"), lon_hw.values),
        }
    )

    regridder = xe.Regridder(
        src_grid,
        dst_grid,
        method="bilinear",
        extrap_method="nearest_s2d",
    ) # reuse_weights=True, filename="weights_conus404_fill.nc",



# ===================================================== #
# get original files
fn_year = sorted(glob(base_dir+f'*{year}*.zarr')+glob(base_dir_new+f'*{year}*.zarr'))

if len(fn_year) > 0:
    file_collect = []
    
    for i_fn, fn in enumerate(fn_year):
        ds = xr.open_zarr(fn)
        ds['time'] = [dt_list[i_fn],]
        file_collect.append(ds)
        
    ds_year = xr.concat(file_collect, dim='time')
    
else:
    print(f'Skip year {year}')
    
# ===================================================== #
# Packing
# ===================================================== #

# merge all
ds_year = ds_year.drop_vars(['WRF_Q', 'WRF_Q_LC', 'WRF_PWAT_LC'], errors="ignore")

ds_year['WRF_precip_025'] = ds_year['WRF_precip']**0.25
ds_year['WRF_radar_composite_025'] = ds_year['WRF_radar_composite']**0.25
ds_year['WRF_PWAT_05'] = ds_year['WRF_PWAT']**0.5
ds_year['WRF_Q_tot_05'] = ds_year['WRF_Q_tot']**0.5

if flag_soil:
    # =================================================== #
    # SMOIS handling
    da_SMOIS = ds_year["WRF_SMOIS"]

    # mask out non-land; keep dims ("time","south_north","west_east")
    da_SMOIS_land = da_SMOIS.where(land_hw == 1)

    # FIX: regridder and data now share ("south_north","west_east") dims
    da_SMOIS_filled = regridder(da_SMOIS_land, skipna=True)

    # land vals corrected by original
    da_SMOIS_correct = xr.where(land_hw == 1, da_SMOIS, da_SMOIS_filled)

    # ocean vals forced to 1.0
    da_SMOIS_correct = xr.where(ocean_hw, 1.0, da_SMOIS_correct)

    da_SMOIS_correct = da_SMOIS_correct.transpose("time", "south_north", "west_east")
    ds_year["WRF_SMOIS"] = da_SMOIS_correct

    # =================================================== #
    # TSLB handling
    da_TSLB = ds_year["WRF_TSLB"]

    # treat lakes specially: mask out lakes (lake==1)
    da_TSLB_nolake = da_TSLB.where(lake_hw == 0)

    da_TSLB_filled = regridder(da_TSLB_nolake, skipna=True)

    # non-lake corrected by original; lakes replaced by filled
    da_TSLB_correct = xr.where(lake_hw == 0, da_TSLB, da_TSLB_filled)
    da_TSLB_correct = da_TSLB_correct.transpose("time", "south_north", "west_east")

    ds_year["WRF_TSLB"] = da_TSLB_correct
    
# =================================================== #
# rechunk
ds_year = ds_year.chunk(
    {
        'time': 16, 
        'bottom_top': 12, 
        'pressure_approx': 12, 
        'south_north': 336, 
        'west_east': 336
    }
)

varnames = list(ds_year.keys())
# zarr encodings
dict_encoding = {}

chunk_size_3d = dict(chunks=(16, 336, 336))
chunk_size_4d = dict(chunks=(16, 12, 336, 336))

compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)

for i_var, var in enumerate(varnames):
    if var in varname_4d:
        dict_encoding[var] = {'compressor': compress, **chunk_size_4d}
    else:
        dict_encoding[var] = {'compressor': compress, **chunk_size_3d}

save_name = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/C404_new/C404_GP_{year}.zarr'
ds_year.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
print(save_name)
print("--- %s seconds ---" % (time.time() - start_time))

