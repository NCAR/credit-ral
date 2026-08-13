import os
import sys
import dask
import zarr
import time
import xesmf as xe
import numpy as np
import xarray as xr
from glob import glob

import wrf

sys.path.insert(0, os.path.realpath('../../libs/'))
import plevel_utils as pu

# parse input
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('ind_start', help='ind_start')
parser.add_argument('ind_end', help='ind_end')
args = vars(parser.parse_args())

ind_start = int(args['ind_start'])
ind_end = int(args['ind_end'])

GRAVITY = pu.GRAVITY

def wrf_destag(darray, stagger_dim):
    if stagger_dim not in darray.dims:
        raise ValueError(f"Dimension '{stagger_dim}' not found in input DataArray.")
    
    return 0.5 * (darray.isel({stagger_dim: slice(0, -1)}) +
                  darray.isel({stagger_dim: slice(1, None)}))

var_2d = [
    'ACDEWC', 'ACDRIPR', 'ACDRIPS', 'ACECAN', 'ACEDIR', 'ACETLSM',
    'ACETRAN', 'ACEVAC', 'ACEVB', 'ACEVC', 'ACEVG', 'ACFROC',
    'ACFRZC', 'ACGHB', 'ACGHFLSM', 'ACGHV', 'ACINTR', 'ACINTS',
    'ACIRB', 'ACIRC', 'ACIRG', 'ACLHFLSM', 'ACLWDNB', 'ACLWDNBC',
    'ACLWDNLSM', 'ACLWDNT', 'ACLWDNTC', 'ACLWUPB', 'ACLWUPBC', 'ACLWUPLSM',
    'ACLWUPT', 'ACLWUPTC', 'ACMELTC', 'ACPAHB', 'ACPAHG', 'ACPAHLSM',
    'ACPAHV', 'ACPONDING', 'ACQLAT', 'ACQRF', 'ACRAINLSM', 'ACRAINSNOW',
    'ACRUNSB', 'ACRUNSF', 'ACSAGB', 'ACSAGV', 'ACSAV', 'ACSHB',
    'ACSHC', 'ACSHFLSM', 'ACSHG', 'ACSNBOT', 'ACSNFRO', 'ACSNOM',
    'ACSNOWLSM', 'ACSNSUB', 'ACSUBC', 'ACSWDNB', 'ACSWDNBC', 'ACSWDNLSM',
    'ACSWDNT', 'ACSWDNTC', 'ACSWUPB', 'ACSWUPBC', 'ACSWUPLSM', 'ACSWUPT',
    'ACSWUPTC', 'ACTHROR', 'ACTHROS', 'ACTR', 'ALBEDO', 'CANICE',
    'CANWAT', 'COSZEN', 'EMISS', 'FORCPLSM', 'FORCQLSM', 'FORCTLSM',
    'FORCWLSM', 'FORCZLSM', 'GLW', 'GRAUPEL_ACC_NC', 'HFX', 'I_ACLWDNB',
    'I_ACLWDNBC', 'I_ACLWDNT', 'I_ACLWDNTC', 'I_ACLWUPB', 'I_ACLWUPBC', 'I_ACLWUPT',
    'I_ACLWUPTC', 'I_ACSWDNB', 'I_ACSWDNBC', 'I_ACSWDNT', 'I_ACSWDNTC', 'I_ACSWUPB',
    'I_ACSWUPBC', 'I_ACSWUPT', 'I_ACSWUPTC', 'LAI', 'LH', 'LWDNB',
    'LWDNBC', 'LWDNT', 'LWDNTC', 'LWUPB', 'LWUPBC', 'LWUPT',
    'LWUPTC', 'MLCAPE', 'MLCINH', 'MLLCL', 'MU', 'MUCAPE',
    'MUCINH', 'OLR', 'P', 'PBLH', 'PREC_ACC_NC', 'PSFC',
    'PWAT', 'Q2', 'QFX', 'QRFS', 'QSLAT', 'QSPRINGS',
    'QVAPOR', 'RAINNCV', 'RECH', 'REFL_10CM', 'REFL_1KM_AGL', 'REFL_COM',
    'SBCAPE', 'SBCINH', 'SBLCL', 'SEAICE', 'SH2O', 'SMCWTD',
    'SMOIS', 'SNICE', 'SNLIQ', 'SNOW', 'SNOWC', 'SNOWENERGY',
    'SNOWH', 'SNOW_ACC_NC', 'SOILENERGY', 'SR', 'SRH01', 'SRH03',
    'SST', 'SSTSK', 'SWDNB', 'SWDNBC', 'SWDNT', 'SWDNTC',
    'SWDOWN', 'SWNORM', 'SWUPB', 'SWUPBC', 'SWUPT', 'SWUPTC',
    'T2', 'TD2', 'TG', 'TH2', 'TK', 'TRAD',
    'TSK', 'TSLB', 'TSNO', 'TV', 'Times', 'U',
    'U10', 'USHR1', 'USHR6', 'UST', 'U_BUNK', 'V',
    'V10', 'VSHR1', 'VSHR6', 'V_BUNK', 'W', 'Z',
    'ZSNSO', 'ZWT', 'totalIce', 'totalLiq', 'totalVap',
    'index_snow_layers_stag', 'index_snso_layers_stag', 'index_soil_layers_stag'
]

var_3d = [
    'CLDFRA', 'O3RAD', 'P', 'P_HYD', 'QCLOUD',
    'QGRAUP', 'QICE', 'QNICE', 'QNRAIN', 'QRAIN',
    'QSNOW', 'QVAPOR', 'REFL_10CM', 'TK', 'Times',
    'U', 'V', 'W', 'Z', 'ilev', 'lev'
]

var_2d_pick = ['Z', 'PSFC']
var_3d_pick= ['P', 'TK', 'Z', 'CLDFRA']

var_2d_drop = list(set(var_2d) - set(var_2d_pick))
var_3d_drop = list(set(var_3d) - set(var_3d_pick))

rename_3d = {'TK': 'TK_3d', 'P': 'P_3d', 'Z': 'Z_3d'}
rename_2d = {'Z': 'Z_2d'}

level_pick = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 21, 24, 30, 36, 42, 49]

varname_time = 'Time'
varname_lat = 'south_north'
varname_lon = 'west_east'

dim_3D = (varname_time, varname_lat, varname_lon)

domain_inds = np.arange(336).astype(np.float32)
level_inds = np.arange(len(level_pick)).astype(np.float32)

pressure = np.array(
    [1000, 960, 940, 920, 880, 850, 800, 750, 680, 610, 500, 420, 280, 180, 100, 50]
).astype(np.float32)


fn_2d_full = sorted(glob('/glade/campaign/collections/rda/data/d559000/wy*/*/wrf2d*'))
fn_2d_subset = fn_2d_full[ind_start:ind_end]
L = len(fn_2d_subset)

for i in range(L):
    start_time = time.time()    
    fn_2d = fn_2d_subset[i]
    fn_3d = fn_2d[:62] + '3d' + fn_2d[64:]
    
    ds_2d = xr.open_dataset(fn_2d, drop_variables=var_2d_drop)
    ds_2d['Z'] = ds_2d['Z'] * GRAVITY 
    ds_2d = ds_2d.drop_vars(['XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V'])
    
    # ------------------------------------------------------------ #
    # 3d destag
    ds_3d = xr.open_dataset(fn_3d, drop_variables=var_3d_drop)
    ds_3d['Z'] = ds_3d['Z'] * GRAVITY 
    
    Z_destag_3d = wrf_destag(ds_3d['Z'], 'bottom_top_stag')
    Z_destag_3d = Z_destag_3d.rename({'bottom_top_stag': 'bottom_top'})

    ds_3d = ds_3d.drop_vars(['Z',])
    ds_3d['Z'] = Z_destag_3d
    
    ds_3d = ds_3d.drop_vars(['XLAT_U', 'XLONG_U', 'XLAT_V', 'XLONG_V'])
    
    # -------------------------------------------------------- #
    # merge together
    ds_2d = ds_2d.rename(rename_2d)
    ds_3d = ds_3d.rename(rename_3d)
    ds_mlevel = xr.merge([ds_2d, ds_3d])
    ds_mlevel = ds_mlevel.isel(west_east=slice(570, 905+1, 1), south_north=slice(200, 535+1, 1))
    
    #ds_mlevel = ds_mlevel.rename(rename_final)
    ds_mlevel = ds_mlevel.map(lambda x: x.astype(np.float32) if x.dtype == np.float64 else x)
    
    ds_raw = ds_mlevel
    
    coords_surface = {
            varname_time: ds_raw[varname_time],
            varname_lat: ds_raw[varname_lat],
            varname_lon: ds_raw[varname_lon],
    }
    # -------------------------------------------------------- #
    # MSL interpolation
    pressure_3D = ds_raw['P_3d'].values
    GP_3D = ds_raw['Z_3d'].values
    T_3D = ds_raw['TK_3d'].values
    Z_model_bot = ds_raw['Z_2d'].values
    PSFC = ds_raw['PSFC'].values

    MSLP = xr.Dataset(
        data_vars={
            'WRF_MSLP': xr.DataArray(
                coords=coords_surface,
                dims=dim_3D,
                name='WRF_MSLP',
            )
        },
        coords=coords_surface,
    )

    for t, time_val in enumerate(ds_raw[varname_time]):
        MSLP['WRF_MSLP'][t] = (
            pu.MSLP_convert(
                PSFC[t],
                T_3D[t],
                pressure_3D[t],
                Z_model_bot[t],
                GP_3D[t],
                temp_height=150.0,
            )
        )
        
    # -------------------------------------------------------- #
    # Total Cloud Cover (TCC)
    ds_mlevel['WRF_TCC'] = 1.0 - (1.0 - ds_mlevel['CLDFRA']).prod(dim='bottom_top', skipna=False)
    
    ds_mlevel = xr.merge([ds_mlevel, MSLP])
    ds_mlevel = ds_mlevel.map(lambda x: x.astype(np.float32) if x.dtype == np.float64 else x)
    ds_mlevel = ds_mlevel.drop_vars(['PSFC', 'Z_2d', 'CLDFRA', 'P_3d', 'TK_3d', 'Z_3d'])

    ds_mlevel = ds_mlevel.rename({'Time': 'time'})    
    ds_mlevel = ds_mlevel.reset_coords(drop=True)
    ds_mlevel = ds_mlevel.assign_coords(
        south_north=domain_inds, 
        west_east=domain_inds, 
        bottom_top=level_inds,
        pressure_approx=pressure
    )
    
    # -------------------------------------------------------- #
    if i == 0:
        varnames = list(ds_mlevel.keys())
        # zarr encodings
        dict_encoding = {}
        chunk_size_3d = dict(chunks=(1, 336, 336))
        
        compress = zarr.Blosc(cname='zstd', clevel=1, shuffle=zarr.Blosc.SHUFFLE, blocksize=0)
        
        for i_var, var in enumerate(varnames):
            dict_encoding[var] = {'compressor': compress, **chunk_size_3d}
    
    save_name = '/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/raw_404_P_extra/' + 'C404_GP_' + fn_2d[-22:-9] + 'H_P_extra.zarr'
    print(save_name)
    ds_mlevel.to_zarr(save_name, mode='w', consolidated=True, compute=True, encoding=dict_encoding)
    print("--- %s seconds ---" % (time.time() - start_time))


