import re
import os
import sys

import zarr
import yaml
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import pandas as pd

sys.path.insert(0, os.path.realpath('../../libs/'))
import verif_utils as vu

ds_geo = xr.open_zarr('/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/static/C404_GP_static.zarr')
XLONG = ds_geo['XLONG'].values
XLAT = ds_geo['XLAT'].values

Rd = 287.05     # J kg^-1 K^-1
g  = 9.80665    # m s^-2
R_earth = 6_371_000.0  # meters

# omega-to-W conversion ratio under terrain following coord
R_adjust = np.array([
    0.1360166, 0.31784073, 0.44694656, 
    0.48276338, 0.5795002, 0.6722734, 
    0.73787826, 0.84869987, 0.9090196, 
    0.91674334, 0.8542894, 0.0
])

domain_inds = np.arange(336).astype(np.float32)

level_pick = [0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42]
level_inds = np.arange(len(level_pick)).astype(np.float32)

pressure = np.array(
    [1000, 946, 915, 867, 800, 715, 610, 500, 420, 280, 180, 100]
).astype(np.float32)

# --- 2) Horizontal divergence ∇·V = du/dx + dv/dy with variable grid spacing ---
def _central_derivative(arr, spacing, dim):
    """
    Centered difference with variable spacing.
    arr: DataArray with ... x dim
    spacing: DataArray of edge spacings with length n-1 along 'dim'
    """
    half = arr.diff(dim) / spacing                     # derivatives on edges (n-1)
    left = half.pad({dim: (1, 0)}, mode='edge')        # shift to left cell center
    right = half.pad({dim: (0, 1)}, mode='edge')       # shift to right cell center
    return 0.5 * (left + right)     

def _omega_from_divergence(div_col, p_col):
    """
    div_col, p_col: 1-D arrays along vertical 'level'
    Returns ω at full levels, same length as input.
    """
    # Ensure pressure increases with index (top -> bottom)
    rev = p_col[0] > p_col[-1]
    if rev:
        p_col = p_col[::-1]
        div_col = div_col[::-1]

    n = p_col.shape[0]
    omega = np.empty_like(div_col)
    omega[0] = 0.0  # top boundary condition: ω = 0 at top level
    for k in range(1, n):
        dp = p_col[k] - p_col[k-1]
        omega[k] = omega[k-1] - 0.5 * (div_col[k-1] + div_col[k]) * dp  # trapezoid

    if rev:
        omega = omega[::-1]
    return omega


for exp_name in ['B1H', 'B3H', 'B6H']:
    for year in range(2021, 2025):
        save_name1 = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/opt_init_ERA5/full_output/opt_{exp_name}_{year-1}_full.zarr'
        ds_final1 = xr.open_zarr(save_name1, chunks={}).sel(time=slice(f'{year-1}-10-01T00', f'{year-1}-12-31T23'))
        
        save_name2 = f'/glade/derecho/scratch/ksha/DWC_data/CONUS_domain_GP/opt_init_ERA5/full_output/opt_{exp_name}_{year}_full.zarr'
        ds_final2 = xr.open_zarr(save_name2, chunks={}).sel(time=slice(f'{year}-01-01T00', f'{year}-09-30T23'))
        
        ds_final = xr.concat([ds_final1, ds_final2], dim='time')
        
        # ========================================= #
        # non-negative variable fix
        vars_to_fix = [
            'WRF_PWAT_05', 'WRF_Q_tot_05', 'WRF_precip_025', 'WRF_radar_composite_025', 'WRF_GLW', 'WRF_SWDOWN'
        ]
        
        for var in vars_to_fix:
            ds_final[var] = ds_final[var].clip(min=0)
        
        # [0, 1] variable fix
        vars_to_fix = ['WRF_SMOIS', 'WRF_TCC']
        for var in vars_to_fix:
            ds_final[var] = ds_final[var].clip(min=0).clip(max=1)
        
        # ========================================= #
        # solar radiation fix
        ds_solar1 = xr.open_zarr(f'/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/solar/solar_GP_{year-1}.zarr')
        ds_solar2 = xr.open_zarr(f'/glade/campaign/ral/hap/ksha/DWC_data/CONUS_domain_GP/solar/solar_GP_{year}.zarr')
        
        ds_solar = xr.concat([ds_solar1, ds_solar2], dim='time')
        ds_solar_ref = ds_solar.sel(time=ds_final['time'])
        
        eps = 1e-6
        is_night = (ds_solar_ref['TSI'].fillna(0) <= eps).all(dim=('south_north', 'west_east'))
        ds_final['WRF_SWDOWN'] = ds_final['WRF_SWDOWN'].where(~is_night, other=0)
        
        # ========================================= #
        # square variables and drop originals
        ds_final = ds_final.assign(
            WRF_PWAT=ds_final['WRF_PWAT_05'] ** 2,
            WRF_Q_tot=ds_final['WRF_Q_tot_05'] ** 2
        ).drop_vars(['WRF_PWAT_05', 'WRF_Q_tot_05'])
        
        ds_final['WRF_precip'] = ds_final['WRF_precip_025']**4
        ds_final['WRF_radar_composite'] = ds_final['WRF_radar_composite_025']**4
        ds_final = ds_final.drop_vars(['WRF_precip_025', 'WRF_radar_composite_025'])
        
        # ========================================= #
        # Rename dimensions
        ds_final = ds_final.rename({
            'bottom_top': 'level',
            'south_north': 'latitude',
            'west_east': 'longitude'
        })
        
        # Subset time
        #ds_final = ds_final.isel(time=slice(None, -1))
        
        # Assign level coordinate
        # ds_final = ds_final.assign_coords(level=[0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42])
        
        # Drop old lat/lon and assign new ones
        ds_final = ds_final.drop_vars(['latitude', 'longitude'], errors='ignore').assign_coords(
            XLAT=(('latitude', 'longitude'), XLAT),
            XLONG=(('latitude', 'longitude'), XLONG)
        )
        
        # Build a DataArray aligned to the dataset's 'level' coordinate
        R_da = xr.DataArray(
            R_adjust,
            dims=["level"],
            coords={"level": ds_final["level"].values},
            name="R_adjust"
        )
        
        ds = ds_final
        
        lat2d = ds['XLAT']              # (latitude, longitude)
        lon2d = ds['XLONG']             # (latitude, longitude)
        latr  = np.deg2rad(lat2d)
        lonr  = np.deg2rad(lon2d)
        
        # Spacing along longitude (x-direction): shape (latitude, longitude-1)
        dlon = lonr.diff('longitude')
        lat_mid_lon = 0.5 * (latr.isel(longitude=slice(1, None)) + latr.isel(longitude=slice(None, -1)))
        dx = R_earth * np.cos(lat_mid_lon) * dlon  # meters
        
        # Spacing along latitude (y-direction): shape (latitude-1, longitude)
        dlat = latr.diff('latitude')
        dy = R_earth * dlat  # meters
        
        du_dx = _central_derivative(ds['WRF_U'], dx, 'longitude')   # s^-1
        dv_dy = _central_derivative(ds['WRF_V'], dy, 'latitude')    # s^-1
        div_h = du_dx + dv_dy
        
        # --- 3) Integrate ∂ω/∂p = -div to get ω (Pa s^-1); ω(top)=0 ---
        p = ds['WRF_P']  # Pa
        
        omega = xr.apply_ufunc(
            _omega_from_divergence, div_h, p,
            input_core_dims=[['level'], ['level']],
            output_core_dims=[['level']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[p.dtype]
        )
        
        ds = ds.assign(WRF_OMEGA=omega)
        
        T = ds['WRF_T']  # Kelvin
        qv = ds['WRF_Q_tot']
        
        Tv = T * (1.0 + 0.61 * qv) if qv is not None else T
        rho = ds['WRF_P'] / (Rd * Tv)                 # kg m^-3
        w = - ds['WRF_OMEGA'] / (rho * g)             # m s^-1 (positive up)
        
        ds = ds.assign(WRF_W=w)
        
        ds = ds.assign(WRF_W = ds["WRF_W"] * R_da)
        ds['WRF_W'] = ds['WRF_W'].transpose('time', 'level', 'latitude', 'longitude')
        
        ds_final = ds.drop_vars(['WRF_OMEGA'])
        
        # ============================================================================== #
        ds_final = ds_final.rename({
            'level': 'bottom_top',
            'latitude': 'south_north',
            'longitude': 'west_east' 
        })
        
        ds_final = ds_final.reset_coords(drop=True)
        ds_final = ds_final.assign_coords(
            south_north=domain_inds, 
            west_east=domain_inds, 
            bottom_top=level_inds,
            pressure_approx=pressure
        )
        
        ds_final = ds_final.assign_coords(level=[0, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 42])
        ds_final = ds_final.drop_vars(['forecast_hour'])
        output_name = f'/glade/derecho/scratch/ksha/GWC_Results_Zarr/final_{exp_name}_{year}_WY.zarr'
        ds_final.to_zarr(output_name, mode='w', consolidated=True, compute=True)
        
        print(output_name)

