

import numpy as np
from numba import njit

# ideal gas constant of dry air
RDGAS = 287.05 # J/kg/K

# gravity
GRAVITY = 9.80665  # m/s^2

LAPSE_RATE = 0.0065  # K / m
ALPHA = LAPSE_RATE * RDGAS / GRAVITY

@njit
def interp1d_extrap(x, xp, fp):
    n = len(xp)
    result = np.empty_like(x)

    for i in range(len(x)):
        xi = x[i]

        if xi <= xp[0]:
            # Left extrapolation
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            result[i] = fp[0] + slope * (xi - xp[0])

        elif xi >= xp[n - 1]:
            # Right extrapolation
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            result[i] = fp[-1] + slope * (xi - xp[-1])

        else:
            # Interpolation
            for j in range(n - 1):
                if xp[j] <= xi < xp[j + 1]:
                    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
                    result[i] = fp[j] + slope * (xi - xp[j])
                    break

    return result

@njit
def interp_hybrid_to_pressure_levels(model_var, model_pressure, interp_pressures, use_log=True):
    
    # output info
    output_shape = (interp_pressures.shape[0], model_var.shape[1], model_var.shape[2])
    
    # allocation
    pressure_var = np.zeros(output_shape)

    # use log pressure
    interp_pres_coord = np.log(interp_pressures)

    # interpolate each air column
    for (i, j), v in np.ndenumerate(model_var[0]):
        pres_coord = np.log(model_pressure[:, i, j])
        pressure_var[:, i, j] = np.interp(interp_pres_coord, pres_coord, model_var[:, i, j])
        
    return pressure_var


@njit
def interp_geopotential_to_pressure_levels(
    geopotential,
    model_pressure,
    interp_pressures,
    surface_pressure,
    surface_geopotential,
    temperature_k,
    temp_height,
):

    # output info
    output_shape = (interp_pressures.shape[0], geopotential.shape[1], geopotential.shape[2])

    # allocation
    pressure_var = np.zeros(output_shape)

    # use log pressure
    log_interp_pressures = np.log(interp_pressures)

    # loop over each column
    for (i, j), v in np.ndenumerate(geopotential[0]):

        # the basic interpolation
        pressure_var[:, i, j] = interp1d_extrap(
            log_interp_pressures, 
            np.log(model_pressure[:, i, j]), 
            geopotential[:, i, j]
        )

        # geopotential adjustment based on lapse rate
        for pl, interp_pressure in enumerate(interp_pressures):
            if interp_pressure > surface_pressure[i, j]:
                height_agl = (geopotential[:, i, j] - surface_geopotential[i, j]) / GRAVITY
                h = np.argmin(np.abs(height_agl - temp_height[i, j]))

                temp_surface_k = temperature_k[h, i, j] + ALPHA * temperature_k[h, i, j] * (
                    surface_pressure[i, j] / model_pressure[h, i, j] - 1
                )

                surface_height = surface_geopotential[i, j] / GRAVITY
                temp_sea_level_k = temp_surface_k + LAPSE_RATE * surface_height
                temp_pl = np.minimum(temp_sea_level_k, 298.0)

                if surface_height > 2500.0:
                    gamma = (GRAVITY / surface_geopotential[i, j] * np.maximum(temp_pl - temp_surface_k, 0))

                elif 2000.0 <= surface_height <= 2500.0:
                    t_adjusted = 0.002 * ((2500 - surface_height) * temp_sea_level_k + (surface_height - 2000.0) * temp_pl)
                    gamma = (GRAVITY / surface_geopotential[i, j] * (t_adjusted - temp_surface_k))
                
                else:
                    gamma = LAPSE_RATE

                a_ln_p = (gamma * RDGAS / GRAVITY * np.log(interp_pressure / surface_pressure[i, j]))

                ln_p_ps = np.log(interp_pressure / surface_pressure[i, j])

                pressure_var[pl, i, j] = surface_geopotential[i, j] - RDGAS * temp_surface_k * ln_p_ps * (1 + a_ln_p / 2.0 + a_ln_p**2 / 6.0)

    return pressure_var


@njit
def interp_temperature_to_pressure_levels(
    model_var,
    model_pressure,
    interp_pressures,
    surface_pressure,
    surface_geopotential,
    geopotential,
    temp_height,
):

    # output info
    output_shape = (interp_pressures.shape[0], geopotential.shape[1], geopotential.shape[2])

    # allocation
    pressure_var = np.zeros(output_shape)

    log_interp_pressures = np.log(interp_pressures)
    
    for (i, j), v in np.ndenumerate(model_var[0]):
        pressure_var[:, i, j] = interp1d_extrap(
            log_interp_pressures, np.log(model_pressure[:, i, j]), model_var[:, i, j]
        )

        for pl, interp_pressure in enumerate(interp_pressures):
            if interp_pressure > surface_pressure[i, j]:
                height_agl = (geopotential[:, i, j] - surface_geopotential[i, j]) / GRAVITY
                h = np.argmin(np.abs(height_agl - temp_height[i, j]))

                temp_surface_k = model_var[h, i, j] + ALPHA * model_var[h, i, j] * (
                    surface_pressure[i, j] / model_pressure[h, i, j] - 1
                )

                surface_height = surface_geopotential[i, j] / GRAVITY
                temp_sea_level_k = temp_surface_k + LAPSE_RATE * surface_height
                temp_pl = np.minimum(temp_sea_level_k, 298.0)

                if surface_height > 2500.0:
                    gamma = (GRAVITY / surface_geopotential[i, j] * np.maximum(temp_pl - temp_surface_k, 0))

                elif 2000.0 <= surface_height <= 2500.0:
                    t_adjusted = 0.002 * ((2500 - surface_height) * temp_sea_level_k + (surface_height - 2000.0) * temp_pl)
                    gamma = (GRAVITY / surface_geopotential[i, j] * (t_adjusted - temp_surface_k))
                else:
                    gamma = LAPSE_RATE

                a_ln_p = (gamma * RDGAS / GRAVITY * np.log(interp_pressure / surface_pressure[i, j]))

                pressure_var[pl, i, j] = temp_surface_k * ( 1 + a_ln_p + 0.5 * a_ln_p**2 + 1 / 6.0 * a_ln_p**3)

    return pressure_var
