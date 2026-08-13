

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
def interp_WRF_to_plevels(model_var, pressure_3D, plevel_interp):
    
    # output info
    output_shape = (plevel_interp.shape[0], model_var.shape[1], model_var.shape[2])
    
    # allocation
    interp_output = np.zeros(output_shape)

    # use log pressure
    interp_pres_coord = np.log(plevel_interp)

    # interpolate each air column
    for (i, j), v in np.ndenumerate(model_var[0]):
        pres_coord = np.log(pressure_3D[:, i, j])
        interp_output[:, i, j] = np.interp(interp_pres_coord, pres_coord, model_var[:, i, j])
        
    return interp_output


@njit
def interp_GP_to_plevels(
    model_GP,
    pressure_3D,
    plevel_interp,
    SP,
    GP_surf,
    T_3D,
    Z_model_bot,
):

    # output info
    output_shape = (plevel_interp.shape[0], model_GP.shape[1], model_GP.shape[2])

    # allocation
    interp_output = np.zeros(output_shape)

    # use log pressure
    log_plevel_interp = np.log(plevel_interp)

    # loop over each column
    for (i, j), v in np.ndenumerate(model_GP[0]):

        # the basic interpolation
        interp_output[:, i, j] = interp1d_extrap(
            log_plevel_interp, 
            np.log(pressure_3D[:, i, j]), 
            model_GP[:, i, j]
        )

        # model_GP adjustment based on lapse rate
        for i_pl, plevel_val in enumerate(plevel_interp):
            if plevel_val > SP[i, j]:
                height_agl = (model_GP[:, i, j] - GP_surf[i, j]) / GRAVITY
                h = np.argmin(np.abs(height_agl - Z_model_bot[i, j]))

                T_surf = T_3D[h, i, j] + ALPHA * T_3D[h, i, j] * (
                    SP[i, j] / pressure_3D[h, i, j] - 1
                )

                Z_surf = GP_surf[i, j] / GRAVITY
                T_msl = T_surf + LAPSE_RATE * Z_surf
                T_plevel = np.minimum(T_msl, 298.0)

                if Z_surf > 2500.0:
                    gamma = (GRAVITY / GP_surf[i, j] * np.maximum(T_plevel - T_surf, 0))

                elif 2000.0 <= Z_surf <= 2500.0:
                    T_star = 0.002 * ((2500 - Z_surf) * T_msl + (Z_surf - 2000.0) * T_plevel)
                    gamma = (GRAVITY / GP_surf[i, j] * (T_star - T_surf))
                
                else:
                    gamma = LAPSE_RATE

                a_ln_p = (gamma * RDGAS / GRAVITY * np.log(plevel_val / SP[i, j]))

                ln_p_ps = np.log(plevel_val / SP[i, j])

                interp_output[i_pl, i, j] = GP_surf[i, j] - RDGAS * T_surf * ln_p_ps * (1 + a_ln_p / 2.0 + a_ln_p**2 / 6.0)

    return interp_output


@njit
def interp_T_to_plevels(
    model_T_3D,
    pressure_3D,
    plevel_interp,
    SP,
    GP_surf,
    model_GP,
    Z_model_bot,
):

    # output info
    output_shape = (plevel_interp.shape[0], model_GP.shape[1], model_GP.shape[2])

    # allocation
    interp_output = np.zeros(output_shape)

    log_plevel_interp = np.log(plevel_interp)
    
    for (i, j), v in np.ndenumerate(model_T_3D[0]):
        interp_output[:, i, j] = interp1d_extrap(
            log_plevel_interp, np.log(pressure_3D[:, i, j]), model_T_3D[:, i, j]
        )

        for i_pl, plevel_val in enumerate(plevel_interp):
            if plevel_val > SP[i, j]:
                height_agl = (model_GP[:, i, j] - GP_surf[i, j]) / GRAVITY
                h = np.argmin(np.abs(height_agl - Z_model_bot[i, j]))

                T_surf = model_T_3D[h, i, j] + ALPHA * model_T_3D[h, i, j] * (
                    SP[i, j] / pressure_3D[h, i, j] - 1
                )

                Z_surf = GP_surf[i, j] / GRAVITY
                T_msl = T_surf + LAPSE_RATE * Z_surf
                T_plevel = np.minimum(T_msl, 298.0)

                if Z_surf > 2500.0:
                    gamma = (GRAVITY / GP_surf[i, j] * np.maximum(T_plevel - T_surf, 0))

                elif 2000.0 <= Z_surf <= 2500.0:
                    T_star = 0.002 * ((2500 - Z_surf) * T_msl + (Z_surf - 2000.0) * T_plevel)
                    gamma = (GRAVITY / GP_surf[i, j] * (T_star - T_surf))
                else:
                    gamma = LAPSE_RATE

                a_ln_p = (gamma * RDGAS / GRAVITY * np.log(plevel_val / SP[i, j]))

                interp_output[i_pl, i, j] = T_surf * ( 1 + a_ln_p + 0.5 * a_ln_p**2 + 1 / 6.0 * a_ln_p**3)

    return interp_output


@njit
def MSLP_convert(
    surface_pressure_pa,
    temperature_k,
    pressure_pa,
    surface_geopotential,
    geopotential,
    temp_height=150.0,
):
    mslp = np.zeros(surface_pressure_pa.shape, dtype=surface_pressure_pa.dtype)
    for (i, j), p in np.ndenumerate(mslp):
        if np.abs(surface_geopotential[i, j] / GRAVITY) < 1e-4:
            mslp[i, j] = surface_pressure_pa[i, j]
        else:
            height_agl = (geopotential[:, i, j] - surface_geopotential[i, j]) / GRAVITY
            h = np.argmin(np.abs(height_agl - temp_height))
            temp_surface_k = temperature_k[h, i, j] + ALPHA * temperature_k[h, i, j] * (
                surface_pressure_pa[i, j] / pressure_pa[h, i, j] - 1
            )
            temp_sealevel_k = (
                temp_surface_k + LAPSE_RATE * surface_geopotential[i, j] / GRAVITY
            )

            if (temp_surface_k <= 290.5) and (temp_sealevel_k > 290.5):
                gamma = GRAVITY / surface_geopotential[i, j] * (290.5 - temp_surface_k)
            elif (temp_surface_k > 290.5) and (temp_sealevel_k > 290.5):
                gamma = 0.0
                temp_surface_k = 0.5 * (290.5 + temp_surface_k)
            else:
                gamma = LAPSE_RATE
                if temp_surface_k < 255:
                    temp_surface_k = 0.5 * (255.0 + temp_surface_k)
            beta = surface_geopotential[i, j] / (RDGAS * temp_surface_k)
            x = gamma * surface_geopotential[i, j] / (GRAVITY * temp_surface_k)
            mslp[i, j] = surface_pressure_pa[i, j] * np.exp(
                beta * (1.0 - x / 2.0 + x**2 / 3.0)
            )
    return mslp