"""
output.py
-------------------------------------------------------
Content:
    - load_metadata()
    - make_xarray()
    - save_netcdf_increment()
"""

import os
import yaml
import torch
import logging
import traceback
import xarray as xr
from credit.data import drop_var_from_dataset
from credit.interp import full_state_pressure_interpolation
from inspect import signature
from credit.credit_ptype import CreditPostProcessor

from datetime import datetime
import numpy as np


logger = logging.getLogger(__name__)


def load_metadata(conf):
    """
    Load metadata attributes from yaml file in credit/metadata directory
    """
    # set priorities for user-specified metadata
    if conf["predict"]["metadata"]:
        meta_file = conf["predict"]["metadata"]
        with open(meta_file) as f:
            meta_data = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        print("conf['predict']['metadata'] not given. Skip.")
        meta_data = False

    return meta_data


def split_and_reshape(tensor, conf):
    """
    Split the output tensor of the model to upper air variables and diagnostics/surface variables.

    Upperair level arrangement: top-of-atmosphere--> near-surface --> single layer
    An example: U (top-of-atmosphere) --> U (near-surface) --> V (top-of-atmosphere) --> V (near-surface)
    The shape of the output tensor is (variables, latitude, longitude)

    Args:
        tensor: PyTorch Tensor containing output of the AI NWP model
        conf: config file for the model

    """

    # get the number of levels
    levels = conf["model"]["levels"]

    # get number of channels
    channels = len(conf["data"]["variables"])
    single_level_channels = len(conf["data"]["surface_variables"]) + len(conf["data"]["diagnostic_variables"])

    # subset upper air variables
    tensor_upper_air = tensor[:, : int(channels * levels), :, :]

    shape_upper_air = tensor_upper_air.shape
    tensor_upper_air = tensor_upper_air.reshape(shape_upper_air[0], channels, levels, shape_upper_air[-2], shape_upper_air[-1])

    # subset surface variables
    tensor_single_level = tensor[:, -int(single_level_channels) :, :, :]

    # return x, surf for B, c, lat, lon output
    return tensor_upper_air, tensor_single_level


def make_xarray(pred, forecast_datetime, lat, lon, conf):
    """
    Create two xarray.DataArray objects for upper air and surface variables.

    Args
        pred (torch.Tensor or np.ndarray): Prediction tensor containing both upper air and surface variables.
    forecast_datetime (datetime): The forecast initialization datetime.
    lat (np.ndarray or list): Latitude values.
    lon (np.ndarray or list): Longitude values.
    conf (dict): Configuration dictionary containing details about the data structure
        and variables.

    Returns:
        darray_upper_air (xarray.DataArray): DataArray containing upper air variables with dimensions
            [time, vars, level, latitude, longitude].
    darray_single_level (xarray.DataArray): DataArray containing surface variables with dimensions
        [time, vars, latitude, longitude].
    """
    # subset upper air and surface variables
    tensor_upper_air, tensor_single_level = split_and_reshape(pred, conf)

    # -------------------------------------------- #
    # level inds
    if "level_ids" in conf["data"].keys():
        level_ids = conf["data"]["level_ids"]
    else:
        level_ids = range(conf["model"]["levels"])

    # save upper air variables
    varname_upper = conf["data"]["variables"]

    # make xr.DatasArray
    darray_upper_air = xr.DataArray(
        tensor_upper_air,
        dims=["time", "vars", "level", "latitude", "longitude"],
        coords=dict(
            vars=varname_upper,
            time=[forecast_datetime],
            level=level_ids,
            latitude=lat,
            longitude=lon,
        ),
    )

    # save surface variables
    varname_single_level = conf["data"]["surface_variables"] + conf["data"]["diagnostic_variables"]

    if len(varname_single_level) > 0:
        # make xr.DatasArray
        darray_single_level = xr.DataArray(
            tensor_single_level.squeeze(2),
            dims=["time", "vars", "latitude", "longitude"],
            coords=dict(
                vars=varname_single_level,
                time=[forecast_datetime],
                latitude=lat,
                longitude=lon,
            ),
        )

        # return x-arrays as outputs
        return darray_upper_air, darray_single_level
    else:
        return darray_upper_air


def make_xarray_diag(pred, forecast_datetime, lat, lon, conf):
    tensor_single_level = pred

    varname_single_level = conf["data"]["diagnostic_variables"]

    # make xr.DatasArray
    darray_single_level = xr.DataArray(
        tensor_single_level.squeeze(2),
        dims=["time", "vars", "latitude", "longitude"],
        coords=dict(
            vars=varname_single_level,
            time=[forecast_datetime],
            latitude=lat,
            longitude=lon,
        ),
    )
    return darray_single_level


def save_netcdf_increment(
    darray_upper_air: xr.DataArray,
    darray_single_level: xr.DataArray,
    nc_filename: str,
    forecast_hour: int,
    meta_data: dict,
    conf: dict,
):
    """
    Save CREDIT model prediction output to netCDF file. Also performs pressure level
    interpolation on the output if you wish.

    Args:
        darray_upper_air (xr.DataArray): upper air variable predictions
        darray_single_level (xr.DataArray): surface variable predictions
        nc_filename (str): file description to go into output filenames
        forecast_hour (int):  how many hours since the initialization of the model.
        meta_data (dict): metadata dictionary for output variables
        conf (dict): configuration dictionary for training and/or rollout

    """
    try:
        """
        Save increment to a unique NetCDF file using Dask for parallel processing.
        """
        # Convert DataArrays to Datasets
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_single = darray_single_level.to_dataset(dim="vars")

        # Merge datasets
        ds_merged = xr.merge([ds_upper, ds_single])

        # Add forecast_hour coordinate
        ds_merged["forecast_hour"] = forecast_hour

        # Add CF convention version
        ds_merged.attrs["Conventions"] = "CF-1.11"

        sig = signature(full_state_pressure_interpolation)
        pres_end = sig.parameters["pres_ending"].default
        height_end = sig.parameters["height_ending"].default
        if "interp_pressure" in conf["predict"].keys():
            if "surface_geopotential_var" in conf["predict"]["interp_pressure"].keys():
                surface_geopotential_var = conf["predict"]["interp_pressure"]["surface_geopotential_var"]
            else:
                surface_geopotential_var = "Z_GDS4_SFC"
            if "pres_ending" in conf["predict"]["interp_pressure"]:
                pres_end = conf["predict"]["interp_pressure"]["pres_ending"]
            if "height_ending" in conf["predict"]["interp_pressure"]:
                height_end = conf["predict"]["interp_pressure"]["height_ending"]

            with xr.open_dataset(conf["predict"]["static_fields"]) as static_ds:
                surface_geopotential = static_ds[surface_geopotential_var].values
            pressure_interp = full_state_pressure_interpolation(ds_merged, surface_geopotential, **conf["predict"]["interp_pressure"])

            # Do ptype here before merging!
            if "use_ptype" in conf.keys() and conf["use_ptype"]:
                credit_processor = CreditPostProcessor()
                ds_output = credit_processor.dewpoint_temp(pressure_interp)
                subset_array = credit_processor.extract_variable_levels(ds_output)

                scaler, input_features = credit_processor.load_scaler(conf["ptype"]["input_scaler_file"])
                transformed_data = credit_processor.transform_data(subset_array, scaler, input_features)
                ptype_model = credit_processor.load_model(conf["ptype"]["ML_model_path"])

                predictions = ptype_model.predict(
                    transformed_data,
                    conf["ptype"]["output_uncertainties"],
                    batch_size=conf["ptype"]["predict_batch_size"],
                )

                gridded_preds = credit_processor.grid_predictions(
                    data=ds_output,
                    predictions=predictions,
                    output_uncertainties=conf["ptype"]["output_uncertainties"],
                )
                ptype_classification = credit_processor.ptype_classification(gridded_preds)

                # check for overlapping variables and remove them
                overlapping_vars = [var for var in ptype_classification.data_vars if var in pressure_interp.data_vars]

                if overlapping_vars:
                    pressure_interp = pressure_interp.drop_vars(overlapping_vars)

                pressure_interp = xr.merge([pressure_interp, ptype_classification])

        ds_merged = xr.merge([ds_merged, pressure_interp])
        logger.info(f"Trying to save forecast hour {forecast_hour} to {nc_filename}")

        save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
        os.makedirs(save_location, exist_ok=True)

        unique_filename = os.path.join(save_location, f"pred_{nc_filename}_{forecast_hour:03d}.nc")
        # ---------------------------------------------------- #
        # If conf['predict']['save_vars'] provided --> drop useless vars
        if "save_vars" in conf["predict"]:
            if len(conf["predict"]["save_vars"]) > 0:
                ds_merged = drop_var_from_dataset(ds_merged, conf["predict"]["save_vars"])

        # when there's no metafile --> meta_data = False
        if meta_data is not False:
            # Add metadata attributes to every model variable if available
            for var in ds_merged.variables.keys():
                if var in meta_data.keys():
                    if var != "time":
                        # use attrs.update for non-datetime variables
                        ds_merged[var].attrs.update(meta_data[var])
                    else:
                        # use time.encoding for datetime variables/coords
                        for metadata_time in meta_data["time"]:
                            ds_merged.time.encoding[metadata_time] = meta_data["time"][metadata_time]
                if "interp_pressure" in conf["predict"].keys():
                    if pres_end in var:
                        var_short = var.strip(pres_end)
                        if var_short in meta_data.keys():
                            ds_merged[var].attrs.update(meta_data[var_short])
                            ds_merged[var].attrs["long_name"] += " (interpolated to isobaric levels)"
                    elif height_end in var:
                        var_short = var.strip(height_end)
                        if var_short in meta_data.keys():
                            ds_merged[var].attrs.update(meta_data[var_short])
                            ds_merged[var].attrs["long_name"] += " (interpolated to constant height AGL levels)"
        encoding_dict = {}
        if "ua_var_encoding" in conf["predict"].keys():
            for ua_var in conf["data"]["variables"]:
                encoding_dict[ua_var] = conf["predict"]["ua_var_encoding"]
        if "surface_var_encoding" in conf["predict"].keys():
            for surface_var in conf["data"]["surface_variables"]:
                encoding_dict[surface_var] = conf["predict"]["surface_var_encoding"]
        if "pressure_var_encoding" in conf["predict"].keys():
            for pres_var in conf["data"]["variables"]:
                encoding_dict[pres_var + pres_end] = conf["predict"]["pressure_var_encoding"]
        if "height_var_encoding" in conf["predict"].keys():
            for height_var in conf["data"]["variables"]:
                encoding_dict[height_var + height_end] = conf["predict"]["height_var_encoding"]
        # Use Dask to write the dataset in parallel
        ds_merged.to_netcdf(unique_filename, mode="w", encoding=encoding_dict)

        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")
    except Exception:
        print(traceback.format_exc())


def save_netcdf_clean(
    darray_upper_air,
    darray_single_level,
    nc_filename,
    forecast_hour,
    meta_data,
    conf,
    use_logger=True,
):
    """
    Save forecast data (upper-air and optionally single-level variables) to a NetCDF file.

    This function is similar to `save_netcdf_increment` but cleaned-up all the interpolations

    Parameters
    ----------
    darray_upper_air : xr.DataArray
        Upper-air forecast data with dimension "vars".
    darray_single_level : xr.DataArray or None
        Single-level forecast data with dimension "vars". If None, only upper-air
        variables are included.
    nc_filename : str
        Base filename (used as subdirectory and prefix for the saved file).
    forecast_hour : int
        Forecast lead time in hours.
    meta_data : dict or bool
        Dictionary containing variable metadata attributes.
        If `False`, no metadata is applied and default time encoding is used.
    conf : dict
        Configuration dictionary. Must include at least:
        - conf["predict"]["save_forecast"]: directory where forecasts are saved.
        - conf["predict"]["save_vars"]: (optional) list of variable names to keep.
    use_logger : bool, optional
        If True (default), configure logging and print progress messages.

    Notes
    -----
    - Files are saved into:
        ``{conf['predict']['save_forecast']}/{nc_filename}/pred_{nc_filename}_{forecast_hour:03d}.nc``

    - If `meta_data` is provided, variable attributes are updated accordingly.
      Otherwise, a default gregorian calendar encoding is applied to the "time" variable.

    - If `conf['predict']['save_vars']` is non-empty, variables not listed there
      are dropped before saving.

    Returns
    -------
    None
        The function saves a NetCDF file to disk and does not return a value.
    """

    if use_logger:
        logging.basicConfig(level=logging.INFO)  # ensures this process logs
        logger = logging.getLogger(__name__)

    # if no single-level field
    if darray_single_level is None:
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_merged = ds_upper
    else:
        # merge upper-air and single-level
        ds_upper = darray_upper_air.to_dataset(dim="vars")
        ds_single = darray_single_level.to_dataset(dim="vars")
        ds_merged = xr.merge([ds_upper, ds_single])

    # add forecast_hour coordinate
    ds_merged["forecast_hour"] = forecast_hour

    if use_logger:
        logger.info(f"Process forecast hour {forecast_hour} to {nc_filename}")

    save_location = os.path.join(conf["predict"]["save_forecast"], nc_filename)
    os.makedirs(save_location, exist_ok=True)

    unique_filename = os.path.join(save_location, f"pred_{nc_filename}_{forecast_hour:03d}.nc")

    # ---------------------------------------------------- #
    # If conf['predict']['save_vars'] provided --> drop useless vars
    if "save_vars" in conf["predict"]:
        if len(conf["predict"]["save_vars"]) > 0:
            ds_merged = drop_var_from_dataset(ds_merged, conf["predict"]["save_vars"])

    # ---------------------------------------------------- #
    # handle meta data and time encoding
    encoding_dict = {}

    if meta_data is not False:
        # Add metadata attributes to every model variable if available
        for var in ds_merged.variables:
            if var in meta_data.keys():
                if var != "time":
                    # use attrs.update for non-datetime variables
                    ds_merged[var].attrs.update(meta_data[var])
                else:
                    # use time.encoding for datetime variables/coords
                    for metadata_time in meta_data["time"]:
                        ds_merged.time.encoding[metadata_time] = meta_data["time"][metadata_time]
    else:
        # if not metadata available, apply time encoding based on gregorian calendar
        time_encoding = {
            "units": "hours since 1900-01-01 00:00:00",
            "calendar": "gregorian",
        }

        encoding_dict = {"time": time_encoding}

    # ---------------------------------------------------- #
    # save
    if use_logger:
        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")

    ds_merged.to_netcdf(unique_filename, mode="w", encoding=encoding_dict)


def save_netcdf_diag(
    darray_single_level: xr.DataArray,
    nc_foldername: str,
    nc_filename: str,
    forecast_hour: int,
    meta_data: dict,
    conf: dict,
    use_logger=True,
):
    if use_logger:
        logging.basicConfig(level=logging.INFO)  # ensures this process logs
        logger = logging.getLogger(__name__)

    # Convert DataArrays to Datasets
    ds_single = darray_single_level.to_dataset(dim="vars")

    ds_merged = ds_single
    ds_merged["forecast_hour"] = forecast_hour
    ds_merged.attrs["Conventions"] = "CF-1.11"

    if use_logger:
        logger.info(f"Process forecast hour {forecast_hour} to {nc_filename}")

    save_location = os.path.join(conf["predict"]["save_forecast"], nc_foldername)
    os.makedirs(save_location, exist_ok=True)

    unique_filename = os.path.join(
        save_location,
        f"pred_{nc_filename}.nc",  # _{forecast_hour:03d}.nc"
    )
    # ---------------------------------------------------- #
    # If conf['predict']['save_vars'] provided --> drop useless vars
    if "save_vars" in conf["predict"]:
        if len(conf["predict"]["save_vars"]) > 0:
            ds_merged = drop_var_from_dataset(ds_merged, conf["predict"]["save_vars"])

    # when there's no metafile --> meta_data = False
    encoding_dict = {}

    if meta_data is not False:
        # Add metadata attributes to every model variable if available
        for var in ds_merged.variables:
            if var in meta_data.keys():
                if var != "time":
                    # use attrs.update for non-datetime variables
                    ds_merged[var].attrs.update(meta_data[var])
                else:
                    # use time.encoding for datetime variables/coords
                    for metadata_time in meta_data["time"]:
                        ds_merged.time.encoding[metadata_time] = meta_data["time"][metadata_time]
    else:
        time_encoding = {
            "units": "hours since 1900-01-01 00:00:00",
            "calendar": "gregorian",
        }
        encoding_dict = {"time": time_encoding}

    # ---------------------------------------------------- #
    # save
    if use_logger:
        logger.info(f"Saved forecast hour {forecast_hour} to {unique_filename}")

    ds_merged.to_netcdf(unique_filename, mode="w", encoding=encoding_dict)



def save_corrdiff_ensemble(
    ensemble_phys,
    init_time,
    south_north,
    west_east,
    conf,
    meta_data=False,
    use_logger=True,
):
    """Save a CorrDiff ensemble forecast as a single NetCDF with member dim.

    All ensemble members are concatenated along a new 'member' dimension,
    producing one file per initialization time.  Each variable has shape
    (member, time, south_north, west_east).

    Output path:
        {save_forecast}/pred_{init_str}.nc

    Parameters
    ----------
    ensemble_phys : torch.Tensor or np.ndarray
        Inverse-transformed ensemble of shape (N_members, C_total, T, H, W),
        in physical units.  Channel order must match the order in which
        upper-air and surface variables were concatenated during inference
        (upper-air first, then surface, matching the training convention).
    init_time : datetime, np.datetime64, or pd.Timestamp
        Initialization timestamp for this forecast.
    south_north : np.ndarray
        Y-axis grid coordinates.
    west_east : np.ndarray
        X-axis grid coordinates.
    conf : dict
        Configuration dict.  Must contain:
          - conf["data"]["variables"]:        upper-air variable names
          - conf["data"]["surface_variables"]: surface variable names
          - conf["data"]["levels"]:           number of upper-air levels
          - conf["predict"]["save_forecast"]: output directory
        Optional:
          - conf["predict"]["save_vars"]:     whitelist of vars to keep
    meta_data : dict or False
        Variable metadata (passed through to the encoder).
    use_logger : bool
        Log progress to stdout.
    """
    # ---- Normalize the ensemble into a numpy array ---- #
    if isinstance(ensemble_phys, torch.Tensor):
        ensemble_phys = ensemble_phys.detach().cpu().numpy()

    if ensemble_phys.ndim != 5:
        raise ValueError(
            f"ensemble_phys must have shape (N_members, C, T, H, W); "
            f"got shape {ensemble_phys.shape}"
        )

    n_members, n_channels, n_time, ny, nx = ensemble_phys.shape

    # ---- Build channel-name list matching the model's output order ---- #
    upper_vars = conf["data"].get("variables", [])
    surf_vars = conf["data"].get("surface_variables", [])
    n_levels = conf["data"].get("levels", 1)

    upper_channels = []
    for v in upper_vars:
        if n_levels == 1:
            upper_channels.append(v)
        else:
            for k in range(n_levels):
                upper_channels.append(f"{v}_{k}")

    channel_names = upper_channels + list(surf_vars)

    if len(channel_names) != n_channels:
        raise ValueError(
            f"Channel count mismatch: ensemble has {n_channels} channels, "
            f"but config lists {len(channel_names)} variable names: "
            f"{channel_names}"
        )

    # ---- Normalize init_time to datetime64 and build init_str ---- #
    if isinstance(init_time, datetime):
        init_dt64 = np.datetime64(init_time)
    elif isinstance(init_time, np.datetime64):
        init_dt64 = init_time
    else:
        init_dt64 = np.datetime64(init_time)

    init_str = (
        init_dt64.astype("datetime64[h]")
        .astype(datetime)
        .strftime("%Y%m%d%H")
    )

    # ---- Output path: one file per init time ---- #
    save_dir = os.path.expandvars(conf["predict"]["save_forecast"])
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"pred_{init_str}.nc")

    # ---- Time coordinate ---- #
    time_values = np.array([init_dt64])
    if n_time > 1:
        time_values = init_dt64 + np.arange(n_time) * np.timedelta64(1, "h")

    # ---- Build a single Dataset with all members concatenated ---- #
    # Each variable has shape (member, time, south_north, west_east).
    # Slicing ensemble_phys[:, i_ch] yields (member, time, ny, nx).
    member_coord = np.arange(n_members, dtype=np.int32)

    data_vars = {}
    for i_ch, vname in enumerate(channel_names):
        data_vars[vname] = xr.DataArray(
            ensemble_phys[:, i_ch],   # (member, time, ny, nx)
            dims=("member", "time", "south_north", "west_east"),
            coords={
                "member": member_coord,
                "time": time_values,
                "south_north": south_north,
                "west_east": west_east,
            },
        )

    ds = xr.Dataset(data_vars)
    ds.attrs["init_time"] = str(init_dt64)
    ds.attrs["n_ensemble"] = n_members
    ds.attrs["description"] = "CorrDiff ensemble forecast with member dimension"

    # ---- Optional metadata pass-through (mirrors save_netcdf_clean) ---- #
    encoding_dict = {}
    if meta_data and meta_data is not False:
        for v in ds.variables:
            if v in meta_data and v != "time":
                ds[v].attrs.update(meta_data[v])
            elif v == "time" and "time" in meta_data:
                for k, val in meta_data["time"].items():
                    ds.time.encoding[k] = val
    else:
        encoding_dict["time"] = {
            "units": "hours since 1900-01-01 00:00:00",
            "calendar": "gregorian",
        }

    # ---- Drop variables not in save_vars whitelist (if set) ---- #
    if conf["predict"].get("save_vars"):
        keep_vars = conf["predict"]["save_vars"]
        drop_these = [v for v in ds.data_vars if v not in keep_vars]
        if drop_these:
            ds = ds.drop_vars(drop_these)

    # ---- Optional per-variable compression to keep files manageable ---- #
    # ~16 members × 7 vars × 1 time × 336 × 336 × 4 bytes ≈ 50 MB uncompressed.
    # With zlib compression, typically 8-15 MB.
    for v in ds.data_vars:
        encoding_dict.setdefault(v, {}).update({
            "zlib": True,
            "complevel": 4,    # mild compression; bigger numbers give marginal gains
        })

    # ---- Write ---- #
    ds.to_netcdf(out_path, mode="w", encoding=encoding_dict)

    if use_logger:
        file_size_mb = os.path.getsize(out_path) / 1e6
        logger.info(
            f"Saved ensemble to {out_path} "
            f"({n_members} members, {len(ds.data_vars)} vars, {file_size_mb:.2f} MB)"
        )

