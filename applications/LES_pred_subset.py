import os
import gc
import sys
import yaml
import logging
import warnings
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from collections import defaultdict

# ---------- #
# Numerics
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import pandas as pd

# ---------- #
import torch
from torchvision import transforms as tforms

# ---------- #
# credit
from credit.models import load_model
from credit.seed import seed_everything
from credit.distributed import get_rank_info

from credit.data import (
    concat_and_reshape,
    reshape_only,
    get_forward_data,
    drop_var_from_dataset,
    extract_month_day_hour,
    find_common_indices,
    next_n_hour,
    previous_hourly_steps,
    encode_datetime64,
    filter_ds,
)

from credit.datasets.wrf_singlestep import WRF_Predict
from credit.transforms.transforms_wrf import Normalize_WRF, ToTensor_WRF

from credit.pbs import launch_script, launch_script_mpi
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state
from credit.parser import credit_main_parser, predict_data_check
from credit.output import load_metadata, make_xarray, save_netcdf_clean
from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ---- cudnn global settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # good if H,W are fixed

# if PyTorch ≥ 2.0
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def predict(rank, world_size, conf, p):
    # ======================================================== #
    # load pytorch model
    # -------------------------------------------------------- #

    # flag for distributed inference
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]

    # set prediction device
    # -------------------------------------------------------- #
    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["predict"]["mode"])

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    # -------------------------------------------------------- #
    # main loading block
    if conf["predict"]["mode"] == "none":
        model = load_model(conf, load_weights=True).to(device)

    elif conf["predict"]["mode"] == "ddp":
        model = load_model(conf).to(device)
        # if conf["trainer"].get("compile", False):
        #     model = torch.compile(model)
        model = distributed_model_wrapper(conf, model, device)
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)
        load_msg = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        load_state_dict_error_handler(load_msg)

    elif conf["predict"]["mode"] == "fsdp":
        model = load_model(conf, load_weights=True).to(device)
        model = distributed_model_wrapper(conf, model, device)
        # Load model weights (if any), an optimizer, scheduler, and gradient scaler
        model = load_model_state(conf, model, device)
    # -------------------------------------------------------- #
    # set to eval
    model.eval()

    # ======================================================== #
    # DATA: data normalization process
    # -------------------------------------------------------- #
    state_transformer = Normalize_WRF(conf)
    to_tensor_scaler = ToTensor_WRF(conf)
    transforms = tforms.Compose([state_transformer, to_tensor_scaler])

    # ======================================================== #
    # DATA: load information from conf
    # -------------------------------------------------------- #
    ind_start = conf["predict"]["forecasts"]["start_ind"]
    N_steps = conf["predict"]["forecasts"]["pred_step"]
    test_years_range = conf["predict"]["forecasts"]["year_range"]

    # random seed
    seed = conf["seed"]
    seed_everything(seed)

    # length of forecast steps (e.g., hourly)
    lead_time_periods = conf["data"]["lead_time_periods"]

    # number of diagnostic variables (e.g., 0)
    varnum_diag = len(conf["data"]["diagnostic_variables"])

    # number of dynamic forcing + forcing + static (e.g., 0)
    static_dim_size = len(conf["data"]["dynamic_forcing_variables"]) + len(conf["data"]["forcing_variables"]) + len(conf["data"]["static_variables"])

    # bool flag for each variable type
    flag_dyn_forcing = ("dynamic_forcing_variables" in conf["data"]) and (len(conf["data"]["dynamic_forcing_variables"]) > 0)
    flag_forcing = ("forcing_variables" in conf["data"]) and (len(conf["data"]["forcing_variables"]) > 0)
    flag_static = ("static_variables" in conf["data"]) and (len(conf["data"]["static_variables"]) > 0)

    # if multiple of static, forcing, dynamic forcing exists, create a bool flag to set their order
    if flag_forcing or flag_static:
        # ======================================================================================== #
        # forcing variable first (new models) vs. static variable first (some old models)
        # this flag makes sure that the class is compatible with some old CREDIT models
        flag_static_first = ("static_first" in conf["data"]) and (conf["data"]["static_first"])
        # ======================================================================================== #
    else:
        has_forcing_static = False

    # ======================================================== #
    # DATA: select relavant data files based on the year
    # -------------------------------------------------------- #
    # upper air files
    upper_files = sorted(glob(conf["data"]["save_loc"]))
    upper_files_outside = sorted(glob(conf["data"]["boundary"]["save_loc"]))

    # --------------- #
    # surface files
    if ("surface_variables" in conf["data"]) and (len(conf["data"]["surface_variables"]) > 0):
        list_surf_ds = sorted(glob(conf["data"]["save_loc_surface"]))
    else:
        list_surf_ds = None

    list_surf_ds_outside = sorted(glob(conf["data"]["boundary"]["save_loc_surface"]))

    # --------------- #
    # dyn forcing files
    if ("dynamic_forcing_variables" in conf["data"]) and (len(conf["data"]["dynamic_forcing_variables"]) > 0):
        list_dyn_forcing_ds = sorted(glob(conf["data"]["save_loc_dynamic_forcing"]))
    else:
        list_dyn_forcing_ds = None

    # convert year info to str for file name search
    test_years = [str(year) for year in range(test_years_range[0], test_years_range[1])]

    # Filter files
    test_files = [file for file in upper_files if any(year in file for year in test_years)]
    test_files_outside = [file for file in upper_files_outside if any(year in file for year in test_years)]

    if list_surf_ds is not None:
        test_list_surf_ds = [file for file in list_surf_ds if any(year in file for year in test_years)]
    else:
        test_list_surf_ds = None

    test_list_surf_ds_outside = [file for file in list_surf_ds_outside if any(year in file for year in test_years)]

    if list_dyn_forcing_ds is not None:
        test_list_dyn_forcing_ds = [file for file in list_dyn_forcing_ds if any(year in file for year in test_years)]
    else:
        test_list_dyn_forcing_ds = None

    # -------------------------------------------------------- #
    # summarize selected file name info
    filenames = test_files
    filename_surface = test_list_surf_ds
    filename_dyn_forcing = test_list_dyn_forcing_ds
    # filename_diagnostic = test_list_diag_ds
    filenames_outside = test_files_outside

    # ======================================================== #
    # DATA: open all data as xr.datasets
    # -------------------------------------------------------- #
    # varname info: major domain
    varname_upper_air = conf["data"]["variables"]
    varname_surface = conf["data"]["surface_variables"]
    varname_dyn_forcing = conf["data"]["dynamic_forcing_variables"]
    varname_forcing = conf["data"]["forcing_variables"]
    varname_static = conf["data"]["static_variables"]
    filename_forcing = conf["data"]["save_loc_forcing"]
    filename_static = conf["data"]["save_loc_static"]
    # -------------------------------------------------------- #
    # varname info: boundary condition
    varname_upper_air_outside = conf["data"]["boundary"]["variables"]
    varname_surface_outside = conf["data"]["boundary"]["surface_variables"]
    filename_surface_outside = test_list_surf_ds_outside
    history_len_outside = conf["data"]["boundary"]["history_len"]
    forecast_len_outside = conf["data"]["boundary"]["forecast_len"]
    lead_time_periods_outside = conf["data"]["boundary"]["lead_time_periods"]

    # time info
    history_len = conf["data"]["history_len"]
    assert history_len == 1, 'only conf["data"]["history_len"] = 1 is supported for this application'

    # -------------------------------------------------------- #
    # open data: major domain
    ds_domain = get_forward_data(conf["loss"]["latitude_weights"])
    meta_data = load_metadata(conf)

    list_upper_ds = []
    list_surf_ds = []
    list_dyn_forcing_ds = []

    # upper‐air
    initial_ds = filter_ds(get_forward_data(filenames[0]), varname_upper_air).isel(time=slice(ind_start, ind_start + history_len))

    # surface
    if filename_surface:
        surf_ds = filter_ds(get_forward_data(filename_surface[0]), varname_surface).isel(time=slice(ind_start, ind_start + history_len))
        initial_ds = xr.merge([initial_ds, surf_ds])
    else:
        surf_ds = False

    # dynamic forcing
    if filename_dyn_forcing:
        list_dyn_forcing_ds = [filter_ds(ds, varname_dyn_forcing) for ds in all_ds]
        # concat multi-year ds to one
        dyn_forcing_ds = xr.concat(list_dyn_forcing_ds, dim="time")
        # also merge the first ds to initial_ds
        initial_ds = xr.merge([initial_ds, list_dyn_forcing_ds[0]])
    else:
        list_dyn_forcing_ds = False

    # forcing
    if filename_forcing is not None:
        # drop variables if they are not in the config
        ds = get_forward_data(filename_forcing)
        xarray_forcing = drop_var_from_dataset(ds, varname_forcing).load()
    else:
        xarray_forcing = False

    # static
    if filename_static is not None:
        # drop variables if they are not in the config
        ds = get_forward_data(filename_static)
        xarray_static = drop_var_from_dataset(ds, varname_static).load()
        xarray_static = xarray_static.expand_dims(dim={"time": len(initial_ds["time"])})
    else:
        xarray_static = False

    # -------------------------------------------------------- #
    # open data: boundary condition
    list_upper_ds_outside = []
    list_surf_ds_outside = []

    for fn_outside in filenames_outside:
        # drop variables if they are not in the config
        ds_outside = get_forward_data(filename=fn_outside)
        ds_upper_outside = drop_var_from_dataset(ds_outside, varname_upper_air_outside)

        if filename_surface_outside is not None:
            ds_surf_outside = drop_var_from_dataset(ds_outside, varname_surface_outside)
            list_surf_ds_outside.append(ds_surf_outside)
        else:
            list_surf_ds_outside = False

        list_upper_ds_outside.append(ds_upper_outside)

    # -------------------------------------------------------------------------- #
    # get sample indices from boundary upper-air files:
    outside_file_year_range = [
        int(np.datetime_as_string(list_upper_ds_outside[0]["time"][0].values, unit="Y")),
        int(np.datetime_as_string(list_upper_ds_outside[-1]["time"][0].values, unit="Y")),
    ]

    outside_file_indices = {}  # <------ change
    for ind_file, outside_file_xarray in enumerate(list_upper_ds_outside):
        outside_file_indices[str(ind_file)] = outside_file_xarray["time"].values

    # ======================================================== #
    # DATA: summarize other relavent conf info
    # -------------------------------------------------------- #
    initial_time = initial_ds["time"].values[0]

    # ---------------------------- #
    # has leap year
    # fcst_timesteps = np.arange(
    #     initial_time,
    #     initial_time + (N_steps + 1) * np.timedelta64(lead_time_periods, "h"),
    #     np.timedelta64(lead_time_periods, "h"),
    # )
    # ---------------------------- #
    # no leap year
    step_hours = f'{lead_time_periods}H'
    end_time = initial_time + (N_steps) * np.timedelta64(lead_time_periods, "h")
    idx = pd.date_range(start=initial_time, end=end_time, freq=step_hours)
    idx_noleap = idx[~((idx.month == 2) & (idx.day == 29))]
    fcst_timesteps = idx_noleap.values.astype('datetime64[ns]')
    
    init_datetime = datetime.utcfromtimestamp(initial_time.astype("datetime64[s]").astype(int))
    init_datetime_str = init_datetime.strftime("%Y-%m-%dT%HZ")

    # ======================================================== #
    # prediction section
    # -------------------------------------------------------- #
    for i_step in range(1, 1 + N_steps, 1):
        results = []

        # -------------------------------------------------------- #
        # pull boundary condition based on the forecasted time
        time_boundary = fcst_timesteps[i_step]
        #time_round = next_n_hour(time_boundary, lead_time_periods_outside)
        if lead_time_periods_outside == 1 and history_len_outside == 1:
            time_round = time_boundary
        else:
            time_round = next_n_hour(time_boundary, lead_time_periods_outside)

        if history_len_outside == 1:
            time_year = int(np.datetime_as_string(time_round, unit="Y"))
            ind_year = time_year - outside_file_year_range[0]
            ind_date = np.searchsorted(outside_file_indices[str(ind_year)], time_round)
            ds_upper_outside = list_upper_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
            ds_surf_outside = list_surf_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1))
            ds_outside = xr.merge([ds_upper_outside, ds_surf_outside])

        else:
            list_ds_upper_outside_slice = []
            list_ds_surf_outside_slice = []

            for i_time_backward in range(history_len_outside):
                time_round_loop = previous_hourly_steps(time_round, lead_time_periods_outside, i_time_backward)
                time_year = int(np.datetime_as_string(time_round_loop, unit="Y"))
                ind_year = time_year - outside_file_year_range[0]
                ind_date = np.searchsorted(outside_file_indices[str(ind_year)], time_round_loop)
                list_ds_upper_outside_slice.append(list_upper_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1)))
                list_ds_surf_outside_slice.append(list_surf_ds_outside[ind_year].isel(time=slice(ind_date, ind_date + 1)))

            ds_upper_outside = xr.concat(list_ds_upper_outside_slice[::-1], dim="time")  # ::-1 so the latest time is the last
            ds_surf_outside = xr.concat(list_ds_surf_outside_slice[::-1], dim="time")
            ds_outside = xr.merge([ds_upper_outside, ds_surf_outside])

        t0 = [
            fcst_timesteps[i_step - 1],
        ]
        t1 = [
            fcst_timesteps[i_step],
        ]
        t2 = ds_outside["time"].values
        time_encode = encode_datetime64(np.concatenate([t0, t1, t2]))

        # -------------------------------------------------------- #
        # add forcing & static to initial conditions

        # static field
        if filename_static is not None:
            xarray_static["time"] = initial_ds["time"]
            initial_ds = xr.merge([initial_ds, xarray_static])

        # forcing field gen
        if filename_forcing is not None:
            month_day_forcing = extract_month_day_hour(np.array(xarray_forcing["time"]))
            month_day_inputs = extract_month_day_hour(np.array(fcst_timesteps[i_step - 1]))
            # indices to subset
            ind_forcing, _ = find_common_indices(month_day_forcing, month_day_inputs)
            xarray_forcing = xarray_forcing.isel(time=ind_forcing)
            # forcing field
            xarray_forcing["time"] = initial_ds["time"]
            initial_ds = initial_ds.merge(xarray_forcing)

        # -------------------------------------------------------- #
        # main prediction loop
        # -------------------------------------------------------- #
        # the first prediction step, use initialization directly
        if i_step == 1:
            x = initial_ds

            sample_x = {
                "WRF_input": x,
                "boundary_input": ds_outside,
                "time_encode": time_encode,
            }

            batch = batch_initial = transforms(sample_x)

            if "x_surf" in batch:
                # combine x and x_surf
                # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                x = concat_and_reshape(batch["x"][None, ...], batch["x_surf"][None, ...]).to(device)
            else:
                # no x_surf
                x = reshape_only(batch["x"][None, ...]).to(device)

            # add forcing and static variables (regardless of fcst hours)
            if "x_forcing_static" in batch:
                # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                x_forcing_batch = batch["x_forcing_static"][None, ...].to(device).permute(0, 2, 1, 3, 4)

                # concat on var dimension
                x = torch.cat((x, x_forcing_batch), dim=1)

        # -------------------------------------------------------- #
        # rolling steps, use the previous output as input
        else:
            sample_x = {"boundary_input": ds_outside, "time_encode": time_encode}

            batch = transforms(sample_x)

            # not the first step, y_pred exist
            # y_pred = state_transformer.transform_array(y_pred) #.to(device)

            # ============================================================ #
            # prepare x
            # ------------------------------------------------------------ #
            # use previous step y_pred as the next step x
            if history_len == 1:
                # cut diagnostic vars from y_pred, they are not inputs
                if "y_diag" in batch:
                    x = y_pred[:, :-varnum_diag, ...].detach()
                else:
                    x = y_pred.detach()
                # TO DO: concat dynamic forcing

            # multi-step in
            else:
                if static_dim_size == 0:
                    x_detach = x[:, :, 1:, ...].detach()
                else:
                    x_detach = x[:, :-static_dim_size, 1:, ...].detach()

                # cut diagnostic vars from y_pred, they are not inputs
                if "y_diag" in batch:
                    x = torch.cat([x_detach, y_pred[:, :-varnum_diag, ...].detach()], dim=2)
                else:
                    x = torch.cat([x_detach, y_pred.detach()], dim=2)

            # ------------------------------------------------------------ #
            # add static, forcing, dynamic forcing, if any, to x

            # if static only, pull static tensor from the initial condition
            if flag_static and not flag_forcing and not flag_dyn_forcing:
                if "x_forcing_static" in batch_initial:
                    # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                    x_forcing_batch = batch_initial["x_forcing_static"][None, ...].to(device).permute(0, 2, 1, 3, 4)

                    # concat on var dimension
                    x = torch.cat((x, x_forcing_batch), dim=1)

            # a more general solution if forcing or dynmaic forcing are invloved
            elif flag_static or flag_forcing or flag_dyn_forcing:
                # define rolling_ds to host static, forcing, dynamic forcing
                rolling_ds = xr.Dataset(
                    coords={"time": ("time", np.array([fcst_timesteps[i_step - 1],]),)}
                )

                # ------------------------------------------------------ #
                # merge static and forcing to rolling_ds
                if flag_static:
                    xarray_static["time"] = rolling_ds["time"]
                    rolling_ds = xr.merge([rolling_ds, xarray_static])

                if flag_forcing:
                    xarray_forcing = rolling_ds["time"]
                    rolling_ds = xr.merge([rolling_ds, xarray_forcing])

                if flag_dyn_forcing:
                    dyn_forcing_subset = dyn_forcing_ds.isel(time=slice(i_step - 1, i_step))
                    rolling_ds = xr.merge([rolling_ds, dyn_forcing_subset])

                # ------------------------------------------------------ #
                # xarray --> np.array --> tensor
                if flag_static_first:
                    varname_forcing_static = varname_static + varname_dyn_forcing + varname_forcing
                else:
                    varname_forcing_static = varname_dyn_forcing + varname_forcing + varname_static

                list_vars_forcing_static = []
                for var_name in varname_forcing_static:
                    var_value = rolling_ds[var_name].values
                    list_vars_forcing_static.append(var_value)
                numpy_vars_forcing_static = np.array(list_vars_forcing_static)

                x_static = torch.as_tensor(numpy_vars_forcing_static).squeeze()

                if len(x_static.shape) == 4:
                    # permute: [forcing_var, time, lat, lon] --> [time, forcing_var, lat, lon]
                    x_static = x_static.permute(1, 0, 2, 3)

                elif len(x_static.shape) == 3:
                    if self.num_forcing_static > 1:
                        # single time, multi-vars
                        x_static = x_static.unsqueeze(0)
                    else:
                        # multi-time, single vars
                        x_static = x_static.unsqueeze(1)
                else:
                    # num_var=1, time=1, only has lat, lon
                    x_static = x_static.unsqueeze(0).unsqueeze(0)
                    # x_static = x_static.unsqueeze(1)

                # ------------------------------------------------------ #
                # concat to x
                x_forcing_batch = x_static[None, ...].to(device).permute(0, 2, 1, 3, 4)
                x = torch.cat((x, x_forcing_batch), dim=1)

        # --------------------------------------------------------------------------------- #
        # boundary conditions
        if "x_surf_boundary" in batch:
            x_boundary = concat_and_reshape(batch["x_boundary"][None, ...], batch["x_surf_boundary"][None, ...]).to(device)
        else:
            x_boundary = reshape_only(batch["x_boundary"][None, ...]).to(device)

        # --------------------------------------------------------------------------------- #
        # time encoding
        x_time_encode = batch["x_time_encode"][None, ...].to(device)

        # # -------------------------------------------------------------------------------------- #
        # # start prediction
        y_pred = model(x, x_boundary, x_time_encode)

        y_pred_save = state_transformer.inverse_transform(y_pred.cpu()).detach()

        utc_datetime = init_datetime + timedelta(hours=lead_time_periods * i_step)

        # convert the current step result as x-array
        darray_upper_air, darray_single_level = make_xarray(
            y_pred_save,
            utc_datetime,
            ds_domain["south_north"].values,
            ds_domain["west_east"].values,
            conf,
        )

        # Save the current forecast hour data in parallel
        result = p.apply_async(
            save_netcdf_clean,
            (
                darray_upper_air,
                darray_single_level,
                init_datetime_str,
                lead_time_periods * i_step,
                meta_data,
                conf,
            ),
        )
        results.append(result)

        # release GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        if i_step == N_steps:
            # Wait for all processes to finish in order
            for result in results:
                result.get()

            # # forecast count = a constant for each run
            # forecast_count += 1

            # # y_pred allocation
            # y_pred = None

            gc.collect()

            if distributed:
                torch.distributed.barrier()

    if distributed:
        torch.distributed.barrier()

    return 1


if __name__ == "__main__":
    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l, -w
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )

    parser.add_argument(
        "-w",
        "--world-size",
        type=int,
        default=4,
        help="Number of processes (world size) for multiprocessing",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=0,
        help="Update the config to use none, DDP, or FSDP",
    )

    parser.add_argument(
        "-nd",
        "--no-data",
        type=str,
        default=0,
        help="If set to True, only pandas CSV files will we saved for each forecast",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=int,
        default=False,
        help="Predict on subset X of forecasts",
    )
    parser.add_argument(
        "-ns",
        "--no_subset",
        type=int,
        default=False,
        help="Break the forecasts list into X subsets to be processed by X GPUs",
    )
    parser.add_argument(
        "-cpus",
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPU workers to use per GPU",
    )

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    no_data = 0 if "no-data" not in args_dict else int(args_dict.pop("no-data"))
    subset = int(args_dict.pop("subset"))
    number_of_subsets = int(args_dict.pop("no_subset"))
    num_cpus = int(args_dict.pop("num_cpus"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # ======================================================== #
    # handling config args
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=False)
    # predict_data_check(conf, print_summary=False)

    # ======================================================== #

    # create a save location for rollout
    # ---------------------------------------------------- #
    assert "save_forecast" in conf["predict"], "Missing conf['predict']['save_forecast']"

    forecast_save_loc = conf["predict"]["save_forecast"]
    os.makedirs(forecast_save_loc, exist_ok=True)

    print("Save roll-outs to {}".format(forecast_save_loc))

    # Create a project directory (to save launch.sh and model.yml) if they do not exist
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)

    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logger.info(f"Setting the running mode to {mode}")
        conf["predict"]["mode"] = mode

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    if number_of_subsets > 0:
        forecasts = load_forecasts(conf)
        if number_of_subsets > 0 and subset >= 0:
            subsets = np.array_split(forecasts, number_of_subsets)
            forecasts = subsets[subset - 1]  # Select the subset based on subset_size
            conf["predict"]["forecasts"] = forecasts

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])

    with mp.Pool(num_cpus) as p:
        if conf["predict"]["mode"] in ["fsdp", "ddp"]:  # multi-gpu inference
            _ = predict(world_rank, world_size, conf, p=p)
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

        # Ensure all processes are finished
        p.close()
        p.join()
