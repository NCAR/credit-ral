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
)

from credit.datasets.diag_singlestep import Diag_Predict
from credit.transforms import Normalize_Diag, ToTensor_Diag
from credit.output import load_metadata, make_xarray_diag, save_netcdf_diag

from credit.pbs import launch_script, launch_script_mpi
from credit.metrics import DiagMetrics
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper, setup
from credit.models.checkpoint import load_model_state
from credit.parser import credit_main_parser, predict_data_check

from credit.postblock import GlobalMassFixer, GlobalWaterFixer, GlobalEnergyFixer

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def predict(rank, world_size, conf, p):
    # setup rank and world size for GPU-based rollout
    if conf["predict"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["predict"]["mode"])

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    # config settings
    seed = conf["seed"]
    seed_everything(seed)

    # length of forecast steps
    lead_time_periods = conf["data"]["lead_time_periods"]

    # number of diagnostic variables
    varnum_diag = len(conf["data"]["diagnostic_variables"])

    # number of dynamic forcing + forcing + static
    static_dim_size = len(conf["data"]["dynamic_forcing_variables"]) + len(conf["data"]["forcing_variables"]) + len(conf["data"]["static_variables"])

    # ======================================================== #
    # testing year range
    test_years_range = conf["data"]["test_years"]

    param = {}
    # --------------- #
    # upper air files
    upper_files = sorted(glob(conf["data"]["save_loc"]))

    # --------------- #
    # surface files
    if ("surface_variables" in conf["data"]) and (len(conf["data"]["surface_variables"]) > 0):
        list_surf_ds = sorted(glob(conf["data"]["save_loc_surface"]))
    else:
        list_surf_ds = None

    # --------------- #
    # dyn forcing files
    if ("dynamic_forcing_variables" in conf["data"]) and (len(conf["data"]["dynamic_forcing_variables"]) > 0):
        list_dyn_forcing_ds = sorted(glob(conf["data"]["save_loc_dynamic_forcing"]))
    else:
        list_dyn_forcing_ds = None

    # # --------------- #
    # # diagnostic files
    # if ('diagnostic_variables' in conf['data']) and (len(conf['data']['diagnostic_variables']) > 0):
    #     list_diag_ds = sorted(glob(conf["data"]["save_loc_diagnostic"]))
    # else:
    #     list_diag_ds = None

    # convert year info to str for file name search
    test_years = [str(year) for year in range(test_years_range[0], test_years_range[1])]
    test_files = [file for file in upper_files if any(year in file for year in test_years)]

    if list_surf_ds is not None:
        test_list_surf_ds = [file for file in list_surf_ds if any(year in file for year in test_years)]
    else:
        test_list_surf_ds = None

    if list_dyn_forcing_ds is not None:
        test_list_dyn_forcing_ds = [file for file in list_dyn_forcing_ds if any(year in file for year in test_years)]
    else:
        test_list_dyn_forcing_ds = None

    # if list_diag_ds is not None:
    #     test_list_diag_ds = [file for file in list_diag_ds if any(year in file for year in test_years)]
    # else:
    #     test_list_diag_ds = None

    param["varname_upper_air"] = conf["data"]["variables"]
    param["varname_surface"] = conf["data"]["surface_variables"]
    param["varname_dyn_forcing"] = conf["data"]["dynamic_forcing_variables"]
    param["varname_forcing"] = conf["data"]["forcing_variables"]
    param["varname_static"] = conf["data"]["static_variables"]
    # param['varname_diagnostic'] = conf['data']['diagnostic_variables']
    param["filename_forcing"] = conf["data"]["save_loc_forcing"]
    param["filename_static"] = conf["data"]["save_loc_static"]
    param["filenames"] = test_files
    param["filename_surface"] = test_list_surf_ds
    param["filename_dyn_forcing"] = test_list_dyn_forcing_ds
    # param['filename_diagnostic'] = test_list_diag_ds
    param["history_len"] = conf["data"]["history_len"]
    param["forecast_len"] = conf["data"]["forecast_len"]

    # ================================ #
    # experimental options
    # param['gaussian'] = conf["data"]['gaussian_filter_augmentation']
    
    if 'level_pick' in conf["data"]:
        param['level_pick'] = conf["data"]['level_pick']
    else:
        param['level_pick'] = None
    # ================================ #
    
    # ----------------------------------------------------------------- #

    state_transformer = Normalize_Diag(conf)
    to_tensor_scaler = ToTensor_Diag(conf)
    transforms = tforms.Compose([state_transformer, to_tensor_scaler])  #

    fcst_datetime = load_forecasts(conf)
    for i in range(len(fcst_datetime)):
        t0 = fcst_datetime[i][0]
        t1 = datetime.strptime(t0, "%Y-%m-%d %H:%M:%S")
        t1 = t1 + timedelta(hours=int(conf["data"]["lead_time_periods"]))
        t1 = t1.strftime("%Y-%m-%d %H:%M:%S")
        fcst_datetime[i] = [t0, t1]

    fcst_datetime = fcst_datetime[1:]

    # ----------------------------------------------------------------- #
    # get dataset
    dataset = Diag_Predict(
        param,
        fcst_datetime,
        lead_time_periods=lead_time_periods,
        transform=transforms,
        rank=rank,
        world_size=world_size,
    )

    # setup the dataloder
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # flag for distributed inference
    distributed = conf["predict"]["mode"] in ["ddp", "fsdp"]

    # ================================================================================ #
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
    # ================================================================================ #

    model.eval()

    # get lat/lons from x-array
    ds_domain = get_forward_data(conf["loss"]["latitude_weights"])

    meta_data = load_metadata(conf)

    # # Set up metrics and containers
    # metrics = DiagMetrics(conf)
    # metrics_results = defaultdict(list)

    # Rollout
    with torch.no_grad():
        # forecast count = a constant for each run
        forecast_count = 0

        # y_pred allocation
        results = []

        # model inference loop
        for k, batch in enumerate(data_loader):
            # get the datetime and forecasted hours
            date_time = batch["datetime"].item()
            forecast_hour = batch["forecast_hour"].item()
            # initialization on the first forecast hour
            if forecast_hour == 1:
                # Initialize x and x_surf with the first time step
                if "x_surf" in batch:
                    # combine x and x_surf
                    # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon)
                    # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                    x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device)
                else:
                    # no x_surf
                    x = reshape_only(batch["x"]).to(device)

                init_datetime_obj = datetime.utcfromtimestamp(date_time)
                init_datetime = init_datetime_obj.strftime("%Y-%m-%dT%HZ")
                # -------------------------------------------------------------------------------------- #
                # add forcing and static variables (regardless of fcst hours)
                if "x_forcing_static" in batch:
                    # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                    x_forcing_batch = batch["x_forcing_static"].to(device).permute(0, 2, 1, 3, 4)

                    # concat on var dimension
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # -------------------------------------------------------------------------------------- #
                # y = batch["y_diag"].to(device).to(device).permute(0, 2, 1, 3, 4)

                # --------------------------------------------------------------------------------- #
                # time encoding
                x_time_encode = batch["x_time_encode"].to(device)

                # -------------------------------------------------------------------------------------- #
                # start prediction
                y_pred = model(x, x_time_encode)

                # y_pred with unit
                y_pred = state_transformer.inverse_transform(y_pred.cpu())
                # y_target with unit
                # y = state_transformer.inverse_transform(y.cpu())

                # # Compute metrics
                # metrics_dict = metrics(y_pred, y, forecast_datetime=forecast_hour)

                # for k, m in metrics_dict.items():
                #     metrics_results[k].append(m.item())
                # metrics_results["forecast_hour"].append(forecast_hour)

                # Save the current forecast hour data in parallel
                utc_datetime = init_datetime_obj  # + timedelta(hours=lead_hour)

                # convert the current step result as x-array
                try:
                    darray_single_level = make_xarray_diag(
                        y_pred,
                        utc_datetime,
                        ds_domain["south_north"].values,
                        ds_domain["west_east"].values,
                        conf,
                    )
                except:
                    darray_single_level = make_xarray_diag(
                        y_pred,
                        utc_datetime,
                        ds_domain["latitude"].values,
                        ds_domain["longitude"].values,
                        conf,
                    )

                # Save the current forecast hour data in parallel
                result = p.apply_async(
                    save_netcdf_diag,
                    (
                        darray_single_level,
                        init_datetime,
                        init_datetime,
                        0,
                        meta_data,
                        conf,
                    ),
                )
                results.append(result)

                # metrics_results["datetime"].append(utc_datetime)

                # print_str = f"Forecast: {forecast_count} "
                # print_str += f"Date: {utc_datetime.strftime('%Y-%m-%d %H:%M:%S')} "
                # print_str += f"Hour: {batch['forecast_hour'].item()} "
                # print_str += f"ACC: {metrics_dict['acc']} "

                # Explicitly release GPU memory
                torch.cuda.empty_cache()
                gc.collect()

                # if batch["stop_forecast"][0]:
                # Wait for all processes to finish in order
                for result in results:
                    result.get()

                # # save metrics file
                # save_location = os.path.join(os.path.expandvars(conf["save_loc"]), "forecasts", "metrics")
                # os.makedirs(save_location, exist_ok=True)  # should already be made above
                # df = pd.DataFrame(metrics_results)
                # df.to_csv(os.path.join(save_location, f"metrics{init_datetime}.csv"))

                # forecast count = a constant for each run
                forecast_count += 1

                # y_pred allocation
                y_pred = None

                gc.collect()

                if distributed:
                    torch.distributed.barrier()
            else:
                pass

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
        for i in range(len(fcst_datetime)):
            t0 = fcst_datetime[i][0]
            t1 = datetime.strptime(t0, "%Y-%m-%d %H:%M:%S")
            t1 = t1 + timedelta(hours=int(conf["data"]["lead_time_periods"]))
            t1 = t1.strftime("%Y-%m-%d %H:%M:%S")
            fcst_datetime[i] = [t0, t1]

        fcst_datetime = fcst_datetime[1:]

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
