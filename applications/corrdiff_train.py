"""
train.py
-------------------------------------------------------
Content
    load_dataset_and_sampler
    load_model_states_and_optimizer
    _load_ema_shards_if_present
    main
"""

import os
import sys
import yaml
import copy
import optuna
import shutil
import logging
import warnings
from glob import glob

from pathlib import Path
from argparse import ArgumentParser
from echo.src.base_objective import BaseObjective

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from credit.distributed import distributed_model_wrapper, setup, get_rank_info

from credit.losses.weighted_loss import VariableTotalLoss2D
from credit.datasets.dscale_singlestep import Dscale_Dataset

from credit.seed import seed_everything
from credit.transforms import load_transforms
from credit.scheduler import load_scheduler, annealed_probability
from credit.parser import credit_main_parser, training_data_check
from credit.trainers import load_trainer

from credit.metrics import CorrDiffMetrics
from credit.pbs import launch_script, launch_script_mpi
from credit.models import load_model
from credit.models.checkpoint import (
    FSDPOptimizerWrapper,
    TorchFSDPCheckpointIO,
    load_state_dict_error_handler,
)

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _run_residual_diagnostic(trainer, train_loader, device, conf, n_samples=200):
    """Compute per-channel residual mean and standard deviation.

    Used to calibrate `sigma_data` and check if your residual statistics
    match the EDM default of 0.5.  Runs once at startup, on a small sample
    of batches, then logs the result.
    """
    import logging
    import numpy as np
    import torch as _t
    logger = logging.getLogger(__name__)

    # Only run on rank 0 to avoid noise across processes
    if hasattr(trainer, "rank") and trainer.rank != 0:
        return

    logger.info(f"[RESIDUAL DIAG] Running residual statistics on {n_samples} batches")

    trainer.model.eval()
    sum_r = None
    sum_r_sq = None
    sum_y = None
    sum_y_sq = None
    sum_mu = None
    sum_mu_sq = None
    n_pixels = 0
    n_processed = 0

    with _t.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= n_samples:
                break

            try:
                assembled = trainer._assemble_batch(batch)
                if assembled is None:
                    continue
                cond, r_clean, x_time_encode, y = assembled
            except Exception as e:
                logger.warning(f"[RESIDUAL DIAG] Skipping batch {i}: {e}")
                continue

            n_channels = r_clean.shape[1]
            r_flat = r_clean.permute(1, 0, 2, 3, 4).reshape(n_channels, -1)
            y_flat = y.permute(1, 0, 2, 3, 4).reshape(n_channels, -1)

            # mu corresponds to the first n_channels of cond.
            mu_for_y = cond[:, :n_channels]
            mu_flat = mu_for_y.permute(1, 0, 2, 3, 4).reshape(n_channels, -1)

            if sum_r is None:
                sum_r    = _t.zeros(n_channels, device=device, dtype=_t.float64)
                sum_r_sq = _t.zeros(n_channels, device=device, dtype=_t.float64)
                sum_y    = _t.zeros(n_channels, device=device, dtype=_t.float64)
                sum_y_sq = _t.zeros(n_channels, device=device, dtype=_t.float64)
                sum_mu    = _t.zeros(n_channels, device=device, dtype=_t.float64)
                sum_mu_sq = _t.zeros(n_channels, device=device, dtype=_t.float64)

            sum_r    += r_flat.double().sum(dim=1)
            sum_r_sq += (r_flat.double() ** 2).sum(dim=1)
            sum_y    += y_flat.double().sum(dim=1)
            sum_y_sq += (y_flat.double() ** 2).sum(dim=1)
            sum_mu    += mu_flat.double().sum(dim=1)
            sum_mu_sq += (mu_flat.double() ** 2).sum(dim=1)
            n_pixels += r_flat.shape[1]
            n_processed += 1

    if n_processed == 0:
        logger.warning("[RESIDUAL DIAG] No batches processed; skipping diagnostic")
        trainer.model.train()
        return

    # ---- Move to numpy and compute stats consistently ---- #
    sum_r_np     = sum_r.cpu().numpy()
    sum_r_sq_np  = sum_r_sq.cpu().numpy()
    sum_y_np     = sum_y.cpu().numpy()
    sum_y_sq_np  = sum_y_sq.cpu().numpy()
    sum_mu_np    = sum_mu.cpu().numpy()
    sum_mu_sq_np = sum_mu_sq.cpu().numpy()

    means_r  = sum_r_np  / n_pixels
    means_y  = sum_y_np  / n_pixels
    means_mu = sum_mu_np / n_pixels

    var_r  = (sum_r_sq_np  / n_pixels) - means_r ** 2
    var_y  = (sum_y_sq_np  / n_pixels) - means_y ** 2
    var_mu = (sum_mu_sq_np / n_pixels) - means_mu ** 2

    stds_r  = np.sqrt(np.clip(var_r,  0.0, None))
    stds_y  = np.sqrt(np.clip(var_y,  0.0, None))
    stds_mu = np.sqrt(np.clip(var_mu, 0.0, None))

    # ---- Build channel name list for readable output ---- #
    upper_vars = conf["data"].get("variables", [])
    surf_vars = conf["data"].get("surface_variables", [])
    n_levels = conf["data"].get("levels", 1)

    channel_names = []
    for v in upper_vars:
        if n_levels == 1:
            channel_names.append(v)
        else:
            for k in range(n_levels):
                channel_names.append(f"{v}_{k}")
    channel_names.extend(surf_vars)

    sigma_data = conf["trainer"].get("diffusion", {}).get("sigma_data", 0.5)

    logger.info("=" * 100)
    logger.info("[RESIDUAL DIAG] Per-channel statistics (after normalization)")
    logger.info(f"  Current sigma_data: {sigma_data}")
    logger.info(f"  Samples used:       {n_processed} batches, {n_pixels:,} pixels per channel")
    logger.info("-" * 100)
    logger.info(
        f"  {'channel':<22s} {'y_mean':>9s} {'y_std':>9s}  "
        f"{'mu_mean':>9s} {'mu_std':>9s}  "
        f"{'r_mean':>9s} {'r_std':>9s}  {'ratio':>8s}"
    )
    logger.info("-" * 100)
    for c in range(len(means_r)):
        name = channel_names[c] if c < len(channel_names) else f"ch{c}"
        ratio = stds_r[c] / sigma_data if sigma_data > 0 else float('nan')
        flag = "  *** check" if (ratio < 0.5 or ratio > 2.0) else ""
        logger.info(
            f"  {name:<22s} {means_y[c]:>9.3f} {stds_y[c]:>9.3f}  "
            f"{means_mu[c]:>9.3f} {stds_mu[c]:>9.3f}  "
            f"{means_r[c]:>9.3f} {stds_r[c]:>9.3f}  {ratio:>8.3f}{flag}"
        )
    logger.info("=" * 100)
    logger.info(
        f"[RESIDUAL DIAG] Recommendation: sigma_data should be set near the "
        f"average residual std.  Current: {sigma_data}, "
        f"observed mean residual std: {stds_r.mean():.3f}"
    )
    logger.info("=" * 100)

    trainer.model.train()

def load_dataset_and_sampler(
    conf,
    param_HR,
    param_LR,
    world_size,
    rank,
    is_train,
):
    """
    Load the Z-score only dataset and sampler for training or validation.
    """
    seed = conf["seed"]
    transforms = load_transforms(conf)

    dataset = Dscale_Dataset(
        param_HR,
        param_LR,
        transform=transforms,
        seed=seed,
    )

    logging.info(f"Downscaling dataset loaded")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
        shuffle=is_train,
        drop_last=True,
    )

    logging.info(f"DistributedSampler created")

    return dataset, sampler


def load_model_states_and_optimizer(conf, model, device):
    """
    Load the model states, optimizer, scheduler, and gradient scaler.

    Returns:
        tuple: (conf, model, optimizer, scheduler, scaler, checkpoint)
            - checkpoint is the loaded torch.load() dict (non-FSDP) or None.
              For FSDP loads, the model and optimizer are loaded via
              TorchFSDPCheckpointIO, but we still load the small checkpoint.pt
              for scheduler/scaler/epoch state and return it for downstream
              use (e.g., EMA state extraction).
    """

    # convert $USER to the actual user name
    conf["save_loc"] = save_loc = os.path.expandvars(conf["save_loc"])

    # training hyperparameters
    learning_rate = float(conf["trainer"]["learning_rate"])
    weight_decay = float(conf["trainer"]["weight_decay"])
    amp = conf["trainer"]["amp"]

    # load weights / states flags
    load_weights = False if "load_weights" not in conf["trainer"] else conf["trainer"]["load_weights"]
    load_optimizer_conf = False if "load_optimizer" not in conf["trainer"] else conf["trainer"]["load_optimizer"]
    load_scaler_conf = False if "load_scaler" not in conf["trainer"] else conf["trainer"]["load_scaler"]
    load_scheduler_conf = False if "load_scheduler" not in conf["trainer"] else conf["trainer"]["load_scheduler"]

    checkpoint = None    # filled in below when reloading state

    # --- Fresh start: no checkpoint to load --- #
    if not load_weights:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        if conf["trainer"]["mode"] == "fsdp":
            optimizer = FSDPOptimizerWrapper(optimizer, model)

        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    # --- Multi-step: load weights only --- #
    elif load_weights and not (load_optimizer_conf or load_scaler_conf or load_scheduler_conf):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        if conf["trainer"]["mode"] == "fsdp":
            logging.info(f"Loading FSDP model from {save_loc}")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
            optimizer = FSDPOptimizerWrapper(optimizer, model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))

            # Also load the auxiliary checkpoint.pt if it exists, so we can
            # pick up EMA / epoch info from it.
            aux_path = os.path.join(save_loc, "checkpoint.pt")
            if os.path.exists(aux_path):
                checkpoint = torch.load(aux_path, map_location=device)

        else:
            ckpt = os.path.join(save_loc, "checkpoint.pt")
            checkpoint = torch.load(ckpt, map_location=device)
            if conf["trainer"]["mode"] == "ddp":
                logging.info(f"Loading DDP model from {save_loc}")
                load_msg = model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
                load_state_dict_error_handler(load_msg)
            else:
                logging.info(f"Loading single-GPU model from {save_loc}")
                load_msg = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                load_state_dict_error_handler(load_msg)

        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

    # --- Full reload: weights + optimizer + scheduler + scaler --- #
    else:
        ckpt = os.path.join(save_loc, "checkpoint.pt")
        checkpoint = torch.load(ckpt, map_location=device)

        if conf["trainer"]["mode"] == "fsdp":
            logging.info(f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
            optimizer = FSDPOptimizerWrapper(optimizer, model)
            checkpoint_io = TorchFSDPCheckpointIO()
            checkpoint_io.load_unsharded_model(model, os.path.join(save_loc, "model_checkpoint.pt"))
            if "load_optimizer" in conf["trainer"] and conf["trainer"]["load_optimizer"]:
                checkpoint_io.load_unsharded_optimizer(optimizer, os.path.join(save_loc, "optimizer_checkpoint.pt"))
        else:
            if conf["trainer"]["mode"] == "ddp":
                logging.info(f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                logging.info(f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
            )
            if "load_optimizer" in conf["trainer"] and conf["trainer"]["load_optimizer"]:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler = load_scheduler(optimizer, conf)
        scaler = ShardedGradScaler(enabled=amp) if conf["trainer"]["mode"] == "fsdp" else GradScaler(enabled=amp)

        # Update the config file to the current epoch
        if "reload_epoch" in conf["trainer"] and conf["trainer"]["reload_epoch"]:
            conf["trainer"]["start_epoch"] = checkpoint["epoch"] + 1

        if conf["trainer"]["start_epoch"] > 0:
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Enable updating the lr if not using a policy
    if conf["trainer"]["update_learning_rate"] if "update_learning_rate" in conf["trainer"] else False:
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    return conf, model, optimizer, scheduler, scaler, checkpoint


def _load_ema_shards_if_present(conf, trainer, rank, device, checkpoint=None):
    """Load EMA state into the trainer's existing self.ema object, if any.

    Search order:
      1. FSDP per-rank shard:  ema_checkpoint_rank{NNNN}.pt
      2. Embedded in the legacy checkpoint dict (non-FSDP case)

    No-op when EMA isn't enabled on the trainer, or no shard / state exists.
    This must be called AFTER the trainer is constructed, because the
    trainer's __init__ creates self.ema and copies the current (loaded) model
    weights into the EMA shadows.  We then overwrite those shadows with
    whatever was persisted on disk.
    """
    # Trainer might not have EMA enabled at all.
    if not hasattr(trainer, "ema") or getattr(trainer, "ema", None) is None:
        if rank == 0:
            logging.info("[EMA] trainer has no EMA object; skipping EMA load")
        return

    save_loc = os.path.expandvars(conf["save_loc"])

    # --- Path 1: per-rank FSDP shard --- #
    ema_path = os.path.join(save_loc, f"ema_checkpoint_rank{rank:04d}.pt")
    if os.path.exists(ema_path):
        ema_local = torch.load(ema_path, map_location="cpu")
        trainer.ema.shadow = {
            k: v.to(device) for k, v in ema_local["shadow"].items()
        }
        trainer.ema.shadow_buffers = {
            k: v.to(device) for k, v in ema_local.get("shadow_buffers", {}).items()
        }
        # Respect the saved decay if present; otherwise keep the YAML value.
        if "decay" in ema_local and ema_local["decay"] is not None:
            trainer.ema.decay = ema_local["decay"]
        if rank == 0:
            logging.info(
                f"[EMA] Loaded per-rank EMA shard from {ema_path} "
                f"(decay={trainer.ema.decay})"
            )
        return

    # --- Path 2: embedded in the auxiliary checkpoint dict --- #
    if checkpoint is not None and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
        ema_local = checkpoint["ema_state_dict"]
        if hasattr(trainer, "load_ema_state_dict"):
            trainer.load_ema_state_dict(ema_local)
        else:
            # Fall back to direct assignment if no method exists.
            trainer.ema.shadow = {
                k: v.to(device) for k, v in ema_local["shadow"].items()
            }
            trainer.ema.shadow_buffers = {
                k: v.to(device) for k, v in ema_local.get("shadow_buffers", {}).items()
            }
            if "decay" in ema_local and ema_local["decay"] is not None:
                trainer.ema.decay = ema_local["decay"]
        if rank == 0:
            logging.info(
                f"[EMA] Loaded EMA state from embedded checkpoint "
                f"(decay={trainer.ema.decay})"
            )
        return

    # --- No EMA state found --- #
    if rank == 0:
        logging.info(
            "[EMA] No saved EMA found.  EMA shadows initialized from "
            "current model weights; accumulation will begin from this point."
        )


def main(rank, world_size, conf, backend, trial=False):
    """
    Main function to set up training and validation processes.
    """

    conf["save_loc"] = os.path.expandvars(conf["save_loc"])

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"], backend)

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    seed = conf["seed"]
    seed_everything(seed)

    train_batch_size = conf["trainer"]["train_batch_size"]
    valid_batch_size = conf["trainer"]["valid_batch_size"]
    thread_workers = conf["trainer"]["thread_workers"]
    valid_thread_workers = conf["trainer"]["valid_thread_workers"] if "valid_thread_workers" in conf["trainer"] else thread_workers

    train_years_range = conf["data"]["train_years"]
    valid_years_range = conf["data"]["valid_years"]

    train_years = [str(year) for year in range(train_years_range[0], train_years_range[1])]
    valid_years = [str(year) for year in range(valid_years_range[0], valid_years_range[1])]

    if conf["data"]["scaler_type"] == "std-dscale":
        param_HR = {}
        param_LR = {}

        upper_files = sorted(glob(conf["data"]["save_loc"]))
        upper_files_LR = sorted(glob(conf["data"]["dscale_input"]["save_loc"]))

        if ("surface_variables" in conf["data"]) and (len(conf["data"]["surface_variables"]) > 0):
            list_surf_ds = sorted(glob(conf["data"]["save_loc_surface"]))
        else:
            list_surf_ds = None

        list_surf_ds_LR = sorted(glob(conf["data"]["dscale_input"]["save_loc_surface"]))

        if ("dynamic_forcing_variables" in conf["data"]) and (len(conf["data"]["dynamic_forcing_variables"]) > 0):
            list_dyn_forcing_ds = sorted(glob(conf["data"]["save_loc_dynamic_forcing"]))
        else:
            list_dyn_forcing_ds = None

        if ("diagnostic_variables" in conf["data"]) and (len(conf["data"]["diagnostic_variables"]) > 0):
            list_diag_ds = sorted(glob(conf["data"]["save_loc_diagnostic"]))
        else:
            list_diag_ds = None

        train_years = [str(year) for year in range(train_years_range[0], train_years_range[1])]
        valid_years = [str(year) for year in range(valid_years_range[0], valid_years_range[1])]

        train_files = [file for file in upper_files if any(year in file for year in train_years)]
        valid_files = [file for file in upper_files if any(year in file for year in valid_years)]

        train_files_LR = [file for file in upper_files_LR if any(year in file for year in train_years)]
        valid_files_LR = [file for file in upper_files_LR if any(year in file for year in valid_years)]

        if list_surf_ds is not None:
            train_list_surf_ds = [file for file in list_surf_ds if any(year in file for year in train_years)]
            valid_list_surf_ds = [file for file in list_surf_ds if any(year in file for year in valid_years)]
        else:
            train_list_surf_ds = None
            valid_list_surf_ds = None

        train_list_surf_ds_LR = [file for file in list_surf_ds_LR if any(year in file for year in train_years)]
        valid_list_surf_ds_LR = [file for file in list_surf_ds_LR if any(year in file for year in valid_years)]

        if list_dyn_forcing_ds is not None:
            train_list_dyn_forcing_ds = [file for file in list_dyn_forcing_ds if any(year in file for year in train_years)]
            valid_list_dyn_forcing_ds = [file for file in list_dyn_forcing_ds if any(year in file for year in valid_years)]
        else:
            train_list_dyn_forcing_ds = None
            valid_list_dyn_forcing_ds = None

        if list_diag_ds is not None:
            train_list_diag_ds = [file for file in list_diag_ds if any(year in file for year in train_years)]
            valid_list_diag_ds = [file for file in list_diag_ds if any(year in file for year in valid_years)]
        else:
            train_list_diag_ds = None
            valid_list_diag_ds = None

        param_HR["varname_upper_air"] = conf["data"]["variables"]
        param_HR["varname_surface"] = conf["data"]["surface_variables"]
        param_HR["varname_dyn_forcing"] = conf["data"]["dynamic_forcing_variables"]
        param_HR["varname_forcing"] = conf["data"]["forcing_variables"]
        param_HR["varname_static"] = conf["data"]["static_variables"]
        param_HR["varname_diagnostic"] = conf["data"]["diagnostic_variables"]
        param_HR["filename_forcing"] = conf["data"]["save_loc_forcing"]
        param_HR["filename_static"] = conf["data"]["save_loc_static"]
        param_HR["levels"] = conf["data"]["levels"]

        param_LR["levels"] = conf["data"]["dscale_input"]["levels"]
        param_LR["varname_upper_air"] = conf["data"]["dscale_input"]["variables"]
        param_LR["varname_surface"] = conf["data"]["dscale_input"]["surface_variables"]

        param_HR_train = copy.deepcopy(param_HR)
        param_HR_train["filenames"] = train_files
        param_HR_train["filename_surface"] = train_list_surf_ds
        param_HR_train["filename_dyn_forcing"] = train_list_dyn_forcing_ds
        param_HR_train["filename_diagnostic"] = train_list_diag_ds
        param_HR_train["history_len"] = conf["data"]["history_len"]
        param_HR_train["forecast_len"] = conf["data"]["forecast_len"]

        param_LR_train = copy.deepcopy(param_LR)
        param_LR_train["filenames"] = train_files_LR
        param_LR_train["filename_surface"] = train_list_surf_ds_LR
        param_LR_train["history_len"] = conf["data"]["dscale_input"]["history_len"]
        param_LR_train["forecast_len"] = conf["data"]["dscale_input"]["forecast_len"]

        train_dataset, train_sampler = load_dataset_and_sampler(
            conf, param_HR_train, param_LR_train, world_size, rank, is_train=True,
        )

        param_HR_valid = copy.deepcopy(param_HR)
        param_HR_valid["filenames"] = valid_files
        param_HR_valid["filename_surface"] = valid_list_surf_ds
        param_HR_valid["filename_dyn_forcing"] = valid_list_dyn_forcing_ds
        param_HR_valid["filename_diagnostic"] = valid_list_diag_ds
        param_HR_valid["history_len"] = conf["data"]["history_len"]
        param_HR_valid["forecast_len"] = conf["data"]["forecast_len"]

        param_LR_valid = copy.deepcopy(param_LR)
        param_LR_valid["filenames"] = valid_files_LR
        param_LR_valid["filename_surface"] = valid_list_surf_ds_LR
        param_LR_valid["history_len"] = conf["data"]["dscale_input"]["history_len"]
        param_LR_valid["forecast_len"] = conf["data"]["dscale_input"]["forecast_len"]

        valid_dataset, valid_sampler = load_dataset_and_sampler(
            conf, param_HR_valid, param_LR_valid, world_size, rank, is_train=False,
        )

    else:
        raise Exception("unsupported scaler")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True if thread_workers > 0 else False,
        num_workers=thread_workers,
        drop_last=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        sampler=valid_sampler,
        pin_memory=False,
        num_workers=valid_thread_workers,
        drop_last=True,
    )

    # model
    m = load_model(conf)
    m.to(device)

    if conf["trainer"].get("compile", False):
        m = torch.compile(m)

    if conf["trainer"]["mode"] in ["ddp", "fsdp"]:
        model = distributed_model_wrapper(conf, m, device)
    else:
        model = m

    # Load model weights (if any), optimizer, scheduler, gradient scaler.
    # Now also returns `checkpoint` so we can extract embedded EMA state if any.
    conf, model, optimizer, scheduler, scaler, checkpoint = load_model_states_and_optimizer(
        conf, model, device
    )

    train_criterion = VariableTotalLoss2D(conf)
    valid_criterion = VariableTotalLoss2D(conf, validation=True)

    metrics = CorrDiffMetrics(conf)

    # ---- Initialize trainer ---- #
    # IMPORTANT: trainer construction must happen AFTER model weights are loaded.
    # The trainer's __init__ creates the EMA object, copying current
    # (already-loaded) model weights into the EMA shadow tensors.  This is
    # what guarantees that a fresh EMA on a resumed run starts from the
    # trained weights rather than random init.
    trainer_cls = load_trainer(conf)

    # Newer trainers may accept conf; older ones don't.  Try the modern signature
    # first and fall back gracefully.
    try:
        trainer = trainer_cls(model, rank, conf)
    except TypeError:
        trainer = trainer_cls(model, rank)

    # ---- Load EMA state (if a saved shard exists) ---- #
    # This overwrites the freshly-initialized EMA shadows with the persisted
    # state.  For the very first resume after enabling EMA save, this is a
    # no-op (no shards exist yet); from the next save onward it activates.
    _load_ema_shards_if_present(conf, trainer, rank, device, checkpoint=checkpoint)


    if conf["trainer"].get("run_residual_diagnostic", False):
        _run_residual_diagnostic(trainer, train_loader, device, conf)

    
    # Fit the model
    result = trainer.fit(
        conf,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        train_criterion=train_criterion,
        valid_criterion=valid_criterion,
        scaler=scaler,
        scheduler=scheduler,
        metrics=metrics,
        rollout_scheduler=annealed_probability,
        trial=trial,
    )

    return result


class Objective(BaseObjective):
    def __init__(self, config, metric="val_loss", device="cpu"):
        BaseObjective.__init__(self, config, metric, device)

    def train(self, trial, conf):
        conf["model"]["dim_head"] = conf["model"]["dim"]
        conf["model"]["vq_codebook_dim"] = conf["model"]["dim"]

        try:
            return main(0, 1, conf, trial=trial)

        except Exception as E:
            if "CUDA" in str(E) or "non-singleton" in str(E):
                logging.warning(f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}.")
                raise optuna.TrialPruned()
            elif "non-singleton" in str(E):
                logging.warning(f"Pruning trial {trial.number} due to shape mismatch: {str(E)}.")
                raise optuna.TrialPruned()
            else:
                logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
                raise E


if __name__ == "__main__":
    description = "Train a Downscaling model"
    parser = ArgumentParser(description=description)

    parser.add_argument(
        "-c", "--config", dest="model_config", type=str, default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l", dest="launch", type=int, default=0,
        help="Submit workers to PBS.",
    )
    parser.add_argument(
        "--backend", type=str, default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Backend for distribted training.",
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    backend = args_dict.pop("backend")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)

    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)

    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(config, os.path.join(save_loc, "model.yml"))

    if launch:
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path, backend)
        sys.exit()

    local_rank, world_rank, world_size = get_rank_info(conf["trainer"]["mode"])
    main(world_rank, world_size, conf, backend)
    