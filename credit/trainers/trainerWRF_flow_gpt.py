"""WRF single-step trainer with large-scale flow-pattern correction."""

import gc
import logging
from collections import defaultdict

import numpy as np
import pandas as pd  # kept for drop-in compatibility with the original module
import torch
import torch.distributed as dist
import tqdm
import xarray as xr
from torch.cuda.amp import autocast
from torch.nn.functional import avg_pool2d, interpolate  # interpolate kept for drop-in compatibility
from torch.utils.data import IterableDataset

import optuna
from credit.data import concat_and_reshape, reshape_only
from credit.models.checkpoint import TorchFSDPCheckpointIO  # kept for drop-in compatibility
from credit.scheduler import update_on_batch, update_on_epoch  # update_on_epoch kept for drop-in compatibility
from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.utils import accum_log, cleanup, cycle  # cleanup kept for drop-in compatibility

logger = logging.getLogger(__name__)


# ====================================================================== #
# flow pattern constraint helper functions


def _encoded_time_to_total_hours(t_encoded):
    """Convert encoded [hour_sin, hour_cos, doy_sin, doy_cos] to total hours."""
    two_pi = t_encoded.new_tensor(2.0 * np.pi)

    h = (torch.atan2(t_encoded[..., 0], t_encoded[..., 1]) % two_pi) * 24.0 / two_pi
    d = (torch.atan2(t_encoded[..., 2], t_encoded[..., 3]) % two_pi) * 365.25 / two_pi
    return d * 24.0 + h


def interpolate_boundary(x_boundary, t1, tb0, tb1):
    """
    Linearly interpolate the two boundary time slices to the target time.

    x_boundary is expected to be shaped like (B, C, 2, H, W).  The time
    tensors are expected to be either (4,) or (B, 4), with columns/entries
    [hour_sin, hour_cos, doy_sin, doy_cos].
    """
    if x_boundary.ndim < 5 or x_boundary.shape[2] < 2:
        raise ValueError(
            "x_boundary must have shape (B, C, >=2, H, W) for boundary interpolation; "
            f"got shape {tuple(x_boundary.shape)}"
        )

    total1 = _encoded_time_to_total_hours(t1)
    totalb0 = _encoded_time_to_total_hours(tb0)
    totalb1 = _encoded_time_to_total_hours(tb1)

    # Same interpolation semantics as the original code, but batched and device-safe.
    w = (total1 - totalb0) / (totalb1 - totalb0)
    while w.ndim < x_boundary.ndim:
        w = w.unsqueeze(-1)

    x0 = x_boundary[:, :, 0:1, :, :]
    x1 = x_boundary[:, :, 1:2, :, :]
    return x0 + w * (x1 - x0)


def convert_zscore_batch(y_ref, varnames, dict_from, dict_to, dict_unit):
    """Convert ERA5 z-scores/units into the corresponding C404 z-score space."""
    if y_ref.shape[1] != len(varnames):
        raise ValueError(
            "Number of variable names must match y_ref channel dimension; "
            f"got {len(varnames)} names and y_ref shape {tuple(y_ref.shape)}"
        )

    y_out = torch.empty_like(y_ref)
    for i, varname in enumerate(varnames):
        mean_from, std_from = dict_from[varname]
        mean_to, std_to = dict_to[varname]
        f_unit = dict_unit[varname]

        mean_from = y_ref.new_tensor(mean_from)
        std_from = y_ref.new_tensor(std_from)
        mean_to = y_ref.new_tensor(mean_to)
        std_to = y_ref.new_tensor(std_to)
        f_unit = y_ref.new_tensor(f_unit)

        if varname[0] == "Q":
            y_ERA5 = (y_ref[:, i] * std_from + mean_from) * f_unit
            y_out[:, i] = (torch.sqrt(y_ERA5) - mean_to) / std_to
        else:
            y_out[:, i] = ((y_ref[:, i] * std_from + mean_from) * f_unit - mean_to) / std_to

    return y_out


def fill_boundaries(low_ref, y_ref_zscore, width=10):
    """Blend smoothed reference fields back to original reference values near edges."""
    if width <= 0:
        return low_ref

    H, W = low_ref.shape[-2], low_ref.shape[-1]
    device = low_ref.device

    # Compute the ramp in fp32 for stable arange math, then cast to the field dtype.
    ramp_dtype = torch.float32
    ramp_h = torch.clamp(
        torch.minimum(
            torch.arange(H, device=device, dtype=ramp_dtype),
            torch.arange(H - 1, -1, -1, device=device, dtype=ramp_dtype),
        )
        / float(width),
        0,
        1,
    )
    ramp_w = torch.clamp(
        torch.minimum(
            torch.arange(W, device=device, dtype=ramp_dtype),
            torch.arange(W - 1, -1, -1, device=device, dtype=ramp_dtype),
        )
        / float(width),
        0,
        1,
    )

    # alpha is 0 at the outer boundary and 1 in the interior.
    alpha = (ramp_h[:, None] * ramp_w[None, :]).to(dtype=low_ref.dtype)
    alpha = alpha.view(*([1] * (low_ref.ndim - 2)), H, W)

    return alpha * low_ref + (1 - alpha) * y_ref_zscore


def adjust_to_reference(y_pred_zscore, y_ref_zscore, sigma=7, width=3):
    """Replace the large-scale component of y_pred_zscore with y_ref_zscore."""
    if y_pred_zscore.shape != y_ref_zscore.shape:
        raise ValueError(
            "Prediction/reference shapes must match for large-scale adjustment; "
            f"got {tuple(y_pred_zscore.shape)} and {tuple(y_ref_zscore.shape)}"
        )

    H, W = y_pred_zscore.shape[-2], y_pred_zscore.shape[-1]
    k = int(sigma) | 1  # ensure odd kernel
    pad = k // 2

    low_pred = avg_pool2d(
        y_pred_zscore.reshape(-1, 1, H, W), k, stride=1, padding=pad
    ).reshape_as(y_pred_zscore)
    low_ref = avg_pool2d(
        y_ref_zscore.reshape(-1, 1, H, W), k, stride=1, padding=pad
    ).reshape_as(y_ref_zscore)

    low_ref = fill_boundaries(low_ref, y_ref_zscore, width=width)
    return (y_pred_zscore - low_pred) + low_ref


def check_time_proximity(t1, tb0, tb1, max_hours=2):
    """Return a bool tensor indicating whether target time is near either boundary."""
    total1 = _encoded_time_to_total_hours(t1)
    totalb0 = _encoded_time_to_total_hours(tb0)
    totalb1 = _encoded_time_to_total_hours(tb1)

    diff0 = torch.abs(total1 - totalb0)
    diff1 = torch.abs(total1 - totalb1)

    return torch.logical_or(diff0 <= max_hours, diff1 <= max_hours)


# ======================================================================================== #


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)

        logger.info("WRF single-step training")

        # Initialized lazily from the conf passed to train_one_epoch/validate.
        # This preserves the original Trainer(model, rank) constructor signature.
        self._flow_constraint_ready = False
        self.ind_C404 = None
        self.ind_ERA5 = None
        self.varnames_ERA5_all = None
        self.dict_ERA5_zscore = None
        self.dict_C404_zscore = None
        self.dict_ERA5_unit = None

    def _ensure_flow_constraint_config(self, conf):
        """Build the hard-coded flow-correction lookup tables once, using conf."""
        if self._flow_constraint_ready:
            return

        # ======================================================= #
        # !!!!!!!!!! hard coded blocks, will fix later !!!!!!!!!! #

        varnames_upper_ERA5 = ["U", "V", "T", "Q"]
        varnames_ERA5 = ["SP", "VAR_2T", "VAR_10U", "VAR_10V", "PWAT_05"]
        units_upper_ERA5 = [1, 1, 1, 1]
        units_ERA5 = [1, 1, 1, 1, 1 / np.sqrt(1000.0)]

        # ERA5 PWAT: kg/m2 == mm; Q: kg/kg
        # CONUS404 PWAT: m; Q: kg/kg

        varnames_upper_C404 = ["WRF_U", "WRF_V", "WRF_T", "WRF_Q_tot_05"]
        varnames_C404 = ["WRF_SP", "WRF_T2", "WRF_U10", "WRF_V10", "WRF_PWAT_05"]

        C404_level_ind = [1, 2, 3, 5, 7, 10]
        ERA5_level_ind = [0, 1, 2, 3, 4, 5]

        varname_C404_map = []
        for var in conf["data"]["variables"]:
            for _ in range(conf["data"]["levels"]):
                varname_C404_map.append(var)
        varname_C404_map += conf["data"]["surface_variables"]

        varname_ERA5_map = []
        for var in conf["data"]["boundary"]["variables"]:
            for _ in range(conf["data"]["boundary"]["levels"]):
                varname_ERA5_map.append(var)
        varname_ERA5_map += conf["data"]["boundary"]["surface_variables"]

        ind_C404_upper = {}
        for var_C404 in varnames_upper_C404:
            ind_C404_upper[var_C404] = [
                i_var for i_var, var in enumerate(varname_C404_map) if var == var_C404
            ]

        ind_ERA5_upper = {}
        for var_ERA5 in varnames_upper_ERA5:
            ind_ERA5_upper[var_ERA5] = [
                i_var for i_var, var in enumerate(varname_ERA5_map) if var == var_ERA5
            ]

        ind_C404 = []
        for var_C404 in varnames_upper_C404:
            for ind_ in C404_level_ind:
                ind_C404.append(ind_C404_upper[var_C404][ind_])

        for var_C404 in varnames_C404:
            ind_C404.append(varname_C404_map.index(var_C404))

        ind_ERA5 = []
        for var_ERA5 in varnames_upper_ERA5:
            for ind_ in ERA5_level_ind:
                ind_ERA5.append(ind_ERA5_upper[var_ERA5][ind_])

        for var_ERA5 in varnames_ERA5:
            ind_ERA5.append(varname_ERA5_map.index(var_ERA5))

        with (
            xr.open_dataset(conf["data"]["boundary"]["mean_path"]) as ds_ERA5_mean,
            xr.open_dataset(conf["data"]["boundary"]["std_path"]) as ds_ERA5_std,
            xr.open_dataset(conf["data"]["mean_path"]) as ds_C404_mean,
            xr.open_dataset(conf["data"]["std_path"]) as ds_C404_std,
        ):
            dict_ERA5_zscore = {}
            dict_ERA5_unit = {}
            varnames_ERA5_all = []

            for i_var, var_ERA5 in enumerate(varnames_upper_ERA5):
                for i_level, ind_level in enumerate(ERA5_level_ind):
                    key = f"{var_ERA5}_{i_level}"
                    varnames_ERA5_all.append(key)
                    dict_ERA5_zscore[key] = [
                        float(ds_ERA5_mean[var_ERA5].values[ind_level]),
                        float(ds_ERA5_std[var_ERA5].values[ind_level]),
                    ]
                    dict_ERA5_unit[key] = units_upper_ERA5[i_var]

            varnames_ERA5_all = varnames_ERA5_all + varnames_ERA5
            for i_var, var_ERA5 in enumerate(varnames_ERA5):
                dict_ERA5_zscore[var_ERA5] = [
                    float(ds_ERA5_mean[var_ERA5].values),
                    float(ds_ERA5_std[var_ERA5].values),
                ]
                dict_ERA5_unit[var_ERA5] = units_ERA5[i_var]

            dict_C404_zscore_raw = {}
            for var_C404 in varnames_upper_C404:
                for i_level, ind_level in enumerate(C404_level_ind):
                    key = f"{var_C404}_{i_level}"
                    dict_C404_zscore_raw[key] = [
                        float(ds_C404_mean[var_C404].values[ind_level]),
                        float(ds_C404_std[var_C404].values[ind_level]),
                    ]

            for var_C404 in varnames_C404:
                dict_C404_zscore_raw[var_C404] = [
                    float(ds_C404_mean[var_C404].values),
                    float(ds_C404_std[var_C404].values),
                ]

        # Map C404 stats onto ERA5 keys, preserving the original variable/level order.
        keys_C404 = list(dict_C404_zscore_raw.keys())
        keys_ERA5 = list(dict_ERA5_zscore.keys())
        if len(keys_C404) != len(keys_ERA5):
            raise ValueError(
                "C404/ERA5 correction variable counts do not match: "
                f"{len(keys_C404)} vs {len(keys_ERA5)}"
            )
        dict_C404_zscore = {
            key_era5: dict_C404_zscore_raw[key_c404]
            for key_c404, key_era5 in zip(keys_C404, keys_ERA5)
        }

        self.ind_C404 = ind_C404
        self.ind_ERA5 = ind_ERA5
        self.varnames_ERA5_all = varnames_ERA5_all
        self.dict_ERA5_zscore = dict_ERA5_zscore
        self.dict_C404_zscore = dict_C404_zscore
        self.dict_ERA5_unit = dict_ERA5_unit
        self._flow_constraint_ready = True

    @staticmethod
    def _reshape_time_encode(x_time_encode, batch_size):
        """
        Return x_time_encode as (B, 4, 4), preserving the original column meaning.

        The original code used x_time_encode.reshape(4, 4) and then selected
        columns 1, 2, and 3.  This helper keeps that meaning while supporting
        an explicit batch dimension.
        """
        if x_time_encode.ndim == 3 and x_time_encode.shape[-2:] == (4, 4):
            x_time_decode = x_time_encode.reshape(-1, 4, 4)
        elif x_time_encode.ndim == 2 and x_time_encode.shape == (4, 4):
            x_time_decode = x_time_encode.unsqueeze(0)
        elif x_time_encode.ndim == 2 and x_time_encode.shape[-1] == 16:
            x_time_decode = x_time_encode.reshape(-1, 4, 4)
        elif x_time_encode.ndim == 1 and x_time_encode.numel() == 16:
            x_time_decode = x_time_encode.reshape(1, 4, 4)
        elif x_time_encode.numel() == batch_size * 16:
            x_time_decode = x_time_encode.reshape(batch_size, 4, 4)
        else:
            raise ValueError(
                "x_time_encode cannot be interpreted as batched 4x4 time encoding; "
                f"got shape {tuple(x_time_encode.shape)} for batch size {batch_size}"
            )

        if x_time_decode.shape[0] == 1 and batch_size > 1:
            x_time_decode = x_time_decode.expand(batch_size, -1, -1)
        elif x_time_decode.shape[0] != batch_size:
            raise ValueError(
                "x_time_encode batch dimension does not match x_boundary batch dimension; "
                f"got {x_time_decode.shape[0]} and {batch_size}"
            )

        return x_time_decode

    @staticmethod
    def _scalar_to_device_tensor(value, device):
        """Create a detached scalar tensor on the requested device."""
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() != 1:
                value = value.mean()
            return value.reshape(1).to(device=device, dtype=torch.float32, non_blocking=True)
        return torch.tensor([float(value)], device=device, dtype=torch.float32)

    def _prepare_batch(self, batch):
        """Move and reshape the batch exactly as the original train/valid code did."""
        if "x_surf" in batch:
            # input: (B, time, var, level, lat, lon), (B, time, var, lat, lon)
            # output: (B, var, time, lat, lon), x first and then x_surf
            x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
        else:
            x = reshape_only(batch["x"]).to(self.device)

        if "x_forcing_static" in batch:
            # (B, time, var, lat, lon) --> (B, var, time, lat, lon)
            x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)
            x = torch.cat((x, x_forcing_batch), dim=1)

        if "y_surf" in batch:
            y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
        else:
            y = reshape_only(batch["y"]).to(self.device)

        if "y_diag" in batch:
            # (B, time, var, lat, lon) --> (B, var, time, lat, lon)
            y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
            y = torch.cat((y, y_diag_batch), dim=1)

        if "x_surf_boundary" in batch:
            x_boundary = concat_and_reshape(batch["x_boundary"], batch["x_surf_boundary"]).to(self.device)
        else:
            x_boundary = reshape_only(batch["x_boundary"]).to(self.device)

        x_time_encode = batch["x_time_encode"].to(self.device)

        return x, y, x_boundary, x_time_encode

    def _apply_flow_constraint(self, y_pred, x_boundary, x_time_encode):
        """Apply the large-scale correction to selected y_pred channels."""
        batch_size = x_boundary.shape[0]
        x_time_decode = self._reshape_time_encode(x_time_encode, batch_size)

        # Preserve the original column selection after reshape(4, 4):
        # column 1 is target, column 2/3 are boundary times.
        t1 = x_time_decode[:, :, 1]
        tb0 = x_time_decode[:, :, 2]
        tb1 = x_time_decode[:, :, 3]

        flag_correct = check_time_proximity(t1, tb0, tb1, max_hours=2)
        if flag_correct.ndim == 0:
            flag_correct = flag_correct.reshape(1)
        flag_correct = flag_correct.to(device=y_pred.device, dtype=torch.bool)

        if not bool(flag_correct.any().item()):
            return y_pred

        x_surf = x_boundary[:, self.ind_ERA5, ...]
        y_ref = interpolate_boundary(x_surf, t1, tb0, tb1)

        y_ref_zscore = convert_zscore_batch(
            y_ref,
            self.varnames_ERA5_all,
            self.dict_ERA5_zscore,
            self.dict_C404_zscore,
            self.dict_ERA5_unit,
        ).to(device=y_pred.device, dtype=y_pred.dtype)

        y_pred_zscore = y_pred[:, self.ind_C404, ...]
        y_pred_correct = adjust_to_reference(y_pred_zscore, y_ref_zscore).to(
            device=y_pred.device, dtype=y_pred.dtype
        )

        if flag_correct.numel() != y_pred_correct.shape[0]:
            raise ValueError(
                "Correction mask batch size does not match prediction batch size; "
                f"got {flag_correct.numel()} and {y_pred_correct.shape[0]}"
            )

        if not bool(flag_correct.all().item()):
            mask = flag_correct.view(-1, *([1] * (y_pred_correct.ndim - 1)))
            y_pred_correct = torch.where(mask, y_pred_correct, y_pred_zscore)

        # Avoid mutating the model output tensor in-place after it has been used
        # to build y_pred_correct. This keeps autograd versioning safe.
        y_pred_out = y_pred.clone()
        y_pred_out[:, self.ind_C404, ...] = y_pred_correct
        return y_pred_out

    # Training function.
    def train_one_epoch(self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        self.model.train()
        self._ensure_flow_constraint_config(conf)

        # training hyperparameters
        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_accum_every = conf["trainer"]["grad_accum_every"]
        grad_max_norm = conf["trainer"]["grad_max_norm"]
        forecast_len = conf["data"]["forecast_len"]
        amp = conf["trainer"]["amp"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        # forecast step
        if "total_time_steps" in conf["data"]:
            total_time_steps = conf["data"]["total_time_steps"]
        else:
            total_time_steps = forecast_len

        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        # update the learning rate if epoch-by-epoch updates that dont depend on a metric
        if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda":
            scheduler.step()

        # ====================================================== #

        # set up a custom tqdm
        if not isinstance(trainloader.dataset, IterableDataset):
            # if batches_per_epoch = 0, use all training samples (i.e., full epoch)
            batches_per_epoch = batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader)

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch),
            total=batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        results_dict = defaultdict(list)

        # dataloader
        dl = cycle(trainloader)
        optimizer.zero_grad()

        for i in batch_group_generator:
            # Get the next batch from the iterator
            batch = next(dl)

            # training log
            logs = {}

            with autocast(enabled=amp):
                x, y, x_boundary, x_time_encode = self._prepare_batch(batch)

                # single step predict
                y_pred = self.model(x, x_boundary, x_time_encode)
                y_pred = self._apply_flow_constraint(y_pred, x_boundary, x_time_encode)

                y = y.to(device=y_pred.device, dtype=y_pred.dtype)

                # loss compute
                loss = criterion(y, y_pred).mean()

                # Metrics
                metrics_dict = metrics(y_pred.float(), y.float())

                # save training metrics
                for name, value in metrics_dict.items():
                    value = self._scalar_to_device_tensor(value, self.device)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

                # backpropagation
                scaler.scale(loss / grad_accum_every).backward()

            accum_log(logs, {"loss": loss.item()})

            if distributed:
                torch.distributed.barrier()

            should_step = ((i + 1) % grad_accum_every == 0) or ((i + 1) == batches_per_epoch)
            if should_step:
                if grad_max_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_max_norm)

                scaler.step(optimizer)
                scaler.update()

                # clear grad
                optimizer.zero_grad()

                if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch:
                    scheduler.step()

            # Handle batch_loss
            batch_loss = self._scalar_to_device_tensor(logs["loss"], self.device)

            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)

            results_dict["train_loss"].append(batch_loss[0].item())

            if "forecast_hour" in batch:
                forecast_hour_tensor = batch["forecast_hour"].to(self.device)
                if distributed:
                    dist.all_reduce(forecast_hour_tensor, dist.ReduceOp.AVG, async_op=False)
                    forecast_hour_avg = forecast_hour_tensor[-1].item()
                else:
                    forecast_hour_avg = batch["forecast_hour"][-1].item()

                results_dict["train_forecast_len"].append(forecast_hour_avg + 1)
            else:
                results_dict["train_forecast_len"].append(forecast_len + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                try:
                    print("Invalid loss value: {}".format(np.mean(results_dict["train_loss"])))
                    raise optuna.TrialPruned()

                except Exception as E:
                    raise E

            # agg the results
            to_print = "Epoch: {} train_loss: {:.6f} train_acc: {:.6f} train_mae: {:.6f} forecast_len {:.6}".format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict["train_acc"]),
                np.mean(results_dict["train_mae"]),
                np.mean(results_dict["train_forecast_len"]),
            )

            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.set_description(to_print)

            if i >= batches_per_epoch and i > 0:
                break

        #  Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        self.model.eval()
        self._ensure_flow_constraint_config(conf)

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]

        forecast_len = conf["data"]["valid_forecast_len"]
        distributed = True if conf["trainer"]["mode"] in ["fsdp", "ddp"] else False

        total_time_steps = conf["data"]["total_time_steps"] if "total_time_steps" in conf["data"] else forecast_len

        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        results_dict = defaultdict(list)

        # ====================================================== #

        # set up a custom tqdm
        if isinstance(valid_loader.dataset, IterableDataset):
            valid_batches_per_epoch = valid_batches_per_epoch
        else:
            valid_batches_per_epoch = valid_batches_per_epoch if 0 < valid_batches_per_epoch < len(valid_loader) else len(valid_loader)

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch),
            total=valid_batches_per_epoch,
            leave=True,
            disable=True if self.rank > 0 else False,
        )

        dl = cycle(valid_loader)

        for i in batch_group_generator:
            batch = next(dl)

            with torch.no_grad():
                x, y, x_boundary, x_time_encode = self._prepare_batch(batch)

                y_pred = self.model(x, x_boundary, x_time_encode)
                y_pred = self._apply_flow_constraint(y_pred, x_boundary, x_time_encode)

                y = y.to(device=y_pred.device, dtype=y_pred.dtype)
                loss = criterion(y, y_pred).mean()

                # Metrics
                metrics_dict = metrics(y_pred.float(), y.float())

                for name, value in metrics_dict.items():
                    value = self._scalar_to_device_tensor(value, self.device)

                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)

                    results_dict[f"valid_{name}"].append(value[0].item())

                batch_loss = self._scalar_to_device_tensor(loss.item(), self.device)

                if distributed:
                    dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
                    torch.distributed.barrier()

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(forecast_len + 1)

                # print to tqdm
                to_print = "Epoch: {} valid_loss: {:.6f} valid_acc: {:.6f} valid_mae: {:.6f}".format(
                    epoch,
                    np.mean(results_dict["valid_loss"]),
                    np.mean(results_dict["valid_acc"]),
                    np.mean(results_dict["valid_mae"]),
                )

                if self.rank == 0:
                    batch_group_generator.set_description(to_print)

                if i >= valid_batches_per_epoch and i > 0:
                    break

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict
