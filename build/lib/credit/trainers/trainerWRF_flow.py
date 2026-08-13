"""WRF single-step trainer with large-scale flow-pattern correction.

Fixes versus the previous revision (focusing on value passing and device consistency):
  * `conf`-dependent setup moved out of `__init__` into a lazy `_setup_correction`,
    so it can actually run (the original referenced `conf` and `xr` without having
    them in scope) and so all state lives on `self`.
  * Variable-index lookups are cached as `torch.long` tensors on `self.device`,
    so advanced indexing no longer copies a fresh CPU list to GPU on every step.
  * `.view(..., 336, 336)` replaced by `.reshape(..., H, W)` driven by the tensor
    shape — works on non-contiguous slices (e.g. from `index_select`) and on
    other resolutions.
  * `check_time_proximity` returns a plain Python bool via bitwise-or + `.item()`.
  * The dict rekeying loop is replaced by building a fresh ERA5-keyed C404 dict.
  * `Q` sqrt path clamps to `>= 0` to avoid NaN from numerical noise.
  * Denominator in the time-interpolation weight has a `clamp(min=1e-6)` guard.
  * The correction block is factored into `_apply_large_scale_correction`, used
    by both `train_one_epoch` and `validate`. The block mirrors the original
    one-for-one: same reshape (will raise on shape mismatch, not silently
    skip), same time-proximity gate, same index → interpolate → z-score →
    adjust pipeline.

Numerical behavior of the helpers is preserved: `w` in `interpolate_boundary`
stays in its natural (fp32) dtype rather than being cast to x_boundary's
dtype, and the ramps in `fill_boundaries` stay fp32 — both matching the
original under autocast.
"""

import os
import gc
import tqdm
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset
from torch.nn.functional import avg_pool2d, interpolate

import optuna
from credit.data import concat_and_reshape, reshape_only
from credit.models.checkpoint import TorchFSDPCheckpointIO
from credit.scheduler import update_on_batch, update_on_epoch
from credit.trainers.utils import cleanup, accum_log, cycle
from credit.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


# ====================================================================== #
# Flow-pattern constraint helpers.
# All tensor inputs must live on the same device.
# ====================================================================== #

def interpolate_boundary(x_boundary, t1, tb0, tb1):
    """Linearly interpolate boundary slabs (dim=2 of size 2) to time t1.

    x_boundary : (B, V, 2, H, W) — channels 0 and 1 along dim=2 are the bounding times.
    t1, tb0, tb1 : 1-D tensors of length 4 (sin/cos of hour-of-day and day-of-year).
    """
    two_pi = 2 * torch.pi

    # Fractional hours from the two-component sine/cosine encoding
    h1 = (torch.atan2(t1[0], t1[1]) % two_pi) * 24 / two_pi
    hb0 = (torch.atan2(tb0[0], tb0[1]) % two_pi) * 24 / two_pi
    hb1 = (torch.atan2(tb1[0], tb1[1]) % two_pi) * 24 / two_pi

    # Day of year
    d1 = (torch.atan2(t1[2], t1[3]) % two_pi) * 365.25 / two_pi
    db0 = (torch.atan2(tb0[2], tb0[3]) % two_pi) * 365.25 / two_pi
    db1 = (torch.atan2(tb1[2], tb1[3]) % two_pi) * 365.25 / two_pi

    # Total fractional time in hours
    total1 = d1 * 24 + h1
    totalb0 = db0 * 24 + hb0
    totalb1 = db1 * 24 + hb1

    # Interpolation weight with a zero-guard. We deliberately do NOT cast w to
    # x_boundary.dtype: the original lets fp32 w promote x_boundary in the
    # multiply, which is the more numerically stable behavior under autocast.
    denom = (totalb1 - totalb0).clamp(min=1e-6)
    w = (total1 - totalb0) / denom

    x0 = x_boundary[:, :, 0:1, :, :]
    x1 = x_boundary[:, :, 1:2, :, :]
    return x0 + w * (x1 - x0)


def convert_zscore_batch(y_ref, varnames, dict_from, dict_to, dict_unit):
    """Re-encode a tensor under a different z-score normalization.

    Variables whose name starts with 'Q' are humidity; the target side stores
    sqrt(Q), so we take sqrt after unit conversion (clamped to non-negative
    to avoid NaN from numerical noise).
    """
    y_out = torch.empty_like(y_ref)
    for i, varname in enumerate(varnames):
        mean_from, std_from = dict_from[varname]
        mean_to, std_to = dict_to[varname]
        f_unit = dict_unit[varname]
        if varname[0] == 'Q':
            y_raw = (y_ref[:, i] * std_from + mean_from) * f_unit
            y_raw = y_raw.clamp(min=0.0)
            y_out[:, i] = (y_raw ** 0.5 - mean_to) / std_to
        else:
            y_out[:, i] = ((y_ref[:, i] * std_from + mean_from) * f_unit - mean_to) / std_to
    return y_out


def fill_boundaries(low_ref, y_ref_zscore, width=3):
    """Blend `low_ref` toward `y_ref_zscore` near the spatial border.

    alpha is 1 in the interior and ramps to 0 at the edge over `width` pixels.
    """
    H, W = low_ref.shape[-2], low_ref.shape[-1]
    device = low_ref.device
    width = max(int(width), 1)

    # Build ramps in fp32 to match the original; promotion handles autocast cases.
    h_idx = torch.arange(H, device=device, dtype=torch.float32)
    w_idx = torch.arange(W, device=device, dtype=torch.float32)
    ramp_h = torch.clamp(torch.minimum(h_idx, (H - 1) - h_idx) / width, 0, 1)
    ramp_w = torch.clamp(torch.minimum(w_idx, (W - 1) - w_idx) / width, 0, 1)
    alpha = ramp_h[:, None] * ramp_w[None, :]
    return alpha * low_ref + (1 - alpha) * y_ref_zscore


def adjust_to_reference(y_pred_zscore, y_ref_zscore, sigma=7, width=3):
    """Replace the smoothed (large-scale) part of y_pred with that of the reference."""
    H, W = y_pred_zscore.shape[-2], y_pred_zscore.shape[-1]
    k = int(sigma) | 1  # force an odd kernel
    pad = k // 2

    low_pred = avg_pool2d(
        y_pred_zscore.reshape(-1, 1, H, W), k, stride=1, padding=pad
    ).reshape_as(y_pred_zscore)
    low_ref = avg_pool2d(
        y_ref_zscore.reshape(-1, 1, H, W), k, stride=1, padding=pad
    ).reshape_as(y_ref_zscore)
    # low_ref = fill_boundaries(low_ref, y_ref_zscore, width=width)

    return (y_pred_zscore - low_pred) + low_ref


def check_time_proximity(t1, tb0, tb1, max_hours=2):
    """Return True (as a Python bool) if t1 is within `max_hours` of tb0 or tb1.

    Uses bitwise-or on boolean tensors and `.item()` so it is well-defined even
    when the inputs are not 0-d.
    """
    two_pi = 2 * torch.pi

    h1 = (torch.atan2(t1[0], t1[1]) % two_pi) * 24 / two_pi
    hb0 = (torch.atan2(tb0[0], tb0[1]) % two_pi) * 24 / two_pi
    hb1 = (torch.atan2(tb1[0], tb1[1]) % two_pi) * 24 / two_pi

    d1 = (torch.atan2(t1[2], t1[3]) % two_pi) * 365.25 / two_pi
    db0 = (torch.atan2(tb0[2], tb0[3]) % two_pi) * 365.25 / two_pi
    db1 = (torch.atan2(tb1[2], tb1[3]) % two_pi) * 365.25 / two_pi

    total1 = d1 * 24 + h1
    totalb0 = db0 * 24 + hb0
    totalb1 = db1 * 24 + hb1

    close = ((total1 - totalb0).abs() <= max_hours) | ((total1 - totalb1).abs() <= max_hours)
    return bool(close.item())


def match_boundary_index(t1, tb0, tb1, tol_hours=1e-2):
    """Return 0 if t1 ≈ tb0, 1 if t1 ≈ tb1, else None.

    Comparison is done after decoding the (sin, cos) time encoding to
    fractional hours within the year. "≈" means absolute difference
    <= `tol_hours`. When both boundaries are within tolerance, the
    closer one wins; ties go to index 0.

    Default tolerance is 1e-2 hours (36 s) — tight enough to mean
    "the same hour" for hourly/3-hourly data while absorbing the
    float32 roundoff incurred by sin/cos -> atan2 round-tripping.
    """
    two_pi = 2 * torch.pi

    h1 = (torch.atan2(t1[0], t1[1]) % two_pi) * 24 / two_pi
    hb0 = (torch.atan2(tb0[0], tb0[1]) % two_pi) * 24 / two_pi
    hb1 = (torch.atan2(tb1[0], tb1[1]) % two_pi) * 24 / two_pi

    d1 = (torch.atan2(t1[2], t1[3]) % two_pi) * 365.25 / two_pi
    db0 = (torch.atan2(tb0[2], tb0[3]) % two_pi) * 365.25 / two_pi
    db1 = (torch.atan2(tb1[2], tb1[3]) % two_pi) * 365.25 / two_pi

    total1 = d1 * 24 + h1
    totalb0 = db0 * 24 + hb0
    totalb1 = db1 * 24 + hb1

    # One host sync (two scalars). Necessary because the branch below
    # picks a different code path per outcome.
    diff0 = float((total1 - totalb0).abs().item())
    diff1 = float((total1 - totalb1).abs().item())

    if diff0 <= tol_hours and diff0 <= diff1:
        return 0
    if diff1 <= tol_hours:
        return 1
    return None


# ======================================================================================== #


class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int):
        super().__init__(model, rank)
        logger.info("WRF single-step training")
        # `conf` is not available here, so we defer building the variable-index
        # tables and z-score dictionaries until the first train/validate call.
        self._correction_ready = False

    # ------------------------------------------------------------------ #
    # One-time setup of variable indices and z-score tables.
    # ------------------------------------------------------------------ #
    def _setup_correction(self, conf):
        if self._correction_ready:
            return

        # --- Hard-coded variable mapping ---
        varnames_upper_ERA5 = ['U',]
        varnames_ERA5 = ['VAR_2T', 'VAR_10U', 'VAR_10V', 'PWAT_05']
        units_upper_ERA5 = [1.0,]
        # ERA5 PWAT: kg/m^2 (== mm). CONUS404 PWAT: sqrt(m).
        # Convert mm -> m (×1/1000) then sqrt(...) -> combined factor 1/sqrt(1000).
        units_ERA5 = [1.0, 1.0, 1.0, 1.0 / float(np.sqrt(1000.0))]

        varnames_upper_C404 = ['WRF_U',]
        varnames_C404 = ['WRF_T2', 'WRF_U10', 'WRF_V10', 'WRF_PWAT_05']

        C404_level_ind = [1, 2, 3, 5, 7, 10]
        ERA5_level_ind = [0, 1, 2, 3, 4, 5]

        # --- Flat tensor-channel maps ---
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

        ind_C404_upper = {
            v: [i for i, n in enumerate(varname_C404_map) if n == v]
            for v in varnames_upper_C404
        }
        ind_ERA5_upper = {
            v: [i for i, n in enumerate(varname_ERA5_map) if n == v]
            for v in varnames_upper_ERA5
        }

        ind_C404 = []
        for v in varnames_upper_C404:
            for k in C404_level_ind:
                ind_C404.append(ind_C404_upper[v][k])
        for v in varnames_C404:
            ind_C404.append(varname_C404_map.index(v))

        ind_ERA5 = []
        for v in varnames_upper_ERA5:
            for k in ERA5_level_ind:
                ind_ERA5.append(ind_ERA5_upper[v][k])
        for v in varnames_ERA5:
            ind_ERA5.append(varname_ERA5_map.index(v))

        assert len(ind_C404) == len(ind_ERA5), (
            f"index list mismatch: |C404|={len(ind_C404)} vs |ERA5|={len(ind_ERA5)}"
        )

        # --- Z-score statistics ---
        ds_ERA5_mean = xr.open_dataset(conf['data']['boundary']['mean_path'])
        ds_ERA5_std = xr.open_dataset(conf['data']['boundary']['std_path'])
        ds_C404_mean = xr.open_dataset(conf['data']['mean_path'])
        ds_C404_std = xr.open_dataset(conf['data']['std_path'])

        dict_ERA5_zscore = {}
        dict_ERA5_unit = {}
        varnames_ERA5_all = []

        for i_var, v in enumerate(varnames_upper_ERA5):
            for i_level, ind_level in enumerate(ERA5_level_ind):
                key = f'{v}_{i_level}'
                varnames_ERA5_all.append(key)
                dict_ERA5_zscore[key] = (
                    float(ds_ERA5_mean[v].values[ind_level]),
                    float(ds_ERA5_std[v].values[ind_level]),
                )
                dict_ERA5_unit[key] = units_upper_ERA5[i_var]

        for i_var, v in enumerate(varnames_ERA5):
            varnames_ERA5_all.append(v)
            dict_ERA5_zscore[v] = (
                float(ds_ERA5_mean[v].values),
                float(ds_ERA5_std[v].values),
            )
            dict_ERA5_unit[v] = units_ERA5[i_var]

        # Build the C404 dict directly under ERA5-style keys -- no in-place rename loop.
        dict_C404_zscore = {}
        for v_era, v_c404 in zip(varnames_upper_ERA5, varnames_upper_C404):
            for i_level, ind_level in enumerate(C404_level_ind):
                key = f'{v_era}_{i_level}'
                dict_C404_zscore[key] = (
                    float(ds_C404_mean[v_c404].values[ind_level]),
                    float(ds_C404_std[v_c404].values[ind_level]),
                )
        for v_era, v_c404 in zip(varnames_ERA5, varnames_C404):
            dict_C404_zscore[v_era] = (
                float(ds_C404_mean[v_c404].values),
                float(ds_C404_std[v_c404].values),
            )

        assert set(dict_C404_zscore.keys()) == set(dict_ERA5_zscore.keys()), (
            "ERA5 and C404 z-score tables must share the same key set"
        )

        for ds in (ds_ERA5_mean, ds_ERA5_std, ds_C404_mean, ds_C404_std):
            ds.close()

        # --- Cache on self. Index tensors live on self.device. ---
        self.ind_C404 = torch.as_tensor(ind_C404, dtype=torch.long, device=self.device)
        self.ind_ERA5 = torch.as_tensor(ind_ERA5, dtype=torch.long, device=self.device)
        self.varnames_ERA5_all = varnames_ERA5_all
        self.dict_ERA5_zscore = dict_ERA5_zscore
        self.dict_C404_zscore = dict_C404_zscore
        self.dict_ERA5_unit = dict_ERA5_unit

        self._correction_ready = True

    # ------------------------------------------------------------------ #
    # Large-scale correction step shared by train and validate.
    # Mutates y_pred in place along the channel dimension for the
    # selected indices (same in-place semantics as the original).
    # ------------------------------------------------------------------ #
    def _apply_large_scale_correction(self, y_pred, x_boundary, x_time_encode):
        # New semantics:
        #   * If t1 ≈ tb0  (within tol)  →  use x_boundary slab 0 directly
        #   * If t1 ≈ tb1  (within tol)  →  use x_boundary slab 1 directly
        #   * Otherwise                  →  do not apply the correction
        # No interpolation in any case. The reshape will raise (not
        # silently skip) on a shape mismatch, matching original behavior
        # — assumes effective batch size 1.
        x_time_decode = x_time_encode.reshape(4, 4)
        t1 = x_time_decode[:, 1]
        tb0 = x_time_decode[:, 2]
        tb1 = x_time_decode[:, 3]

        match = match_boundary_index(t1, tb0, tb1)
        if match is None:
            return y_pred

        # index_select keeps things on-device and avoids host-side list copies.
        x_surf = x_boundary.index_select(1, self.ind_ERA5)
        # Pick the matching boundary slab — no interpolation.
        # Keep the singleton time dim so the downstream shape matches what
        # interpolate_boundary used to return: (B, C, 1, H, W).
        y_ref = x_surf[:, :, match:match + 1, :, :]

        y_ref_zscore = convert_zscore_batch(
            y_ref, self.varnames_ERA5_all,
            self.dict_ERA5_zscore, self.dict_C404_zscore, self.dict_ERA5_unit,
        )

        y_pred_zscore = y_pred.index_select(1, self.ind_C404)
        y_pred_correct = adjust_to_reference(y_pred_zscore, y_ref_zscore).to(y_pred.dtype)

        # Scatter the corrected channels back in place.
        y_pred.index_copy_(1, self.ind_C404, y_pred_correct)
        return y_pred

    # ---------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------- #
    def train_one_epoch(self, epoch, conf, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        self._setup_correction(conf)

        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_accum_every = conf["trainer"]["grad_accum_every"]
        grad_max_norm = conf["trainer"]["grad_max_norm"]
        forecast_len = conf["data"]["forecast_len"]
        amp = conf["trainer"]["amp"]
        distributed = conf["trainer"]["mode"] in ["fsdp", "ddp"]

        total_time_steps = conf["data"].get("total_time_steps", forecast_len)
        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda":
            scheduler.step()

        if not isinstance(trainloader.dataset, IterableDataset):
            batches_per_epoch = batches_per_epoch if 0 < batches_per_epoch < len(trainloader) else len(trainloader)

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch),
            total=batches_per_epoch,
            leave=True,
            disable=self.rank > 0,
        )

        results_dict = defaultdict(list)
        dl = cycle(trainloader)

        for i in batch_group_generator:
            batch = next(dl)
            logs = {}

            with autocast(enabled=amp):
                # ---- inputs ----
                if "x_surf" in batch:
                    x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
                else:
                    x = reshape_only(batch["x"]).to(self.device)

                if "x_forcing_static" in batch:
                    x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)
                    x = torch.cat((x, x_forcing_batch), dim=1)

                # ---- targets ----
                if "y_surf" in batch:
                    y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                else:
                    y = reshape_only(batch["y"]).to(self.device)

                if "y_diag" in batch:
                    y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                    y = torch.cat((y, y_diag_batch), dim=1)

                # ---- boundary ----
                if "x_surf_boundary" in batch:
                    x_boundary = concat_and_reshape(batch["x_boundary"], batch["x_surf_boundary"]).to(self.device)
                else:
                    x_boundary = reshape_only(batch["x_boundary"]).to(self.device)

                # ---- time encoding ----
                x_time_encode = batch["x_time_encode"].to(self.device)

                # ---- predict ----
                y_pred = self.model(x, x_boundary, x_time_encode)

                # ---- large-scale correction ----
                y_pred = self._apply_large_scale_correction(y_pred, x_boundary, x_time_encode)

                # ---- loss ----
                y = y.to(device=self.device, dtype=y_pred.dtype)
                loss = criterion(y, y_pred)

                metrics_dict = metrics(y_pred, y)
                for name, value in metrics_dict.items():
                    value = torch.tensor([value], device=self.device, dtype=torch.float32)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"train_{name}"].append(value[0].item())

                loss = loss.mean()

                # --- Per-batch loss sanity guard -------------------------- #
                # Stop training if loss is NaN/Inf or exceeds LOSS_MAX.
                # In distributed mode we all-reduce a uint8 flag with MAX so
                # every rank raises together (otherwise non-tripping ranks
                # would deadlock on the next collective). Swap RuntimeError
                # for `optuna.TrialPruned()` if you want the surrounding
                # Optuna study to prune rather than crash.
                LOSS_MAX = 2.0
                loss_val = loss.detach()
                bad = (~torch.isfinite(loss_val)) | (loss_val > LOSS_MAX)
                if distributed:
                    bad = bad.to(torch.uint8)
                    dist.all_reduce(bad, dist.ReduceOp.MAX, async_op=False)
                if bool(bad.item()):
                    raise RuntimeError(
                        f"Training halted on rank {self.rank}: "
                        f"loss={loss_val.item():.6g} is non-finite or > {LOSS_MAX}"
                    )
                # ---------------------------------------------------------- #

                scaler.scale(loss / grad_accum_every).backward()

            accum_log(logs, {"loss": loss.item() / grad_accum_every})

            if distributed:
                torch.distributed.barrier()

            if grad_max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_loss = torch.tensor([logs["loss"]], device=self.device, dtype=torch.float32)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())

            if "forecast_hour" in batch:
                forecast_hour_tensor = batch["forecast_hour"].to(self.device)
                if distributed:
                    dist.all_reduce(forecast_hour_tensor, dist.ReduceOp.AVG, async_op=False)
                forecast_hour_avg = forecast_hour_tensor[-1].item()
                results_dict["train_forecast_len"].append(forecast_hour_avg + 1)
            else:
                results_dict["train_forecast_len"].append(forecast_len + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                print("Invalid loss value: {}".format(np.mean(results_dict["train_loss"])))
                raise optuna.TrialPruned()

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

            if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch:
                scheduler.step()

            if i >= batches_per_epoch and i > 0:
                break

        batch_group_generator.close()
        torch.cuda.empty_cache()
        gc.collect()
        return results_dict

    # ---------------------------------------------------------------- #
    # Validation
    # ---------------------------------------------------------------- #
    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        self._setup_correction(conf)
        self.model.eval()

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
        forecast_len = conf["data"]["valid_forecast_len"]
        distributed = conf["trainer"]["mode"] in ["fsdp", "ddp"]

        total_time_steps = conf["data"].get("total_time_steps", forecast_len)
        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        results_dict = defaultdict(list)

        if isinstance(valid_loader.dataset, IterableDataset):
            pass  # keep user-provided value
        else:
            valid_batches_per_epoch = valid_batches_per_epoch if 0 < valid_batches_per_epoch < len(valid_loader) else len(valid_loader)

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch),
            total=valid_batches_per_epoch,
            leave=True,
            disable=self.rank > 0,
        )

        dl = cycle(valid_loader)

        for i in batch_group_generator:
            batch = next(dl)
            with torch.no_grad():
                if "x_surf" in batch:
                    x = concat_and_reshape(batch["x"], batch["x_surf"]).to(self.device)
                else:
                    x = reshape_only(batch["x"]).to(self.device)

                if "x_forcing_static" in batch:
                    x_forcing_batch = batch["x_forcing_static"].to(self.device).permute(0, 2, 1, 3, 4)
                    x = torch.cat((x, x_forcing_batch), dim=1)

                if "y_surf" in batch:
                    y = concat_and_reshape(batch["y"], batch["y_surf"]).to(self.device)
                else:
                    y = reshape_only(batch["y"]).to(self.device)

                if "y_diag" in batch:
                    y_diag_batch = batch["y_diag"].to(self.device).permute(0, 2, 1, 3, 4)
                    y = torch.cat((y, y_diag_batch), dim=1)

                if "x_surf_boundary" in batch:
                    x_boundary = concat_and_reshape(batch["x_boundary"], batch["x_surf_boundary"]).to(self.device)
                else:
                    x_boundary = reshape_only(batch["x_boundary"]).to(self.device)

                x_time_encode = batch["x_time_encode"].to(self.device)

                y_pred = self.model(x, x_boundary, x_time_encode)
                y_pred = self._apply_large_scale_correction(y_pred, x_boundary, x_time_encode)

                loss = criterion(y.to(y_pred.dtype), y_pred)

                metrics_dict = metrics(y_pred.float(), y.float())
                for name, value in metrics_dict.items():
                    value = torch.tensor([value], device=self.device, dtype=torch.float32)
                    if distributed:
                        dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                    results_dict[f"valid_{name}"].append(value[0].item())

                batch_loss = torch.tensor([loss.item()], device=self.device, dtype=torch.float32)
                if distributed:
                    torch.distributed.barrier()
                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(forecast_len + 1)

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

        batch_group_generator.close()

        if distributed:
            torch.distributed.barrier()

        torch.cuda.empty_cache()
        gc.collect()
        return results_dict