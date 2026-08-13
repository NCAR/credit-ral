"""
corrdiff_predict.py
-------------------------------------------------------
Inference script for the trained CorrDiff (EDM corrector) model.

Workflow:
  1. Load trained model weights and EMA shadow weights, swap EMA in.
  2. For each forecast initialization time:
     - Pull HR conditioning (mu + static/forcing) and LR predictors.
     - Generate N ensemble members via Heun sampler from pure noise.
     - Combine mu + residual to produce final HR field.
     - Inverse-transform and save each member as NetCDF.
"""

import os
import gc
import sys
import yaml
import glob
import logging
import warnings
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

import torch
import torch.distributed as dist
from torchvision import transforms as tforms

from credit.models import load_model
# from credit.seed import seed_everything
from credit.distributed import get_rank_info
from credit.data import (
    get_forward_data,
    drop_var_from_dataset,
    extract_month_day_hour,
    find_common_indices,
    encode_datetime64,
    filter_ds,
)
from credit.transforms.transforms_dscale import Normalize_Dscale, ToTensor_Dscale
from credit.parser import credit_main_parser
from credit.output import load_metadata, make_xarray, save_corrdiff_ensemble
from credit.pbs import launch_script, launch_script_mpi

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

logger = logging.getLogger(__name__)


# =========================================================================== #
# EDM components: preconditioning wrapper and Heun sampler
# =========================================================================== #
class EDMPrecond(torch.nn.Module):
    """Same preconditioning wrapper used at training time."""

    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(self, x_noisy, sigma, cond, x_time_encode):
        sigma = sigma.to(x_noisy.dtype).view(-1, 1, 1, 1, 1)
        sd = self.sigma_data

        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = sigma.log().flatten() / 4.0

        f = self.model(c_in * x_noisy, cond, c_noise, x_time_encode)
        return c_skip * x_noisy + c_out * f


@torch.no_grad()
def edm_sample(
    denoiser,
    cond,
    x_time_encode,
    shape,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    device="cuda",
):
    """Heun 2nd-order sampler for EDM corrector.

    Returns one residual sample of the given shape.
    """
    step_indices = torch.arange(num_steps, device=device, dtype=torch.float32)
    sigmas = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])

    x = torch.randn(shape, device=device) * sigmas[0]

    for i in range(num_steps):
        s_cur = sigmas[i].expand(shape[0])
        s_next = sigmas[i + 1].expand(shape[0])

        denoised = denoiser(x, s_cur, cond, x_time_encode)
        d_cur = (x - denoised) / s_cur.view(-1, 1, 1, 1, 1)
        x_next = x + (s_next - s_cur).view(-1, 1, 1, 1, 1) * d_cur

        if i < num_steps - 1:
            denoised_next = denoiser(x_next, s_next, cond, x_time_encode)
            d_next = (x_next - denoised_next) / s_next.view(-1, 1, 1, 1, 1)
            x_next = x + (s_next - s_cur).view(-1, 1, 1, 1, 1) * 0.5 * (d_cur + d_next)

        x = x_next

    return x


# =========================================================================== #
# EMA loading: handles both per-rank shards (FSDP-trained) and embedded format
# =========================================================================== #
def _gather_ema_from_shards(save_loc, device):
    """Find and gather per-rank EMA shard files into a single full state dict.

    For models trained with FSDP, EMA was saved as ema_checkpoint_rank{NNNN}.pt
    files (one per training rank).  At inference time we typically run on
    fewer GPUs (often just 1), so we concatenate the shards back into the
    full parameter tensors.

    Returns:
        dict: full EMA state with concatenated shadow tensors, or None if
            no shard files are found.
    """
    shard_paths = sorted(glob.glob(os.path.join(save_loc, "ema_checkpoint_rank*.pt")))
    if not shard_paths:
        return None

    logger.info(f"Found {len(shard_paths)} per-rank EMA shard files")

    # Load all shards.
    shards = [torch.load(p, map_location="cpu") for p in shard_paths]

    # Concatenate shadow tensors across ranks (FSDP shards along dim 0 of
    # the flattened param).
    full_shadow = {}
    param_names = list(shards[0]["shadow"].keys())
    for name in param_names:
        shard_tensors = [s["shadow"][name] for s in shards]
        # Each shard is a 1-D flat slice of the original parameter.
        full_flat = torch.cat([t.flatten() for t in shard_tensors], dim=0)
        full_shadow[name] = full_flat
        # Note: the flat tensor will be reshaped to match the live param
        # at copy time (see _copy_ema_to_model).

    return {
        "shadow": full_shadow,
        "shadow_buffers": shards[0].get("shadow_buffers", {}),
        "decay": shards[0].get("decay"),
    }

def _strip_fsdp_prefix(state_dict):
    """Strip FSDP, DDP, and activation-checkpoint wrapper artifacts from
    state-dict keys.

    Handles four prefix forms that can appear in our shards:
      1. Leading 'module.'                       - DDP-style outer wrapper
      2. '_fsdp_wrapped_module.'                 - FSDP root wrapper
      3. Embedded '._fsdp_wrapped_module.'       - FSDP sub-module auto-wraps
      4. Embedded '._checkpoint_wrapped_module.' - activation checkpointing

    Forms 2-4 can appear anywhere in the path, possibly nested.  We strip
    ALL occurrences via .replace() rather than just leading prefixes.
    """
    if not state_dict:
        return state_dict

    sample_key = next(iter(state_dict.keys()))
    has_wrapper = (
        "module." in sample_key
        or "_fsdp_wrapped_module." in sample_key
        or "_checkpoint_wrapped_module." in sample_key
    )
    if not has_wrapper:
        return state_dict

    stripped = {}
    for k, v in state_dict.items():
        new_k = k
        # Strip leading 'module.' (DDP wrapper)
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        # Strip all occurrences of FSDP and checkpoint wrappers,
        # wherever they appear in the path.
        new_k = new_k.replace("_fsdp_wrapped_module.", "")
        new_k = new_k.replace("_checkpoint_wrapped_module.", "")
        stripped[new_k] = v

    logger.info(
        f"Stripped wrappers from {len(stripped)} EMA shadow keys "
        f"(sample: '{sample_key}' -> '{next(iter(stripped.keys()))}')"
    )
    return stripped


def _copy_ema_to_model(ema_state, model):
    """Copy EMA shadow tensors into the model's live parameter buffers."""
    shadow = ema_state.get("shadow", {})
    shadow_buffers = ema_state.get("shadow_buffers", {})

    # Detect and strip FSDP/DDP wrapper prefixes from saved shadow keys.
    shadow = _strip_fsdp_prefix(shadow)
    shadow_buffers = _strip_fsdp_prefix(shadow_buffers)

    n_copied = 0
    n_skipped = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in shadow:
                n_skipped += 1
                continue

            src = shadow[name].to(dtype=param.dtype, device=param.device)

            # Flat -> reshape for FSDP-sharded parameters that got concatenated
            # back into 1-D form by _gather_ema_from_shards.
            if src.dim() == 1 and param.dim() > 1:
                if src.numel() >= param.numel():
                    src = src[: param.numel()].reshape(param.shape)
                else:
                    logger.warning(
                        f"EMA shadow for {name} has {src.numel()} elements but "
                        f"param needs {param.numel()}; skipping."
                    )
                    n_skipped += 1
                    continue
            elif src.shape != param.shape:
                logger.warning(
                    f"EMA shape mismatch for {name}: "
                    f"shadow={tuple(src.shape)} vs param={tuple(param.shape)}.  Skipping."
                )
                n_skipped += 1
                continue

            param.copy_(src)
            n_copied += 1

        for name, buf in model.named_buffers():
            if name in shadow_buffers:
                src = shadow_buffers[name].to(dtype=buf.dtype, device=buf.device)
                if src.shape == buf.shape:
                    buf.copy_(src)

    logger.info(f"EMA copy complete: {n_copied} parameters copied, {n_skipped} skipped")
    if n_copied == 0:
        logger.warning(
            "ZERO parameters were copied from EMA. Inference is using LIVE "
            "weights, not EMA-averaged weights. Check that EMA shadow keys "
            "match the model's named_parameters()."
        )


def load_inference_model(conf, device, rank=0):
    """Load the trained CorrDiff UNet, wrap in EDMPrecond, swap in EMA weights."""
    save_loc = os.path.expandvars(conf["save_loc"])

    model = load_model(conf).to(device)

    use_best = conf["predict"].get("use_best_weights", True)
    if use_best:
        ckpt_path = os.path.join(save_loc, "best_checkpoint.pt")
        model_ckpt_path = os.path.join(save_loc, "best_model_checkpoint.pt")
        if not os.path.exists(ckpt_path) and not os.path.exists(model_ckpt_path):
            logger.warning("best checkpoints not found; falling back to latest")
            use_best = False
    if not use_best:
        ckpt_path = os.path.join(save_loc, "checkpoint.pt")
        model_ckpt_path = os.path.join(save_loc, "model_checkpoint.pt")

    # ---- Load model weights ---- #
    # Detect FSDP-trained checkpoint by presence of model_checkpoint.pt.
    fsdp_trained = os.path.exists(model_ckpt_path)

    if fsdp_trained:
        model_state = torch.load(model_ckpt_path, map_location=device)
        if "model_state_dict" in model_state:
            model_state = model_state["model_state_dict"]
        model.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded FSDP-trained model weights from {model_ckpt_path}")
    elif os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        logger.info(f"Loaded model weights from {ckpt_path}")
    else:
        raise FileNotFoundError(f"No model checkpoint found in {save_loc}")

    # ---- Load EMA weights ---- #
    # Try per-rank shards first (FSDP-trained), then embedded format (non-FSDP).
    ema_loaded = False
    if conf["predict"].get("use_ema", True):
        ema_state = _gather_ema_from_shards(save_loc, device)

        if ema_state is not None:

            # import torch as _t
            # import glob as _glob 
            # shard_paths = sorted(_glob.glob(os.path.join(save_loc, "ema_checkpoint_rank*.pt")))
            # sample = _t.load(shard_paths[0], map_location="cpu")
            # logger.info(f"[EMA DEBUG] Shard file keys: {list(sample.keys())}")
            # logger.info(f"[EMA DEBUG] Shard file decay value: {sample.get('decay')!r}")
            shadow = ema_state["shadow"]
            # Apply the strip first so we compare stripped-vs-model
            stripped_shadow = _strip_fsdp_prefix(shadow)
        
            model_keys = set(name for name, _ in model.named_parameters())
            shadow_keys = set(stripped_shadow.keys())
        
            in_both = shadow_keys & model_keys
            only_shadow = shadow_keys - model_keys
            only_model = model_keys - shadow_keys
        
            logger.info(f"[EMA DEBUG] After stripping:")
            logger.info(f"[EMA DEBUG]   shadow_keys: {len(shadow_keys)}")
            logger.info(f"[EMA DEBUG]   model_keys:  {len(model_keys)}")
            logger.info(f"[EMA DEBUG]   matching:    {len(in_both)}")
            logger.info(f"[EMA DEBUG]   shadow-only: {len(only_shadow)}")
            logger.info(f"[EMA DEBUG]   model-only:  {len(only_model)}")
        
            logger.info(f"[EMA DEBUG] First 5 shadow-only keys (in shadow, not model):")
            for k in sorted(only_shadow)[:5]:
                logger.info(f"  {k}")
        
            logger.info(f"[EMA DEBUG] First 5 model-only keys (in model, not shadow):")
            for k in sorted(only_model)[:5]:
                logger.info(f"  {k}")
        
            logger.info(f"[EMA DEBUG] First 5 matching keys:")
            for k in sorted(in_both)[:5]:
                logger.info(f"  {k}")
        
        if ema_state is None and os.path.exists(ckpt_path):
            # Try embedded format.
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ema_state = ckpt.get("ema_state_dict")
            if ema_state is not None:
                logger.info(f"Found EMA state embedded in {ckpt_path}")

        if ema_state is not None:
            _copy_ema_to_model(ema_state, model)
            ema_loaded = True
            if "decay" in ema_state and ema_state["decay"] is not None:
                logger.info(f"EMA loaded (decay={ema_state['decay']})")
            else:
                logger.info("EMA loaded")

    if not ema_loaded:
        logger.warning(
            "No EMA weights found; using live model weights.  "
            "Sample quality may be noticeably worse than EMA-based inference."
        )

    # ---- Wrap with EDM preconditioning ---- #
    diff_conf = conf["trainer"].get("diffusion", {})
    sigma_data = diff_conf.get("sigma_data", 0.5)
    denoiser = EDMPrecond(model, sigma_data=sigma_data).to(device)
    denoiser.eval()

    return denoiser


# =========================================================================== #
# Inference dataset
# =========================================================================== #
class Dscale_Inference_Dataset:
    """Inference-only dataset.  Given a forecast initialization timestamp,
    returns the conditioning needed by the EDM denoiser."""

    def __init__(self, conf, transform=None):
        self.conf = conf
        self.transform = transform

        d = conf["data"]
        dscale = d["dscale_input"]

        self.varname_upper_air_HR = d["variables"]
        self.varname_surface_HR = d["surface_variables"]
        self.varname_dyn_forcing = d.get("dynamic_forcing_variables", [])
        self.varname_forcing = d.get("forcing_variables", [])
        self.varname_static = d.get("static_variables", [])
        self.varname_upper_air_LR = dscale["variables"]
        self.varname_surface_LR = dscale["surface_variables"]
        self.history_len = d.get("history_len", 1)
        self.levels_LR = dscale.get("levels", 1)
        self.levels_HR = d.get("levels", 1)

        hr_files = sorted(glob.glob(d["save_loc"]))
        hr_files_surf = sorted(glob.glob(d["save_loc_surface"]))
        self.ds_hr_upper = [filter_ds(get_forward_data(f), self.varname_upper_air_HR)
                            for f in hr_files]
        self.ds_hr_surf = [filter_ds(get_forward_data(f), self.varname_surface_HR)
                           for f in hr_files_surf]

        lr_files = sorted(glob.glob(dscale["save_loc"]))
        lr_files_surf = sorted(glob.glob(dscale["save_loc_surface"]))
        self.ds_lr_upper = [filter_ds(get_forward_data(f), self.varname_upper_air_LR)
                            for f in lr_files]
        self.ds_lr_surf = [filter_ds(get_forward_data(f), self.varname_surface_LR)
                           for f in lr_files_surf]

        if d.get("save_loc_forcing"):
            self.ds_forcing = drop_var_from_dataset(
                get_forward_data(d["save_loc_forcing"]), self.varname_forcing
            ).load()
        else:
            self.ds_forcing = None

        if d.get("save_loc_static"):
            self.ds_static = drop_var_from_dataset(
                get_forward_data(d["save_loc_static"]), self.varname_static
            ).load()
        else:
            self.ds_static = None

        if d.get("save_loc_dynamic_forcing"):
            dyn_files = sorted(glob.glob(d["save_loc_dynamic_forcing"]))
            self.ds_dyn_forcing = [
                filter_ds(get_forward_data(f), self.varname_dyn_forcing) for f in dyn_files
            ]
        else:
            self.ds_dyn_forcing = None

        self._hr_time_index = self._build_time_index(self.ds_hr_upper)
        self._lr_time_index = self._build_time_index(self.ds_lr_upper)

    @staticmethod
    def _build_time_index(ds_list):
        idx = {}
        for fi, ds in enumerate(ds_list):
            for ti, t in enumerate(ds["time"].values):
                idx[np.datetime64(t)] = (fi, ti)
        return idx

    def list_init_times(self):
        return np.array(sorted(self._hr_time_index.keys()))

    def get_sample(self, init_time):
        init_time = np.datetime64(init_time)

        if init_time not in self._lr_time_index:
            raise KeyError(f"LR file has no data at {init_time}")
        lr_fi, lr_ti = self._lr_time_index[init_time]
        ds_lr_up = self.ds_lr_upper[lr_fi].isel(time=slice(lr_ti, lr_ti + self.history_len))
        ds_lr_sf = self.ds_lr_surf[lr_fi].isel(time=slice(lr_ti, lr_ti + self.history_len))
        ds_lr = xr.merge([ds_lr_up, ds_lr_sf]).load()

        hr_fi, hr_ti = self._hr_time_index[init_time]
        ds_hr_up = self.ds_hr_upper[hr_fi].isel(time=slice(hr_ti, hr_ti + self.history_len))
        ds_hr_sf = self.ds_hr_surf[hr_fi].isel(time=slice(hr_ti, hr_ti + self.history_len))
        ds_hr = xr.merge([ds_hr_up, ds_hr_sf]).load()

        hr_forcing = xr.Dataset(coords={k: v.copy(deep=False) for k, v in ds_hr.coords.items()})
        if self.ds_dyn_forcing is not None:
            ds_dyn = self.ds_dyn_forcing[hr_fi].isel(time=slice(hr_ti, hr_ti + self.history_len))
            hr_forcing = hr_forcing.merge(ds_dyn.load())
        if self.ds_forcing is not None:
            mdh_forcing = extract_month_day_hour(np.array(self.ds_forcing["time"]))
            mdh_hr = extract_month_day_hour(np.array(ds_hr["time"]))
            ind, _ = find_common_indices(mdh_forcing, mdh_hr)
            fsub = self.ds_forcing.isel(time=ind)
            fsub["time"] = ds_hr["time"]
            hr_forcing = hr_forcing.merge(fsub)
        if self.ds_static is not None:
            stat = self.ds_static.expand_dims(dim={"time": len(ds_hr["time"])})
            stat = stat.assign_coords({"time": ds_hr["time"]})
            hr_forcing = hr_forcing.merge(stat)

        time_encode = encode_datetime64(ds_lr["time"].values)

        sample = {
            "HR_input": hr_forcing,
            "HR_target": ds_hr,
            "LR_input": ds_lr,
            "time_encode": time_encode,
            "datetime_index": ds_hr["time"].values.astype("datetime64[s]").astype(int),
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# =========================================================================== #
# Shape helpers: ensure tensors have the right rank for concat_and_reshape
# =========================================================================== #
def _ensure_upper_air_5d(t):
    """Make sure an upper-air tensor has shape (B, time, var, level, lat, lon).

    The training dataloader produces 6-D, but ToTensor_Dscale may emit fewer
    dims depending on configuration.  We normalize to the expected layout.
    """
    if t.dim() == 5:
        # (time, var, level, lat, lon) — add batch.
        return t.unsqueeze(0)
    elif t.dim() == 6:
        # Already (B, time, var, level, lat, lon).
        return t
    elif t.dim() == 4:
        # (time, var, lat, lon) — add level then batch.
        # Assume one level (matches config when levels=1).
        return t.unsqueeze(2).unsqueeze(0)
    elif t.dim() == 3:
        # (var, lat, lon) — add time, level, batch.
        return t.unsqueeze(0).unsqueeze(2).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected upper-air tensor rank: {t.dim()}, shape={tuple(t.shape)}")


def _ensure_surface_5d(t):
    """Make sure a surface tensor has shape (B, time, var, lat, lon)."""
    if t.dim() == 4:
        # (time, var, lat, lon) — add batch.
        return t.unsqueeze(0)
    elif t.dim() == 5:
        return t
    elif t.dim() == 3:
        # (var, lat, lon) — add time, batch.
        return t.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected surface tensor rank: {t.dim()}, shape={tuple(t.shape)}")


def _concat_upper_and_surface(x_upper, x_surf):
    """Mimic concat_and_reshape: fold (var, level) into channels and concat
    with surface.

    Input shapes:
        x_upper: (B, time, var, level, lat, lon)
        x_surf:  (B, time, var, lat, lon)
    Output shape:
        (B, var_total, time, lat, lon)  where var_total = upper.var*upper.level + surf.var
    """
    B, T, V, L, H, W = x_upper.shape
    # Combine var and level into a single channel dim.
    x_upper_flat = x_upper.reshape(B, T, V * L, H, W)
    # Concatenate along channel dim.
    combined = torch.cat([x_upper_flat, x_surf], dim=2)
    # Permute to (B, channels, time, H, W).
    return combined.permute(0, 2, 1, 3, 4).contiguous()


def assemble_conditioning(sample, device):
    """Build (mu, cond, x_time_encode, y) for the EDM denoiser.

    Output shapes:
        mu:           (1, C_target, T, H, W)
        cond:         (1, C_cond, T, H, W) — mu + forcing/static
        x_time_encode: (1, D_time)
        y:            (1, C_target, T, H, W) — ground truth
    """
    # ---- mu (from LR predictors) ---- #
    x_LR = _ensure_upper_air_5d(sample["x_LR"]).to(device)
    if "x_surf_LR" in sample:
        x_surf_LR = _ensure_surface_5d(sample["x_surf_LR"]).to(device)
        mu = _concat_upper_and_surface(x_LR, x_surf_LR)
    else:
        # Just upper, no surface.
        B, T, V, L, H, W = x_LR.shape
        mu = x_LR.reshape(B, T, V * L, H, W).permute(0, 2, 1, 3, 4).contiguous()

    # ---- HR forcing/static, concatenated to mu to form cond ---- #
    if "x_forcing_static_HR" in sample:
        xf = sample["x_forcing_static_HR"]
        # Expected shape (B, time, var, lat, lon) after batching.
        xf = _ensure_surface_5d(xf).to(device).permute(0, 2, 1, 3, 4)
        cond = torch.cat((mu, xf), dim=1)
    else:
        cond = mu

    # ---- y (HR ground truth) ---- #
    y_HR = _ensure_upper_air_5d(sample["y_HR"]).to(device)
    if "y_surf_HR" in sample:
        y_surf_HR = _ensure_surface_5d(sample["y_surf_HR"]).to(device)
        y = _concat_upper_and_surface(y_HR, y_surf_HR)
    else:
        B, T, V, L, H, W = y_HR.shape
        y = y_HR.reshape(B, T, V * L, H, W).permute(0, 2, 1, 3, 4).contiguous()

    # ---- Time encoding ---- #
    x_time_encode = sample["x_time_encode"]
    if not isinstance(x_time_encode, torch.Tensor):
        x_time_encode = torch.as_tensor(x_time_encode)
    if x_time_encode.dim() == 1:
        x_time_encode = x_time_encode.unsqueeze(0)
    x_time_encode = x_time_encode.to(device).float()

    return mu, cond, x_time_encode, y


# =========================================================================== #
# Main
# =========================================================================== #
def main(conf):
    save_loc = os.path.expandvars(conf["save_loc"])
    rank, world_rank, world_size = get_rank_info(conf.get("predict", {}).get("mode", "none"))
    device = (torch.device(f"cuda:{rank % torch.cuda.device_count()}")
              if torch.cuda.is_available() else torch.device("cpu"))
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())

    # seed_everything(conf["seed"])

    denoiser = load_inference_model(conf, device, rank=rank)

    state_transformer = Normalize_Dscale(conf)
    to_tensor = ToTensor_Dscale(conf)
    transforms_combined = tforms.Compose([state_transformer, to_tensor])

    dataset = Dscale_Inference_Dataset(conf, transform=transforms_combined)

    test_years_range = conf["predict"]["forecasts"]["year_range"]
    test_years = list(range(test_years_range[0], test_years_range[1]))

    all_init_times = dataset.list_init_times()
    init_times = [t for t in all_init_times
                  if int(np.datetime_as_string(t, unit="Y")) in test_years]

    start_ind = conf["predict"]["forecasts"].get("start_ind", 0)
    n_steps = conf["predict"]["forecasts"].get("pred_step", len(init_times))
    init_times = init_times[start_ind:start_ind + n_steps]
    print(init_times)
    logger.info(f"Running inference on {len(init_times)} initialization times")

    n_ensemble = conf["predict"].get("ensemble_size", 16)
    num_steps = conf["predict"].get("sampler_steps", 18)
    sigma_min = conf["predict"].get("sigma_min", 0.002)
    sigma_max = conf["predict"].get("sigma_max", 80.0)
    rho = conf["predict"].get("rho", 7.0)
    output_dir = os.path.expandvars(conf["predict"]["save_forecast"])
    os.makedirs(output_dir, exist_ok=True)

    meta_data = load_metadata(conf)

    grid_ds = dataset.ds_hr_upper[0]
    south_north = grid_ds["south_north"].values if "south_north" in grid_ds.coords else None
    west_east = grid_ds["west_east"].values if "west_east" in grid_ds.coords else None

    for i_init, init_time in enumerate(init_times):
        logger.info(f"[{i_init+1}/{len(init_times)}] init_time={init_time}")
    
        try:
            sample = dataset.get_sample(init_time)
        except KeyError as e:
            logger.warning(f"Skipping {init_time}: {e}")
            continue
    
        mu, cond, x_time_encode, y_true = assemble_conditioning(sample, device)
    
        n_target_channels = y_true.shape[1]
        residual_shape = (1, n_target_channels, *y_true.shape[2:])
    
        # Generate ensemble.
        ensemble_members = []
        for i_member in range(n_ensemble):
            r_pred = edm_sample(
                denoiser=denoiser,
                cond=cond,
                x_time_encode=x_time_encode,
                shape=residual_shape,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                device=device,
            )
            # New code (with zero-padding):
            n_mu_channels = mu.shape[1]            # 6
            n_target_channels = r_pred.shape[1]    # 7
            
            if n_mu_channels < n_target_channels:
                pad_shape = list(mu.shape)
                pad_shape[1] = n_target_channels - n_mu_channels
                mu_padded = torch.cat([mu, torch.zeros(pad_shape, device=mu.device,
                                                        dtype=mu.dtype)], dim=1)
            else:
                mu_padded = mu[:, :n_target_channels]
            
            y_pred = mu_padded + r_pred
            ensemble_members.append(y_pred.cpu())
    
        # Stack into (N_members, C, T, H, W).
        ensemble = torch.cat(ensemble_members, dim=0)
        # logger.info(f"Pre-inverse ensemble shape: {tuple(ensemble.shape)}")
        
        # Inverse-transform to physical units.
        ensemble_phys = state_transformer.inverse_transform(ensemble).detach()
        # logger.info(f"Post-inverse ensemble shape: {tuple(ensemble_phys.shape)}")
    
        # ---- One unified save call replaces the per-member loop ---- #
        save_corrdiff_ensemble(
            ensemble_phys=ensemble_phys,
            init_time=init_time,
            south_north=south_north,
            west_east=west_east,
            conf=conf,
            meta_data=meta_data,
        )

    logger.info("Inference complete.")

    # ---- Clean shutdown ---- #
    # Free GPU memory and run garbage collection so any lingering tensor
    # references are released before exit.
    try:
        del denoiser
    except NameError:
        pass
    torch.cuda.empty_cache()
    gc.collect()

    # Tear down the distributed process group if it was initialized.
    # Without this, NCCL backends can leave hanging threads that prevent
    # clean process exit.
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()    # ensure all ranks reach this point
        except Exception as e:
            logger.warning(f"Final barrier failed (non-fatal): {e}")
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")

    # Flush all log handlers so final messages reach the output file
    # before we exit.
    for handler in logging.root.handlers:
        try:
            handler.flush()
        except Exception:
            pass

    if rank == 0:
        logger.info("exit")

    # Explicit exit ensures the process terminates even if some background
    # thread (e.g., dataloader workers, NCCL watchdog) would otherwise
    # keep it alive.
    sys.exit(0)


if __name__ == "__main__":
    parser = ArgumentParser(description="CorrDiff inference")
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument("-l", "--launch", type=int, default=0)
    args = parser.parse_args()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(ch)

    with open(args.model_config) as cf:
        conf = yaml.safe_load(cf)
    conf = credit_main_parser(conf, parse_training=False, parse_predict=True, print_summary=True)

    if args.launch:
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            launch_script(args.model_config, script_path)
        else:
            launch_script_mpi(args.model_config, script_path, "nccl")
        sys.exit()
    main(conf)
    