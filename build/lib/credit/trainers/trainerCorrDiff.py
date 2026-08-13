"""CorrDiff EDM corrector trainer with constant-decay EMA.

Stage-2 diffusion training for residual r = y_HR - mu, where mu is the
frozen stage-1 regression output passed through the batch as `x_surf_LR`.
Implements Karras et al. 2022 (EDM) preconditioning and loss, plus an
exponential moving average of model weights for stable sampling.

EMA uses a fixed decay (no warmup ramp).  Only the shadow tensors persist
across restarts; no step counter is tracked.

Also handles batches where the LR lookup produced empty tensors (data gaps
between HR and LR coverage) by skipping those batches in a rank-synchronized
way so FSDP collectives don't deadlock.
"""

import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import IterableDataset

import optuna
from credit.data import concat_and_reshape, reshape_only
from credit.scheduler import update_on_batch
from credit.trainers.utils import accum_log, cycle
from credit.trainers.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# EDM preconditioning wrapper (Karras et al. 2022, Eq. 7 and Table 1).
# --------------------------------------------------------------------------- #
class EDMPrecond(torch.nn.Module):
    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(self, x_noisy, sigma, cond, x_time_encode):
        sigma = sigma.to(x_noisy.dtype).view(-1, 1, 1, 1, 1)
        sd = self.sigma_data

        c_skip  = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out   = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in    = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = sigma.log().flatten() / 4.0

        f = self.model(c_in * x_noisy, cond, c_noise, x_time_encode)
        return c_skip * x_noisy + c_out * f


# --------------------------------------------------------------------------- #
# Loss: EDM denoising score matching with sigma-weighted MSE.
# --------------------------------------------------------------------------- #
class EDMLoss:
    def __init__(self, p_mean=-1.2, p_std=1.2, sigma_data=0.5, channel_weights=None):
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data
        # Optional per-channel weighting. If None, all channels are equally weighted.
        self.channel_weights = channel_weights

    def sample_sigma(self, batch_size, device, dtype):
        rnd = torch.randn(batch_size, device=device, dtype=dtype)
        return (rnd * self.p_std + self.p_mean).exp()

    def __call__(self, denoiser, r_clean, cond, x_time_encode):
        # ---- Sample sigma and the corresponding sigma-weighting ---- #
        sigma = self.sample_sigma(r_clean.shape[0], r_clean.device, r_clean.dtype)
        sigma_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        sigma_weight = sigma_weight.view(-1, 1, 1, 1, 1)

        # ---- Add noise to the clean residual and pass through denoiser ---- #
        noise = torch.randn_like(r_clean) * sigma.view(-1, 1, 1, 1, 1)
        r_noisy = r_clean + noise
        r_pred = denoiser(r_noisy, sigma, cond, x_time_encode)

        # ---- Per-channel weighting (optional) ---- #
        # Boosts under-trained channels (smaller r_std) so they contribute more
        # to the overall loss.
        if self.channel_weights is not None:
            ch_w = torch.tensor(
                self.channel_weights,
                device=r_clean.device,
                dtype=r_clean.dtype,
            ).view(1, -1, 1, 1, 1)
            combined_weight = sigma_weight * ch_w
        else:
            combined_weight = sigma_weight

        # ---- Compute weighted squared error ---- #
        return (combined_weight * (r_pred - r_clean) ** 2).mean()


# --------------------------------------------------------------------------- #
# Constant-decay EMA.
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.shadow_buffers = {
            name: buf.detach().clone()
            for name, buf in model.named_buffers()
        }
        self._backup = None
        self._backup_buffers = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(d).add_(param.detach(), alpha=1 - d)
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                self.shadow_buffers[name].copy_(buf.detach())

    def state_dict(self):
        return {
            "shadow": self.shadow,
            "shadow_buffers": self.shadow_buffers,
            "decay": self.decay,
        }

    def load_state_dict(self, sd):
        self.shadow = sd["shadow"]
        self.shadow_buffers = sd["shadow_buffers"]
        if "decay" in sd:
            self.decay = sd["decay"]

    @torch.no_grad()
    def copy_to(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                buf.copy_(self.shadow_buffers[name])

    @torch.no_grad()
    def store(self, model):
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        self._backup_buffers = {
            name: buf.detach().clone()
            for name, buf in model.named_buffers()
        }

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            return
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        for name, buf in model.named_buffers():
            if name in self._backup_buffers:
                buf.copy_(self._backup_buffers[name])
        self._backup = None
        self._backup_buffers = None


# --------------------------------------------------------------------------- #
# Helper: detect empty/degenerate tensors in a batch dict.
# Returns True if any required tensor has a zero dimension.
# --------------------------------------------------------------------------- #
def _batch_has_empty_tensors(batch, required_keys):
    """Check whether any required tensor in the batch has size 0 in any dim.

    Used to detect samples where the dataset's LR/HR lookup returned empty
    slices (e.g., HR timestamp falls outside LR file coverage).  When this
    happens, the resulting tensor has shape (0, ...) and downstream torch.cat
    operations fail.
    """
    for key in required_keys:
        if key not in batch:
            continue
        t = batch[key]
        if not isinstance(t, torch.Tensor):
            continue
        if t.numel() == 0:
            return True
        if any(s == 0 for s in t.shape):
            return True
    return False


# --------------------------------------------------------------------------- #
# Helper: synchronize "skip this batch" decision across all ranks.
# --------------------------------------------------------------------------- #
def _all_ranks_skip(local_skip, distributed, device):
    """Coordinate batch-skipping across all ranks.

    If ANY rank wants to skip a batch (because of empty tensors), ALL ranks
    must skip together.  Otherwise some ranks would enter the training step's
    NCCL collectives while others wouldn't, causing a deadlock.

    Returns True if any rank flagged this batch as invalid.
    """
    if not distributed:
        return local_skip

    flag = torch.tensor(
        [1 if local_skip else 0], dtype=torch.long, device=device
    )
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    return flag.item() > 0


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int, conf: dict = None):
        super().__init__(model, rank)
        
        diff_conf = (conf or {}).get("trainer", {}).get("diffusion", {})
        sigma_data = diff_conf.get("sigma_data", 0.5)
        p_mean = diff_conf.get("p_mean", -1.2)
        p_std = diff_conf.get("p_std", 1.2)
        channel_weights = diff_conf.get("channel_weights", None)
        
        self.denoiser = EDMPrecond(self.model, sigma_data=sigma_data)
        self.edm_loss = EDMLoss(
            p_mean=p_mean,
            p_std=p_std,
            sigma_data=sigma_data,
            channel_weights=channel_weights,
        )
        
        if channel_weights is not None and rank == 0:
            logger.info(f"Per-channel loss weights enabled: {channel_weights}")
            
        # ---- EMA setup ---- #
        ema_conf = (conf or {}).get("trainer", {}).get("ema", {})
        self.use_ema = True   # ema_conf.get("activate", True)

        if self.use_ema:
            self.ema = EMA(self.model, decay=ema_conf.get("decay", 0.9999))
            logger.info(f"EMA enabled (decay={self.ema.decay})")
        else:
            self.ema = None
            logger.info("EMA disabled")

        # ---- Tracking: count how many batches were skipped due to data gaps ---- #
        self._n_skipped_batches = 0

    # ------------------------------------------------------------------- #
    def _assemble_batch(self, batch):
        """Build (cond, r_clean, x_time_encode, y) from a raw batch dict.
    
        mu = LR predictors (Q, SP, T2, U10, V10, PWAT) upsampled to HR.
        The HR target has 7 channels: 6 residual-style (Q, SP, T2, U10, V10, PWAT)
        + 1 direct-prediction (precip).  We compute residual for the first 6 and
        use full HR for the 7th.
    
        Returns None if the batch contains empty tensors (HR-LR coverage gap).
        Caller must check return value and skip via rank-coordinated logic so
        FSDP collectives don't deadlock.
        """
        # ---- Pre-flight check for empty tensors ---- #
        # Detects samples where the dataset's LR/HR lookup returned empty slices
        # (e.g., HR timestamp falls outside LR file coverage at year boundaries).
        # Returning None here lets the trainer coordinate a synchronized skip.
        required = ["x_LR", "y_HR", "x_time_encode"]
        if "x_surf_LR" in batch:
            required.append("x_surf_LR")
        if "x_forcing_static_HR" in batch:
            required.append("x_forcing_static_HR")
        if "y_surf_HR" in batch:
            required.append("y_surf_HR")
        if "y_diag_HR" in batch:
            required.append("y_diag_HR")
    
        for key in required:
            if key not in batch:
                continue
            t = batch[key]
            if not isinstance(t, torch.Tensor):
                continue
            if t.numel() == 0 or any(s == 0 for s in t.shape):
                return None
    
        # ---- Build mu from LR predictors (6 channels: 1 upper + 5 surface) ---- #
        if "x_surf_LR" in batch:
            mu = concat_and_reshape(batch["x_LR"], batch["x_surf_LR"]).to(self.device)
        else:
            mu = reshape_only(batch["x_LR"]).to(self.device)
    
        # ---- cond = mu + statics/forcing (6 + 2 = 8 channels) ---- #
        if "x_forcing_static_HR" in batch:
            x_forcing = batch["x_forcing_static_HR"].to(self.device).permute(0, 2, 1, 3, 4)
            cond = torch.cat((mu, x_forcing), dim=1)
        else:
            cond = mu
    
        # ---- Build HR target y (7 channels: 1 upper + 6 surface) ---- #
        if "y_surf_HR" in batch:
            y = concat_and_reshape(batch["y_HR"], batch["y_surf_HR"]).to(self.device)
        else:
            y = reshape_only(batch["y_HR"]).to(self.device)
    
        if "y_diag_HR" in batch:
            y_diag = batch["y_diag_HR"].to(self.device).permute(0, 2, 1, 3, 4).float()
            y = torch.cat((y, y_diag), dim=1)
    
        # ---- Compute mixed residual ---- #
        # Channel order in y:  [Q, SP, T2, U10, V10, PWAT, precip]
        # Channel order in mu: [Q, SP, T2, U10, V10, PWAT]    (no precip)
        #
        # For first 6 channels: residual = y - mu
        # For 7th channel (precip): residual = y (i.e., predict full HR precip)
        #
        # We pad mu with a zero channel for precip to keep shape consistent.
        n_mu_channels = mu.shape[1]      # 6
        n_y_channels = y.shape[1]        # 7
    
        if n_mu_channels < n_y_channels:
            pad_shape = list(mu.shape)
            pad_shape[1] = n_y_channels - n_mu_channels
            mu_padded = torch.cat(
                [mu, torch.zeros(pad_shape, device=mu.device, dtype=mu.dtype)],
                dim=1,
            )
        else:
            mu_padded = mu[:, :n_y_channels]
    
        r_clean = (y - mu_padded).to(self.device)
    
        x_time_encode = batch["x_time_encode"].to(self.device)
        return cond, r_clean, x_time_encode, y

    # ------------------------------------------------------------------- #
    def ema_state_dict(self):
        return self.ema.state_dict() if self.ema is not None else None

    def load_ema_state_dict(self, sd):
        if self.ema is not None and sd is not None:
            self.ema.load_state_dict(sd)
            if self.rank == 0:
                logger.info(f"Loaded EMA shadow weights (decay={self.ema.decay})")
        else:
            if self.rank == 0:
                logger.info(f"EMA load skipped: ema={self.ema is not None}, sd={sd is not None}")

    # ------------------------------------------------------------------- #
    def train_one_epoch(self, epoch, conf, trainloader, optimizer, criterion,
                        scaler, scheduler, metrics):
        batches_per_epoch = conf["trainer"]["batches_per_epoch"]
        grad_accum_every  = conf["trainer"]["grad_accum_every"]
        grad_max_norm     = conf["trainer"]["grad_max_norm"]
        forecast_len      = conf["data"]["forecast_len"]
        amp               = conf["trainer"]["amp"]
        distributed       = conf["trainer"]["mode"] in ["fsdp", "ddp"]

        total_time_steps = conf["data"].get("total_time_steps", forecast_len)
        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        if (conf["trainer"]["use_scheduler"]
                and conf["trainer"]["scheduler"]["scheduler_type"] == "lambda"):
            scheduler.step()

        if not isinstance(trainloader.dataset, IterableDataset):
            batches_per_epoch = (batches_per_epoch
                                 if 0 < batches_per_epoch < len(trainloader)
                                 else len(trainloader))

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch), total=batches_per_epoch,
            leave=True, disable=self.rank > 0,
        )

        results_dict = defaultdict(list)
        dl = cycle(trainloader)

        n_skipped_this_epoch = 0

        for i in batch_group_generator:
            batch = next(dl)
            logs = {}

            # ---- Pre-flight: detect empty batches and synchronize across ranks ---- #
            # _assemble_batch returns None when the batch has empty tensors.
            # We must coordinate the skip decision across ranks BEFORE any
            # collective operation (forward/backward/all_reduce) — otherwise
            # ranks that skip and ranks that proceed will deadlock at the
            # next collective.
            assembled = self._assemble_batch(batch)
            local_skip = (assembled is None)
            should_skip = _all_ranks_skip(local_skip, distributed, self.device)

            if should_skip:
                n_skipped_this_epoch += 1
                self._n_skipped_batches += 1
                if self.rank == 0 and n_skipped_this_epoch <= 5:
                    logger.warning(
                        f"[Epoch {epoch} batch {i}] Skipped due to empty LR/HR "
                        f"tensor (data gap).  Skipped so far this epoch: "
                        f"{n_skipped_this_epoch}"
                    )
                continue

            cond, r_clean, x_time_encode, y = assembled

            with autocast(enabled=amp):
                # ---- EDM loss on the residual ---- #
                loss = self.edm_loss(self.denoiser, r_clean, cond, x_time_encode)

                # ---- Reportable metrics: single-step denoise at sigma_data ---- #
                with torch.no_grad():
                    sigma_eval = torch.full(
                        (r_clean.shape[0],), self.edm_loss.sigma_data,
                        device=r_clean.device, dtype=r_clean.dtype,
                    )
                    noise = torch.randn_like(r_clean) * sigma_eval.view(-1, 1, 1, 1, 1)
                    r_pred = self.denoiser(r_clean + noise, sigma_eval, cond, x_time_encode)
                    y_pred = cond[:, : y.shape[1]] + r_pred

                    metrics_dict = metrics(y_pred.float(), y.float())
                    for name, value in metrics_dict.items():
                        value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                        if distributed:
                            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                        results_dict[f"train_{name}"].append(value[0].item())

                loss = loss.mean()
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

            if self.ema is not None:
                self.ema.update(self.model)

            batch_loss = torch.Tensor([logs["loss"]]).cuda(self.device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss"].append(batch_loss[0].item())

            if "forecast_hour" in batch:
                fh = batch["forecast_hour"].to(self.device)
                if distributed:
                    dist.all_reduce(fh, dist.ReduceOp.AVG, async_op=False)
                results_dict["train_forecast_len"].append(fh[-1].item() + 1)
            else:
                results_dict["train_forecast_len"].append(forecast_len + 1)

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                print("Invalid loss: {}".format(np.mean(results_dict["train_loss"])))
                raise optuna.TrialPruned()

            to_print = ("Epoch: {} train_loss: {:.6f} "
                        "train_acc: {:.6f} train_mae: {:.6f} forecast_len {:.6}").format(
                epoch,
                np.mean(results_dict["train_loss"]),
                np.mean(results_dict.get("train_acc", [0.0])),
                np.mean(results_dict.get("train_mae", [0.0])),
                np.mean(results_dict["train_forecast_len"]),
            )
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            if self.rank == 0:
                batch_group_generator.set_description(to_print)

            if (conf["trainer"]["use_scheduler"]
                    and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_batch):
                scheduler.step()

            if i >= batches_per_epoch and i > 0:
                break

        batch_group_generator.close()

        if self.rank == 0 and n_skipped_this_epoch > 0:
            logger.info(
                f"[Epoch {epoch}] Skipped {n_skipped_this_epoch} batches due "
                f"to data gaps (total across run: {self._n_skipped_batches})"
            )

        torch.cuda.empty_cache()
        gc.collect()
        return results_dict

    # ------------------------------------------------------------------- #
    def validate(self, epoch, conf, valid_loader, criterion, metrics):
        self.model.eval()

        valid_batches_per_epoch = conf["trainer"]["valid_batches_per_epoch"]
        forecast_len = conf["data"]["valid_forecast_len"]
        distributed  = conf["trainer"]["mode"] in ["fsdp", "ddp"]

        total_time_steps = conf["data"].get("total_time_steps", forecast_len)
        assert total_time_steps == 0, "This trainer supports `forecast_len=0` only"

        results_dict = defaultdict(list)

        if not isinstance(valid_loader.dataset, IterableDataset):
            valid_batches_per_epoch = (valid_batches_per_epoch
                                       if 0 < valid_batches_per_epoch < len(valid_loader)
                                       else len(valid_loader))

        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch), total=valid_batches_per_epoch,
            leave=True, disable=self.rank > 0,
        )

        dl = cycle(valid_loader)
        n_skipped_valid = 0

        if self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)

        try:
            for i in batch_group_generator:
                batch = next(dl)

                # ---- Same skip-coordination as training ---- #
                with torch.no_grad():
                    assembled = self._assemble_batch(batch)
                local_skip = (assembled is None)
                should_skip = _all_ranks_skip(local_skip, distributed, self.device)

                if should_skip:
                    n_skipped_valid += 1
                    continue

                cond, r_clean, x_time_encode, y = assembled

                with torch.no_grad():
                    loss = self.edm_loss(self.denoiser, r_clean, cond, x_time_encode)

                    sigma_eval = torch.full(
                        (r_clean.shape[0],), self.edm_loss.sigma_data,
                        device=r_clean.device, dtype=r_clean.dtype,
                    )
                    noise = torch.randn_like(r_clean) * sigma_eval.view(-1, 1, 1, 1, 1)
                    r_pred = self.denoiser(r_clean + noise, sigma_eval, cond, x_time_encode)
                    y_pred = cond[:, : y.shape[1]] + r_pred

                    metrics_dict = metrics(y_pred.float(), y.float())
                    for name, value in metrics_dict.items():
                        value = torch.Tensor([value]).cuda(self.device, non_blocking=True)
                        if distributed:
                            dist.all_reduce(value, dist.ReduceOp.AVG, async_op=False)
                        results_dict[f"valid_{name}"].append(value[0].item())

                    batch_loss = torch.Tensor([loss.item()]).cuda(self.device)
                    if distributed:
                        torch.distributed.barrier()
                    results_dict["valid_loss"].append(batch_loss[0].item())
                    results_dict["valid_forecast_len"].append(forecast_len + 1)

                    to_print = ("Epoch: {} valid_loss: {:.6f} "
                                "valid_acc: {:.6f} valid_mae: {:.6f}").format(
                        epoch,
                        np.mean(results_dict["valid_loss"]),
                        np.mean(results_dict.get("valid_acc", [0.0])),
                        np.mean(results_dict.get("valid_mae", [0.0])),
                    )
                    if self.rank == 0:
                        batch_group_generator.set_description(to_print)

                    if i >= valid_batches_per_epoch and i > 0:
                        break
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)

        batch_group_generator.close()

        if self.rank == 0 and n_skipped_valid > 0:
            logger.info(
                f"[Validation epoch {epoch}] Skipped {n_skipped_valid} batches "
                f"due to data gaps"
            )

        if distributed:
            torch.distributed.barrier()
        torch.cuda.empty_cache()
        gc.collect()
        return results_dict
        