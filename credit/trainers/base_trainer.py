"""
base_trainer.py
-------------------------------------------------------
"""

import os
import gc
import shutil
import logging
from collections import defaultdict

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from credit.models.checkpoint import TorchFSDPCheckpointIO, copy_checkpoint
from credit.scheduler import update_on_epoch
from credit.trainers.utils import cleanup

logger = logging.getLogger(__name__)


def _dist_barrier():
    """Safe barrier that's a no-op when distributed isn't initialized.

    Called after rank-0-only file writes to prevent non-rank-0 processes
    from racing past while rank 0 is still writing.  Without this, downstream
    NCCL collectives can time out because rank 0 isn't yet participating.
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class BaseTrainer(ABC):
    def __init__(self, model: torch.nn.Module, rank: int):
        """
        Abstract base class for training and validating machine learning models.

        Attributes:
            model (torch.nn.Module): The model to be trained.
            rank (int): The rank of the process in distributed training.
            device (torch.device): The device on which to train the model.
        """
        super(BaseTrainer, self).__init__()
        self.model = model
        self.rank = rank
        self.device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")

    @abstractmethod
    def train_one_epoch(
        self,
        epoch: int,
        conf: Dict[str, Any],
        trainloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        epoch: int,
        conf: Dict[str, Any],
        valid_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        raise NotImplementedError

    def save_checkpoint(self, save_loc: str, state_dict: Dict[str, Any]) -> None:
        if self.rank == 0:
            torch.save(state_dict, f"{save_loc}/checkpoint.pt")
            logger.info(f"Saved checkpoint to {save_loc}/checkpoint.pt")
        _dist_barrier()

    def save_fsdp_checkpoint(self, save_loc: str, state_dict: Dict[str, Any]) -> None:
        """
        Save a checkpoint for FSDP training.

        Args:
            save_loc (str): The location to save the checkpoint.
            state_dict (Dict[str, Any]): The state dictionary to save.
        """
        from credit.models.checkpoint import TorchFSDPCheckpointIO

        checkpoint_io = TorchFSDPCheckpointIO()

        checkpoint_io.save_unsharded_model(
            self.model,
            os.path.join(save_loc, "model_checkpoint.pt"),
            gather_dtensor=True,
            use_safetensors=False,
            rank=self.rank,
        )
        if self.rank == 0:
            logger.info(f"Saved FSDP model checkpoint to {save_loc}/model_checkpoint.pt")

        # EMA save: each rank writes its own shard.  No cross-rank collective,
        # no risk of NCCL timeout.  Restart at the same world size to reload.
        self._save_ema_shard(save_loc)

        if self.rank == 0:
            torch.save(state_dict, os.path.join(save_loc, "checkpoint.pt"))
            logger.info(f"Saved FSDP scheduler and scaler states to {save_loc}/checkpoint.pt")
        _dist_barrier()

    def _save_ema_shard(self, save_loc: str) -> None:
        """Save the rank-local EMA shadow tensors to a per-rank file.

        Each rank writes ema_checkpoint_rank{NNNN}.pt holding its slice of
        the EMA state.  Loading reassembles the full state from all shards.

        No-op when the trainer doesn't have EMA enabled.
        """
        if not hasattr(self, "ema") or getattr(self, "ema", None) is None:
            return

        ema_path = os.path.join(save_loc, f"ema_checkpoint_rank{self.rank:04d}.pt")
        torch.save({
            "shadow": self.ema.shadow,
            "shadow_buffers": self.ema.shadow_buffers,
            "decay": self.ema.decay,
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "rank": self.rank,
        }, ema_path)
        if self.rank == 0:
            logger.info(f"Saved per-rank EMA shards to {save_loc}/ema_checkpoint_rank*.pt")

    def fit(
        self,
        conf: Dict[str, Any],
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: Optimizer,
        train_criterion: torch.nn.Module,
        valid_criterion: torch.nn.Module,
        scaler: GradScaler,
        scheduler: LRScheduler,
        metrics: Dict[str, Any],
        rollout_scheduler: Optional[callable] = None,
        trial: bool = False,
    ) -> Dict[str, Any]:
        """
        Fit the model to the data.
        """

        # convert $USER to the actual user name
        conf["save_loc"] = save_loc = os.path.expandvars(conf["save_loc"])

        # training hyperparameters
        start_epoch = conf["trainer"]["start_epoch"]
        epochs = conf["trainer"]["epochs"]
        skip_validation = conf["trainer"]["skip_validation"] if "skip_validation" in conf["trainer"] else False
        flag_load_weights = conf["trainer"]["load_weights"]

        training_metric = conf["trainer"].get("training_metric", "train_loss" if skip_validation else "valid_loss")
        direction = conf["trainer"].get("training_metric_direction", "min")
        logger.info(f"The training metric being used is {training_metric} which has direction {direction}")
        direction = min if direction == "min" else max

        save_metric_vars = conf["trainer"].get("save_metric_vars", [])

        if "num_epoch" in conf["trainer"]:
            logger.info("The current job will run {} epochs max".format(conf["trainer"]["num_epoch"]))
        else:
            conf["trainer"]["num_epoch"] = 1e8

        if (start_epoch == 0) or (flag_load_weights is False):
            results_dict = defaultdict(list)
            if "train_one_epoch" in conf["trainer"] and conf["trainer"]["train_one_epoch"]:
                epochs = 1
        else:
            results_dict = defaultdict(list)
            saved_results = pd.read_csv(os.path.join(save_loc, "training_log.csv"))

            if "train_one_epoch" in conf["trainer"] and conf["trainer"]["train_one_epoch"]:
                start_epoch = len(saved_results)
                epochs = start_epoch + 1

            for key in saved_results.columns:
                if key == "index":
                    continue
                results_dict[key] = list(saved_results[key])

        count = 0
        for epoch in range(start_epoch, epochs):
            if count >= conf["trainer"]["num_epoch"]:
                logger.info("Completed {} epochs, exiting".format(conf["trainer"]["num_epoch"]))
                break

            # ========================= #
            # backup the previous epoch
            # ========================= #
            if count > 0 and conf["trainer"]["save_backup_weights"]:
                if self.rank == 0:
                    shutil.copyfile(
                        os.path.join(save_loc, "checkpoint.pt"),
                        os.path.join(save_loc, "backup_checkpoint.pt"),
                    )

                    if conf["trainer"]["mode"] == "fsdp":
                        shutil.copyfile(
                            os.path.join(save_loc, "model_checkpoint.pt"),
                            os.path.join(save_loc, "backup_model_checkpoint.pt"),
                        )
                        shutil.copyfile(
                            os.path.join(save_loc, "optimizer_checkpoint.pt"),
                            os.path.join(save_loc, "backup_optimizer_checkpoint.pt"),
                        )
                _dist_barrier()

            logger.info(f"Beginning epoch {epoch}")

            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)

            if not conf["trainer"]["skip_validation"]:
                with torch.no_grad():
                    if hasattr(valid_loader, "sampler") and hasattr(valid_loader.sampler, "set_epoch"):
                        valid_loader.sampler.set_epoch(epoch)

                    if hasattr(valid_loader.dataset, "set_epoch"):
                        valid_loader.dataset.set_epoch(epoch)

            ############
            #
            # Train
            #
            ############

            train_results = self.train_one_epoch(
                epoch, conf, train_loader, optimizer, train_criterion,
                scaler, scheduler, metrics,
            )

            ############
            #
            # Validation
            #
            ############

            if skip_validation:
                valid_results = train_results
            else:
                valid_results = self.validate(epoch, conf, valid_loader, valid_criterion, metrics)

            #################
            #
            # Save results
            #
            #################

            results_dict["epoch"].append(epoch)

            required_metrics = ["loss", "acc", "mae", "forecast_len"]
            if isinstance(save_metric_vars, list) and len(save_metric_vars) > 0:
                names = [key.replace("train_", "") for key in train_results.keys() if any(var in key for var in save_metric_vars)]
            elif isinstance(save_metric_vars, bool) and save_metric_vars:
                names = [key.replace("train_", "") for key in train_results.keys()]
            else:
                names = []
            names = list(set(names + required_metrics))

            for name in names:
                results_dict[f"train_{name}"].append(np.mean(train_results[f"train_{name}"]))
                if skip_validation:
                    continue
                results_dict[f"valid_{name}"].append(np.mean(valid_results[f"valid_{name}"]))
            results_dict["lr"].append(optimizer.param_groups[0]["lr"])

            if conf["trainer"]["use_scheduler"] and conf["trainer"]["scheduler"]["scheduler_type"] in update_on_epoch:
                if conf["trainer"]["scheduler"]["scheduler_type"] == "plateau":
                    scheduler.step(results_dict[training_metric][-1])
                else:
                    scheduler.step()

            max_len = max(len(lst) for lst in results_dict.values())

            padded_dict = OrderedDict()
            for key, lst in results_dict.items():
                if len(lst) < max_len:
                    padded_dict[key] = [np.nan] * (max_len - len(lst)) + lst
                else:
                    padded_dict[key] = lst

            df = pd.DataFrame.from_dict(padded_dict).reset_index()

            if self.rank == 0:
                if trial:
                    df.to_csv(
                        os.path.join(f"{save_loc}", "trial_results",
                                     f"training_log_{trial.number}.csv"),
                        index=False,
                    )
                else:
                    df.to_csv(os.path.join(f"{save_loc}", "training_log.csv"), index=False)

            ############
            #
            # Checkpoint
            #
            ############

            if not trial:
                if conf["trainer"]["mode"] != "fsdp":
                    if self.rank == 0:
                        logger.info(f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")
                        if conf["trainer"]["mode"] == "ddp":
                            model_state_dict = self.model.module.state_dict()
                        else:
                            model_state_dict = self.model.state_dict()
                        state_dict = {
                            "epoch": epoch,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict() if conf["trainer"]["use_scheduler"] else None,
                            "scaler_state_dict": scaler.state_dict(),
                        }
                        # Embed EMA state directly for non-FSDP saves.  No
                        # sharding needed; the trainer's helper returns the
                        # full state dict (or None if EMA is disabled).
                        if hasattr(self, "ema_state_dict"):
                            state_dict["ema_state_dict"] = self.ema_state_dict()

                        torch.save(state_dict, f"{save_loc}/checkpoint.pt")

                        if conf.get("trainer", {}).get("save_every_epoch", False):
                            copy_checkpoint(os.path.join(save_loc, "checkpoint.pt"), epoch)
                    _dist_barrier()

                else:
                    logger.info(f"Saving FSDP model, optimizer, grad scaler, learning rate scheduler, and EMA states to {save_loc}")

                    checkpoint_io = TorchFSDPCheckpointIO()

                    checkpoint_io.save_unsharded_model(
                        self.model,
                        os.path.join(save_loc, "model_checkpoint.pt"),
                        gather_dtensor=True,
                        use_safetensors=False,
                        rank=self.rank,
                    )
                    checkpoint_io.save_unsharded_optimizer(
                        optimizer,
                        os.path.join(save_loc, "optimizer_checkpoint.pt"),
                        gather_dtensor=True,
                        rank=self.rank,
                    )

                    state_dict = {
                        "epoch": epoch,
                        "scheduler_state_dict": scheduler.state_dict() if conf["trainer"]["use_scheduler"] else None,
                        "scaler_state_dict": scaler.state_dict(),
                    }

                    # Per-rank EMA save: each rank writes its own shadow shard.
                    # Avoids cross-rank collectives that were causing NCCL timeouts.
                    self._save_ema_shard(save_loc)

                    if self.rank == 0:
                        torch.save(state_dict, os.path.join(save_loc, "checkpoint.pt"))

                    if conf.get("trainer", {}).get("save_every_epoch", False) and self.rank == 0:
                        copy_checkpoint(os.path.join(save_loc, "model_checkpoint.pt"), epoch)

                    _dist_barrier()

            torch.cuda.empty_cache()
            gc.collect()
            count += 1

            if skip_validation:
                pass
            else:
                best_epoch = [i for i, j in enumerate(results_dict[training_metric]) if j == direction(results_dict[training_metric])][0]
                offset = epoch - best_epoch

                if offset == 0 and conf["trainer"]["save_best_weights"]:
                    if self.rank == 0:
                        shutil.copyfile(
                            os.path.join(save_loc, "checkpoint.pt"),
                            os.path.join(save_loc, "best_checkpoint.pt"),
                        )

                        if conf["trainer"]["mode"] == "fsdp":
                            shutil.copyfile(
                                os.path.join(save_loc, "model_checkpoint.pt"),
                                os.path.join(save_loc, "best_model_checkpoint.pt"),
                            )
                            shutil.copyfile(
                                os.path.join(save_loc, "optimizer_checkpoint.pt"),
                                os.path.join(save_loc, "best_optimizer_checkpoint.pt"),
                            )

                            # Also copy the per-rank EMA shards.  Each rank
                            # copies its own file; we issue a barrier after
                            # to synchronize.
                            ema_src = os.path.join(save_loc, f"ema_checkpoint_rank{self.rank:04d}.pt")
                            ema_dst = os.path.join(save_loc, f"best_ema_checkpoint_rank{self.rank:04d}.pt")
                            if os.path.exists(ema_src):
                                shutil.copyfile(ema_src, ema_dst)
                    _dist_barrier()

                    # In FSDP mode, every rank copies its own EMA shard
                    # (the copy above only ran on rank 0).  Do that now,
                    # outside the rank-0 block.
                    if conf["trainer"]["mode"] == "fsdp" and offset == 0 and conf["trainer"]["save_best_weights"]:
                        if hasattr(self, "ema") and getattr(self, "ema", None) is not None:
                            ema_src = os.path.join(save_loc, f"ema_checkpoint_rank{self.rank:04d}.pt")
                            ema_dst = os.path.join(save_loc, f"best_ema_checkpoint_rank{self.rank:04d}.pt")
                            if os.path.exists(ema_src) and self.rank != 0:
                                shutil.copyfile(ema_src, ema_dst)
                        _dist_barrier()

                if offset >= conf["trainer"]["stopping_patience"]:
                    logger.info("Best {} were in epoch {}; current epoch is {}; early stopping.".format(training_metric, best_epoch, epoch))
                    break

            if "stop_after_epoch" in conf["trainer"]:
                if conf["trainer"]["stop_after_epoch"]:
                    break

        best_epoch = [i for i, j in enumerate(results_dict[training_metric]) if j == direction(results_dict[training_metric])][0]
        result = {k: v[best_epoch] for k, v in results_dict.items()}

        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
            cleanup()

        return result
        