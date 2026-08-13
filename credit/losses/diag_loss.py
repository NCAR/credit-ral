import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class DiagLoss(torch.nn.Module):
    def __init__(self, conf, validation=False):
        super(DiagLoss, self).__init__()

        self.conf = conf
        self.training_loss = conf["loss"]["training_loss"]
        self.vars = conf["data"]["diagnostic_variables"]

        self.lat_weights = None
        if conf["loss"]["use_latitude_weights"]:
            logger.info("Using latitude weights in loss calculations")
            self.lat_weights = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        self.var_weights = None
        if conf["loss"]["use_variable_weights"]:
            logger.info("Using variable weights in loss calculations")
            var_weights = [value if isinstance(value, list) else [value] for value in conf["loss"]["variable_weights"].values()]
            var_weights = np.array([item for sublist in var_weights for item in sublist])
            self.var_weights = torch.from_numpy(var_weights)

        # ------------------------------------------------------------- #
        self.validation = validation

        if self.validation:
            self.loss_fn = nn.L1Loss(reduction="none")
        else:
            self.loss_fn = nn.L1Loss(reduction="none")

    def forward(self, target, pred):
        # User defined loss
        loss = self.loss_fn(target, pred)

        # Latitutde and variable weights
        loss_dict = {}

        for i, var in enumerate(self.vars):
            var_loss = loss[:, i]

            if self.lat_weights is not None:
                var_loss = torch.mul(var_loss, self.lat_weights.to(target.device))

            if self.var_weights is not None:
                var_loss *= self.var_weights[i].to(target.device)

            loss_dict[f"loss_{var}"] = var_loss.mean()

        loss = torch.mean(torch.stack(list(loss_dict.values())))

        return loss
