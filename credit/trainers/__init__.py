import copy
import logging

from credit.trainers.trainerERA5 import Trainer as TrainerERA5
from credit.trainers.trainerCorrDiff import Trainer as TrainerCorrDiff
from credit.trainers.trainerLES import Trainer as TrainerLES
from credit.trainers.trainerWRF import Trainer as TrainerWRF
from credit.trainers.trainerWRF_multi import Trainer as TrainerWRF_Multi
from credit.trainers.trainerDscale import Trainer as TrainerDscale
from credit.trainers.trainerDiag import Trainer as TrainerDiag

logger = logging.getLogger(__name__)

# define trainer types
trainer_types = {
    "era5": (TrainerERA5, "Loading a single or multi-step trainer for the ERA5 dataset"),
    "standard-les": (TrainerLES, "Loading a single-step LES trainer"),
    "standard-wrf": (TrainerWRF, "Loading a single-step WRF trainer"),
    "multi-step-wrf": (TrainerWRF_Multi, "Loading a multi-step WRF trainer"),
    "standard-dscale": (TrainerDscale, "Loading a downscaling trainer"),
    "standard-diag": (TrainerDiag, "Loading a diagnostic model trainer"),
    "corrdiff": (TrainerCorrDiff, "Import CorrDiff Trainer")
}


def load_trainer(conf, load_weights=False):
    conf = copy.deepcopy(conf)
    trainer_conf = conf["trainer"]

    if "type" not in trainer_conf:
        msg = f"You need to specify a trainer 'type' in the config file. Choose from {list(trainer_types.keys())}"
        logger.warning(msg)
        raise ValueError(msg)

    trainer_type = trainer_conf.pop("type")

    if trainer_type in trainer_types:
        trainer, message = trainer_types[trainer_type]
        logger.info(message)
        return trainer
    else:
        msg = f"Trainer type {trainer_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)
