from .neumf_trainer import NeuMFTrainer
from .deepconn_trainer import DeepCoNNTrainer
from .narre_trainer import NARRETrainer
from .rgcl_trainer import RGCLTrainer
from .iard_rm_trainer import IARDRMTrainer

MODEL_TRAINER_DICT = {
    "neumf": NeuMFTrainer,
    "deepconn": DeepCoNNTrainer,
    "narre": NARRETrainer,
    "rgcl": RGCLTrainer,
    "iard_rm": IARDRMTrainer,
}
