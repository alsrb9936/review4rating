from .neumf_trainer import NeuMFTrainer
from .deepconn_trainer import DeepCoNNTrainer
from .narre_trainer import NARRETrainer
from .ssg_trainer import SSGTrainer
from .rgcl_trainer import RGCLTrainer
from .iard_rm_trainer import IARDRMTrainer
from .sgdn_trainer import SGDNTrainer

MODEL_TRAINER_DICT = {
    "neumf": NeuMFTrainer,
    "deepconn": DeepCoNNTrainer,
    "narre": NARRETrainer,
    "ssg": SSGTrainer,
    "rgcl": RGCLTrainer,
    "iard_rm": IARDRMTrainer,
    "sgdn": SGDNTrainer,
}
