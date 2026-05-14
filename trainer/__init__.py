from .neumf_trainer import NeuMFTrainer
from .deepconn_trainer import DeepCoNNTrainer
from .narre_trainer import NARRETrainer
from .ssg_trainer import SSGTrainer
from .rgcl_trainer import RGCLTrainer
from .scg_rgcl_trainer import SCGRGCLTrainer
from .iard_rm_trainer import IARDRMTrainer
from .iarm_rm_senti_trainer import IARMRMSentiTrainer
from .sgdn_trainer import SGDNTrainer
from .daml_trainer import DAMLTrainer

MODEL_TRAINER_DICT = {
    "neumf": NeuMFTrainer,
    "deepconn": DeepCoNNTrainer,
    "narre": NARRETrainer,
    "ssg": SSGTrainer,
    "rgcl": RGCLTrainer,
    "scg_rgcl": SCGRGCLTrainer,
    "iard_rm": IARDRMTrainer,
    "iarm_rm_senti": IARMRMSentiTrainer,
    "sgdn": SGDNTrainer,
    "daml": DAMLTrainer,
}
