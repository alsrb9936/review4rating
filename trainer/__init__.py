from .neumf_trainer import NeuMFTrainer
from .deepconn_trainer import DeepCoNNTrainer
from .narre_trainer import NARRETrainer
from .ssg_trainer import SSGTrainer
from .rgcl_trainer import RGCLTrainer
from .scg_rgcl_trainer import SCGRGCLTrainer
from .ma_rgcl_trainer import MARGCLTrainer
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
    "ma_rgcl": MARGCLTrainer,
    "iard_rm": IARDRMTrainer,
    "iard_rm_gen": IARDRMTrainer,
    "iarm_rm_senti": IARMRMSentiTrainer,
    "sgdn": SGDNTrainer,
    "daml": DAMLTrainer,
}
