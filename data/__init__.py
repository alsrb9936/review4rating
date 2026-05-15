from .neumf_dataset import NeuMFDataset
from .deepconn_dataset import DeepCoNNDataset
from .narre_dataset import NARREDataset
from .ssg_dataset import SSGDataset
from .rgcl_dataset import RGCLDataset
from .scg_rgcl_dataset import SCGRGCLDataset
from .ma_rgcl_dataset import MARGCLDataset
from .iard_rm_dataset import IARDRMDataset
from .iarm_rm_senti_dataset import IARMRMSentiDataset
from .sgdn_dataset import SGDNDataset
from .daml_dataset import DAMLDataset

DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
    "narre": NARREDataset,
    "ssg": SSGDataset,
    "rgcl": RGCLDataset,
    "scg_rgcl": SCGRGCLDataset,
    "ma_rgcl": MARGCLDataset,
    "iard_rm": IARDRMDataset,
    "iarm_rm_senti": IARMRMSentiDataset,
    "sgdn": SGDNDataset,
    "daml": DAMLDataset,
}
