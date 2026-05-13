from .neumf_dataset import NeuMFDataset
from .deepconn_dataset import DeepCoNNDataset
from .narre_dataset import NARREDataset
from .ssg_dataset import SSGDataset
from .rgcl_dataset import RGCLDataset
from .iard_rm_dataset import IARDRMDataset
from .sgdn_dataset import SGDNDataset
from .daml_dataset import DAMLDataset

DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
    "narre": NARREDataset,
    "ssg": SSGDataset,
    "rgcl": RGCLDataset,
    "iard_rm": IARDRMDataset,
    "sgdn": SGDNDataset,
    "daml": DAMLDataset,
}
