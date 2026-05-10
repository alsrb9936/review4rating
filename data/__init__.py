from .neumf_dataset import NeuMFDataset
from .deepconn_dataset import DeepCoNNDataset
from .narre_dataset import NARREDataset
from .rgcl_dataset import RGCLDataset
from .iard_rm_dataset import IARDRMDataset

DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
    "narre": NARREDataset,
    "rgcl": RGCLDataset,
    "iard_rm": IARDRMDataset,
}
