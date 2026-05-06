from .neumf_dataset import NeuMFDataset
from .deepconn_dataset import DeepCoNNDataset
from .rgcl_dataset import RGCLDataset
from .iard_rm_dataset import IARDRMDataset

DATASET_DICT = {
    "neumf": NeuMFDataset,
    "deepconn": DeepCoNNDataset,
    "rgcl": RGCLDataset,
    "iard_rm": IARDRMDataset,
}
