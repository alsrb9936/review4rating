from .neumf import NeuMF
from .deepconn import DeepCoNN
from .rgcl import RGCL
from .iard_rm import IARDRM


MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "rgcl": RGCL,
    "iard_rm": IARDRM,
}
