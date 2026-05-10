from .neumf import NeuMF
from .deepconn import DeepCoNN
from .narre import NARRE
from .rgcl import RGCL
from .iard_rm import IARDRM


MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "narre": NARRE,
    "rgcl": RGCL,
    "iard_rm": IARDRM,
}
