from .neumf import NeuMF
from .deepconn import DeepCoNN
from .narre import NARRE
from .ssg import SSG
from .rgcl import RGCL
from .iard_rm import IARDRM


MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "narre": NARRE,
    "ssg": SSG,
    "rgcl": RGCL,
    "iard_rm": IARDRM,
}
