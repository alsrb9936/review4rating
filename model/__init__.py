from .neumf import NeuMF
from .deepconn import DeepCoNN
from .narre import NARRE
from .ssg import SSG
from .rgcl import RGCL
from .scg_rgcl import SCG_RGCL
from .ma_rgcl import MA_RGCL
from .iard_rm import IARDRM
from .iard_rm_gen import IARDRMGen
from .iarm_rm_senti import IARMRMSenti
from .sgdn import SGDN
from .daml import DAML


MODEL_DICT = {
    "neumf": NeuMF,
    "deepconn": DeepCoNN,
    "narre": NARRE,
    "ssg": SSG,
    "rgcl": RGCL,
    "scg_rgcl": SCG_RGCL,
    "ma_rgcl": MA_RGCL,
    "iard_rm": IARDRM,
    "iard_rm_gen": IARDRMGen,
    "iarm_rm_senti": IARMRMSenti,
    "sgdn": SGDN,
    "daml": DAML,
}
