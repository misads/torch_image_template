from .Model import Model
from .Model_corse import Model as CoarseModel
from .Model_corse_3 import Model as CoarseModel_3
from .Model_corse_3_pretrained import Model as CoarseModel_3_Pretrained

models = {
    'default': Model,
    'coarse': CoarseModel,
    'coarse_3': CoarseModel_3,
    'coarse_3_trans': CoarseModel_3,
    'coarse_3_pretrained': CoarseModel_3_Pretrained
}