from .Model import Model
from .Model_corse import Model as CoarseModel
from .Model_corse_3 import Model as CoarseModel_3

models = {
    'default': Model,
    'coarse': CoarseModel,
    'coarse_3': CoarseModel_3,
    'coarse_3_trans': CoarseModel_3
}