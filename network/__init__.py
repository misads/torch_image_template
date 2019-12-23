from .Model import Model
from .Model_corse import Model as CoarseModel

models = {
    'default': Model,
    'coarse': CoarseModel,
}