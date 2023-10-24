import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, p: float = 1., *args, **kwargs):
        self._aug = torch_audiomentations.Gain(*args, p=p, **kwargs, output_type='tensor')

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
