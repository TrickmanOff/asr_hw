import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class PitchShift(AugmentationBase):
    def __init__(self, p: float = 1., *args, **kwargs):
        self._aug = torch_audiomentations.PitchShift(*args, p=p, **kwargs)

    def __call__(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
