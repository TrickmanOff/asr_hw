import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class FrequencyMasking(AugmentationBase):
    def __init__(self, p: float = 1., *args, **kwargs):
        self._aug = RandomApply(torchaudio.transforms.FrequencyMasking(*args, **kwargs), p)

    def __call__(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(0)
        return self._aug(x).squeeze()
