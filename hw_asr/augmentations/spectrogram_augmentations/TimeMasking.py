import torchaudio
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, p: float = 1., *args, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(*args, p=p, **kwargs)

    def __call__(self, data: Tensor) -> Tensor:
        x = data.unsqueeze(0)
        return self._aug(x).squeeze()
