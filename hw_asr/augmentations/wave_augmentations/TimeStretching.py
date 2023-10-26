import librosa
import numpy as np
import torch
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class TimeStretchingBase(AugmentationBase):
    def __init__(self, min_rate: float = 1., max_rate: float = 1.):
        assert min_rate <= max_rate
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, audio: Tensor) -> Tensor:
        rate = np.random.uniform(self.min_rate, self.max_rate)
        shifted_audio = librosa.effects.time_stretch(audio.numpy().squeeze(), rate=rate)
        shifted_audio = torch.from_numpy(shifted_audio)

        return shifted_audio


class TimeStretching(AugmentationBase):
    def __init__(self, p: float = 1., *args, **kwargs):
        self._aug = RandomApply(TimeStretchingBase(*args, **kwargs), p=p)

    def __call__(self, audio: Tensor) -> Tensor:
        return self._aug(audio)
