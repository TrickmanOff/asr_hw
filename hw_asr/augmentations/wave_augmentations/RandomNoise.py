import numpy as np
import torch
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class RandomNoiseBase(AugmentationBase):
    def __init__(self, max_noise_ampl: float = 5e-3):
        self.max_noise_ampl = max_noise_ampl

    def __call__(self, audio: Tensor) -> Tensor:
        noise_ampl = np.random.uniform(0, self.max_noise_ampl)
        noise = np.random.normal(size=audio.shape)
        noise /= np.abs(noise).max()  # [-1, 1]
        noised_audio = audio + noise_ampl / np.abs(audio).max() * noise
        return noised_audio.to(audio.dtype)


class RandomNoise(AugmentationBase):
    def __init__(self, p: float = 1., *args, **kwargs):
        self._aug = RandomApply(RandomNoiseBase(*args, **kwargs), p=p)

    def __call__(self, audio: Tensor) -> Tensor:
        return self._aug(audio)
