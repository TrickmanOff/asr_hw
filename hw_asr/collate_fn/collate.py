import logging
from typing import List

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


PADDING_VALUE = 0


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    # audio wave
    waves = [item["audio"] for item in dataset_items]
    waves_length = [item["audio"].shape[1] for item in dataset_items]
    result_batch["audio_length"] = waves_length

    max_wave_len = max(waves_length)
    padded_waves = [
        F.pad(wave, (0, max_wave_len - wave.shape[1]), value=PADDING_VALUE)
        for wave in waves
    ]
    concat_wave = torch.concat(padded_waves, dim=0)
    result_batch["audio"] = concat_wave

    # spectrogram
    specs = [item["spectrogram"].squeeze(0).T for item in dataset_items]  # each spectrogram - of shape (1, num_features, time_dim)
    specs_lens = torch.tensor([spec.shape[0] for spec in specs])
    result_batch["spectrogram_length"] = specs_lens

    concat_spec = pad_sequence(specs, batch_first=True, padding_value=PADDING_VALUE).transpose(-2, -1)
    result_batch["spectrogram"] = concat_spec

    # encoded text
    encoded_texts = [item["text_encoded"] for item in dataset_items]
    encoded_texts_lengths = torch.tensor([encoded_text.shape[1] for encoded_text in encoded_texts])
    result_batch["text_encoded_length"] = encoded_texts_lengths

    max_encoded_text_len = max(encoded_texts_lengths)
    padded_encoded_texts = [
        F.pad(encoded_text, (0, max_encoded_text_len - encoded_text.shape[1]), value=PADDING_VALUE)
        for encoded_text in encoded_texts
    ]
    concat_encoded_text = torch.concat(padded_encoded_texts, dim=0)
    result_batch["text_encoded"] = concat_encoded_text

    # text
    texts = [item["text"] for item in dataset_items]
    result_batch["text"] = texts

    # audio paths
    audio_paths = [item["audio_path"] for item in dataset_items]
    result_batch["audio_path"] = audio_paths

    return result_batch
