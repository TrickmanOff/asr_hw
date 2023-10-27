import gzip
import logging
import multiprocessing
import os
import shutil
from copy import copy
from pathlib import Path
from pyctcdecode import build_ctcdecoder
from typing import Callable, List

import kenlm
import numpy as np
from speechbrain.utils.data_utils import download_file

from .ctc_char_text_encoder import CTCCharTextEncoder


logger = logging.getLogger(__name__)

URL_LINKS = {
    'librispeech-vocab': 'https://www.openslr.org/resources/11/librispeech-vocab.txt',
    '3-gram.arpa': 'https://www.openslr.org/resources/11/3-gram.arpa.gz',
    '3-gram.pruned.1e-7.arpa': 'https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz',
    '3-gram.pruned.3e-7.arpa': 'https://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz',
}


class LibrispeechKenLMCTCCharTextEncoder(CTCCharTextEncoder):
    VOCAB_FILENAME = 'unigram_vocab.txt'

    def __init__(self, model='lowercase_3-gram.pruned.1e-7.arpa', models_dirpath: Path = Path('lms'),
                 alpha: float = 0.5, beta: float = 1.0, beam_width: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._models_dirpath = models_dirpath
        alphabet = copy(self.alphabet)
        alphabet = list('' if token == self.EMPTY_TOK else token for token in alphabet)

        model_path = self._get_or_load_model(model)
        unigrams = self._get_unigrams(str.lower)

        self.decoder = build_ctcdecoder(
            alphabet,
            str(model_path),
            unigrams,
            alpha=alpha,
            beta=beta
        )

        self.beam_width = beam_width

    def lm_ctc_decode(self, logits: np.ndarray) -> str:
        """
        logits of shape (seq_len, alphabet_size)
        """
        return self.decoder.decode(logits, beam_width=self.beam_width)

    def batch_lm_ctc_decode(self, logits_list: List[np.ndarray]) -> List[str]:
        with multiprocessing.get_context("fork").Pool(8) as pool:
            return self.decoder.decode_batch(pool, logits_list, beam_width=self.beam_width)

    def _get_unigrams(self, normalize_func: Callable[[str], str]) -> List[str]:
        vocab_filepath = self._models_dirpath / self.VOCAB_FILENAME
        if not os.path.exists(str(vocab_filepath)):
            self._load_vocabulary(vocab_filepath)
        with open(vocab_filepath) as f:
            unigram_list = [normalize_func(t) for t in f.read().strip().split("\n")]
        return unigram_list

    def _load_vocabulary(self, to_filepath: Path) -> Path:
        print(f"Loading vocabulary...")
        download_file(URL_LINKS['librispeech-vocab'], to_filepath)
        return to_filepath

    def _get_or_load_model(self, model_name: str) -> Path:
        if not os.path.exists(str(self._models_dirpath / model_name)):
            if model_name.startswith('lowercase'):
                init_model_name = model_name.removeprefix('lowercase_')
                init_model_path = self._get_or_load_model(init_model_name)
                to_model_path = self._models_dirpath / model_name
                self._convert_kenlm_model_tokens(str.lower, init_model_path, to_model_path)
            else:
                self._load_model(model_name)
        return self._models_dirpath / model_name

    def _load_model(self, model_name: str) -> Path:
        arch_path = self._models_dirpath / (model_name + '.gz')
        if not os.path.exists(arch_path):
            print(f"Loading LM {model_name}...")
            download_file(URL_LINKS[model_name], arch_path)
        with gzip.open(arch_path, 'rb') as file_in:
            with open(self._models_dirpath / model_name, 'wb') as file_out:
                shutil.copyfileobj(file_in, file_out)
        os.remove(str(arch_path))

    def _convert_kenlm_model_tokens(self, normalize_func: Callable[[str], str], from_model_path: str, to_model_path: str) -> None:
        print('Normalizing langugage model...')
        if os.path.exists(to_model_path):
            print('Already exists')
            return

        with open(from_model_path, 'r') as from_model:
            with open(to_model_path, 'w') as to_model:
                for line in from_model:
                    splitted = line.split('\t')
                    if len(splitted) >= 2:
                        splitted[1] = normalize_func(splitted[1])
                        to_model.write('\t'.join(splitted))
                    else:
                        to_model.write(line)
