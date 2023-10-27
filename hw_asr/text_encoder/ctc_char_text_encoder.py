from collections import defaultdict
from typing import List, NamedTuple, DefaultDict

import torch

from .char_text_encoder import CharTextEncoder


class IndexHypothesis(NamedTuple):
    inds: List[int]
    prob: float


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.alphabet = vocab

    def ctc_decode(self, inds: List[int]) -> str:
        text = []
        prev_ind = None

        for i in range(len(inds) + 1):
            ind = inds[i] if i < len(inds) else None
            if ind != prev_ind:
                if prev_ind is not None:
                    prev_ch = self.ind2char[prev_ind]
                    if prev_ch != self.EMPTY_TOK:
                        text.append(prev_ch)
                prev_ind = ind
        return ''.join(text)

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        hypos: List[IndexHypothesis] = [IndexHypothesis([], 1.)]

        for ind_probs in probs[:probs_length]:
            hypos_probs = torch.tensor([hyp.prob for hyp in hypos]).to(device=probs.device)
            prod_probs = hypos_probs.outer(ind_probs)
            top_indices = prod_probs.flatten().topk(min(beam_size, prod_probs.numel())).indices
            top_indices = [(i.item() // len(ind_probs), i.item() % len(ind_probs))
                           for i in top_indices]
            hypos = [
                IndexHypothesis(hypos[hypo_index].inds + [new_ind],
                                prod_probs[hypo_index, new_ind].item())
                for hypo_index, new_ind in top_indices
            ]

        text_hypos_map: DefaultDict[str, float] = defaultdict(float)
        for hypo in hypos:
            text_hypos_map[self.ctc_decode(hypo.inds)] += hypo.prob

        text_hypos: List[Hypothesis] = [
            Hypothesis(text, prob) for text, prob in text_hypos_map.items()
        ]

        return sorted(text_hypos, key=lambda x: x.prob, reverse=True)
