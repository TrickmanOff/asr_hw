import unittest
from typing import List, Tuple

import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder, Hypothesis


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def _check_beam_search_res(self, decoded: List[Hypothesis], expected: List[Tuple]):
        self.assertEqual(len(decoded), len(expected))
        for i, (text, prob) in enumerate(expected):
            self.assertEqual(decoded[i].text, text)
            self.assertAlmostEqual(decoded[i].prob, prob)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder()

        # initialize probabilities
        empty_token_ind = text_encoder.char2ind[text_encoder.EMPTY_TOK]
        a_ind = text_encoder.char2ind['a']
        b_ind = text_encoder.char2ind['b']

        probs = torch.zeros(2, len(text_encoder))

        probs[0][empty_token_ind] = 0.3
        probs[0][a_ind] = 0.2
        probs[0][b_ind] = 0.5

        probs[1][empty_token_ind] = 0.6
        probs[1][a_ind] = 0.4

        # beam_size = 1
        decoded = text_encoder.ctc_beam_search(probs, 2, beam_size=1)
        expected = [('b', 0.3)]
        self._check_beam_search_res(decoded, expected)

        # beam_size = 2
        decoded = text_encoder.ctc_beam_search(probs, 2, beam_size=2)
        expected = [('b', 0.3), ('ba', 0.2)]
        self._check_beam_search_res(decoded, expected)

        # beam_size = 3
        decoded = text_encoder.ctc_beam_search(probs, 2, beam_size=3)
        expected = [('b', 0.3), ('ba', 0.2), ('', 0.18)]
        self._check_beam_search_res(decoded, expected)
