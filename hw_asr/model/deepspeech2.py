from enum import Enum
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import torch
from torch import Tensor, nn

from hw_asr.base.base_model import BaseModel

# TODO: use masked BatchNorm


def _broadcast_to_lists(*args, target_length: Optional[int] = None) -> List:
    length = None
    for seq in args:
        if isinstance(seq, List):
            if length is not None:
                assert len(seq) == length
            else:
                length = len(seq)
    assert (length is not None) ^ (target_length is not None)
    if target_length is None:
        target_length = length
    return [seq if isinstance(seq, List) else [seq] * target_length for seq in args]


def _calc_padding_for_same(kernel_size: Tuple) -> Tuple:
    assert all(size % 2 == 1 for size in kernel_size)
    padding = tuple((size - 1) // 2 for size in kernel_size)
    return padding


class SpectrogramCNNBlock(nn.Module):
    def __init__(self,
                 output_channels: Union[int, List[int]],
                 filters: Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]]],
                 stride: Union[int, Tuple[int, int], List[Union[int, Tuple[int, int]]]] = 1,
                 activation_type=nn.ReLU,
                 num_of_layers: Optional[int] = None):
        """
        :param filters: if int or List[int], then 1D-convolution over time is applied
        (currently 1D-convolution is not supported)
        """
        super().__init__()

        output_channels, filters, stride = \
            _broadcast_to_lists(output_channels, filters, stride, target_length=num_of_layers)
        self._stride = stride

        input_channels = 1
        paddings = []
        layers: List[nn.Module] = []
        for in_channels, out_channels, kernel_size, stride in \
                zip([input_channels] + output_channels[:-1], output_channels, filters, stride):

            if isinstance(kernel_size, int):  # 1D
                raise NotImplementedError()
            else:  # 2D
                # input: # (batch_size, in_channels, num_features, time_frames)
                padding = _calc_padding_for_same(kernel_size=kernel_size)
                paddings.append(padding)
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                norm_layer = nn.BatchNorm2d(num_features=out_channels)
                activation = activation_type()
            layers += [conv_layer, norm_layer, activation]

        self._padding = paddings
        self._output_channels_dim = layers[-3].out_channels
        self._time_dim_reduction_times = np.prod([stride[1] for stride in self._stride]).item()
        self.block = nn.Sequential(*layers)

    def forward(self, spectrogram_batch: Tensor) -> Tensor:
        """
        input of shape  (batch_dim, num_features, time_dim)
        output of shape (batch_dim, output_channels_dim, num_features, time_dim)
        """
        input = spectrogram_batch.unsqueeze(1)
        output = self.block(input)
        return output

    def calc_output_features_dim(self, in_features: int) -> int:
        stride_prod = np.prod([stride[0] for stride in self._stride])
        assert in_features % stride_prod == 0
        return in_features // stride_prod

    def transform_input_lengths(self, input_lengths):
        for stride in self._stride:
            input_lengths = torch.div(input_lengths + stride[1] - 1, stride[1], rounding_mode='trunc')
        return input_lengths

    @property
    def output_channels_dim(self) -> int:
        return self._output_channels_dim

    @property
    def time_dim_reduction_times(self) -> int:
        return self._time_dim_reduction_times


class GRUBlock(nn.Module):
    def __init__(self,
                 in_num_features: int,
                 hidden_dims: Union[int, List[int]],
                 num_of_layers: Optional[int] = None,
                 use_batch_norm: bool = True):
        super().__init__()

        hidden_dims, *_ = _broadcast_to_lists(hidden_dims, target_length=num_of_layers)

        rnn_layers: List[nn.Module] = []
        bn_layers: List[nn.Module] = []
        for in_features, hidden_dim in zip([in_num_features] + hidden_dims[:-1], hidden_dims):
            rnn_layer = nn.GRU(input_size=in_features, hidden_size=hidden_dim, batch_first=True)
            rnn_layers.append(rnn_layer)
            if use_batch_norm:
                bn_layer = nn.BatchNorm1d(hidden_dim)
                bn_layers.append(bn_layer)
        if not use_batch_norm:
            bn_layers = [None] * len(rnn_layers)

        self._out_num_features = hidden_dims[-1]
        self.rnn_layers = nn.ModuleList(rnn_layers)
        self.bn_layers = nn.ModuleList(bn_layers)

    def forward(self, input: Tensor) -> Tensor:
        """
        input of shape  (batch_dim, in_num_features, time_dim)
        output of shape (batch_dim, out_num_features, time_dim)
        TODO: use `torch.nn.utils.rnn.pack_padded_sequence()`
        """
        # input shape for RNN block: (batch_dim, time_dim, num_features)
        prev_output = input
        for rnn_layer, bn_layer in zip(self.rnn_layers, self.bn_layers):
            prev_output = rnn_layer(prev_output.transpose(-2, -1))[0]  # (batch_dim, time_dim, hidden_dim)
            if bn_layer is not None:
                prev_output = bn_layer(prev_output.transpose(-2, -1))
        return prev_output

    @property
    def out_num_features(self) -> int:
        return self._out_num_features


class LinearBlock(nn.Module):
    def __init__(self,
                 input_num_features: int,
                 output_num_features: Union[int, List[int]],
                 num_of_layers: Optional[int] = None,
                 use_batch_norm: bool = False):
        super().__init__()

        assert use_batch_norm is False, 'Not implemented'

        output_num_features, *_ = _broadcast_to_lists(output_num_features, num_of_layers)

        layers: List[nn.Module] = []
        for in_features, out_features in \
                zip([input_num_features] + output_num_features[:-1], output_num_features):
            layers.append(nn.Linear(in_features, out_features))

    def forward(self, input: Tensor) -> Tensor:
        """
        input of shape (batch_dim, num_features, time_dim)
        output of shape (batch_dim, num_features, time_dim)
        """
        pass


class Lookahead(nn.Module):
    def __init__(self, in_features: int, future_time: int):
        super().__init__()
        self._future_time = future_time
        self.conv = nn.Conv1d(in_features, in_features, kernel_size=future_time,
                              padding=future_time-1, groups=in_features, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        """
        input of shape  (batch_dim, num_features, time_dim)
        output of shape (batch_dim, num_features, time_dim)
        """
        output = self.conv(input)
        return output[:, :, self._future_time-1:]


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, cnn_config: Dict, gru_config: Dict, lookahead_config: Dict):
        super().__init__(n_feats, n_class)
        self.cnn_block = SpectrogramCNNBlock(**cnn_config)
        cnn_block_output_features_dim = self.cnn_block.calc_output_features_dim(n_feats)
        num_features_after_cnn = cnn_block_output_features_dim * self.cnn_block.output_channels_dim
        self.gru_block = GRUBlock(num_features_after_cnn, **gru_config)
        self.lookahead = Lookahead(self.gru_block.out_num_features, **lookahead_config)
        self.linear = nn.Linear(self.gru_block.out_num_features, n_class)

    def forward(self, spectrogram, **batch) -> Union[Tensor, dict]:
        """
        input shape:  (batch_dim, n_feats, time_dim)
        output shape: (batch_dim, time_dim, n_class)
        """
        cnn_output = self.cnn_block(spectrogram)  # (batch_dim, output_channels_dim, n_feats, time_dim)
        cnn_concat_channels = cnn_output.flatten(-3, -2)  # (batch_dim, output_channels_dim * cnn_block_output_features_dim, time_dim)
        gru_output = self.gru_block(cnn_concat_channels)  # (batch_dim, gru_out_num_features, time_dim)
        gru_output_with_lookahead = self.lookahead(gru_output).transpose(-2, -1)  # (batch_dim, time_dim, gru_out_num_features)
        logits = self.linear(gru_output_with_lookahead)  # (batch_dim, time_dim, n_class)
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        return self.cnn_block.transform_input_lengths(input_lengths)
