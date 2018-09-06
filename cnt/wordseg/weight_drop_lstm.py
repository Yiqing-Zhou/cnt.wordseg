# Reference:
# https://github.com/salesforce/awd-lstm-lm/blob/master/weight_drop.py
# https://github.com/salesforce/awd-lstm-lm/issues/51
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
#
# Notes:
#
# 1. To make use of cuDNN, `flatten_parameters` must be called to flatten weights.
#
# 2. `flatten_parameters` reference many attrs of the module (lstm). This method will
# be called in `module.cuda` (actually, in `module._apply`).
#
# Idea of impl:
#
# 1. Hide the params in `__init__`, to which we want to apply DropConnect,
# by **renaming** (x -> x_raw). According to the Pytorch 0.4.1 impl, we need to hide
# the params with the pattern of `weight_hh_l{layer}{suffix}` (`layer` starts with 0;
# `suffix` is empty ("") in forward direction and "_reverse" in backward direction).
# Notice, the original param names still exist in self._all_weights, which will be
# accessed by `flatten_parameters` if `cuda` is called. Hence we need to hide the
# `flatten_parameters` as well, by replacing with a no op function.
#
# 2. Afterward, in every `forward` call, create a copy of the renamed params, fill with
# the original names, and apply dropout. As a consequence, the hideen params will be
# update during backprop. Notice, in other to use cuDNN, we need to call
# `flatten_parameters` before `module.forward`.
#
# 3. For optim, step (1) & (2) only needed to be executed once during inference.
# (optional)
#
from typing import List
import torch.nn as nn


def _no_op():
    return


def _weight_drop(module: nn.Module,
                 param_names: List[str],
                 dropout: float = 0.0) -> None:
    original_module_forward = module.forward
    original_module_flatten_parameters = module.flatten_parameters

    def _hidden_param_name(name):
        return name + '_raw'

    def _initialize():
        if isinstance(module, nn.RNNBase):
            module.flatten_parameters = _no_op

        for param_name in param_names:
            # delete original name.
            param = getattr(module, param_name)
            del module._parameters[param_name]
            # register to new name.
            module.register_parameter(
                _hidden_param_name(param_name), nn.Parameter(param.data),
            )

    def _forward(*args, **kwargs):
        # 1. prepare params.
        for param_name in param_names:
            # apply dropout.
            param = getattr(module, _hidden_param_name(param_name))
            dropout_param = nn.functional.dropout(
                param,
                p=dropout,
                training=module.training,
            )
            # inject back to module.
            setattr(module, param_name, dropout_param)

        # 2. flattening if necessary.
        if isinstance(module, nn.RNNBase):
            original_module_flatten_parameters()

        # 3. pass forward.
        return original_module_forward(*args, **kwargs)

    _initialize()
    setattr(module, 'forward', _forward)


class WeightDropoutLSTM(nn.LSTM):
    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        param_names = [
            'weight_hh_l{}{}'.format(
                layer,
                '_reverse' if direction == 1 else '',
            )
            for layer in range(self.num_layers)
            for direction in range(2 if self.bidirectional else 1)
        ]
        _weight_drop(self, param_names, dropout=weight_dropout)
